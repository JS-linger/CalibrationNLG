from typing import Optional, Tuple, Union
from dataclasses import dataclass
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoConfig
)
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    ModelOutput
)
from transformers.utils import get_device_map
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import numpy as np

def log1mexp(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0."""
    mask = -np.log(2) < x
    x = torch.clamp_max(x, -eps)
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )

@dataclass
class CausalNADOOutputWithCrossAttentions(ModelOutput):
    """Base class for causal language model outputs with DiNADO additions."""
    loss: Optional[torch.FloatTensor] = None
    reg_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

class DiNADOMergeLMHeadModel(PreTrainedModel):
    def __init__(self, config, base_model=None):
        super().__init__(config)
        self.base_model = base_model or AutoModelForCausalLM.from_config(config)
        
        # DiNADO-specific head
        self.norm_prediction_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, 1, bias=False)
        )
        
        # Model parallel settings
        self.model_parallel = False
        self.device_map = None
        
        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        reference_model: Optional[PreTrainedModel] = None,
    ) -> Union[Tuple, CausalNADOOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # We need hidden states
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]
        lm_logits = outputs.logits

        reg_loss = None
        class_loss = None
        
        if labels is not None and reference_model is not None:
            # Move labels to correct device
            labels = labels.to(lm_logits.device)
            
            # Shift for next token prediction
            shift_logits_policy = lm_logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Get reference model outputs
            with torch.no_grad():
                ref_outputs = reference_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                ref_hidden_states = ref_outputs.hidden_states[-1]
                
            # Calculate DiNADO normalization factors
            betas = torch.nn.functional.logsigmoid(self.norm_prediction_head(hidden_states))
            
            # Get policy and reference distributions
            r_policy = shift_logits_policy.log_softmax(dim=-1).clamp(-70., 0)
            r_reference = ref_outputs.logits[..., :-1, :].log_softmax(dim=-1).clamp(-70., 0)
            
            # Calculate normalized ratios
            r = (r_policy - r_reference).log_softmax(dim=-1)
            r = r - r.amax(dim=-1, keepdim=True)
            log_R = r + betas
            
            # Get token-specific ratios
            log_Ri = log_R.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            log_1mRi = log1mexp(log_Ri)
            
            # Calculate forward-looking ratios
            log_Ri_one_step_forward = (log_R + r_reference).logsumexp(dim=-1)
            log_1mRi_one_step_forward = log1mexp(log_Ri_one_step_forward)
            Ri_one_step_forward = log_Ri_one_step_forward.exp()
            
            # Create masks for loss calculation
            token_mask = (shift_labels != self.config.pad_token_id).float()
            last_token_mask = torch.zeros_like(token_mask)
            last_token_mask[torch.arange(token_mask.size(0)), token_mask.sum(dim=-1).long() - 1] = 1
            token_mask = token_mask * (1 - last_token_mask)
            
            # Calculate losses
            reg_loss = -Ri_one_step_forward * (log_Ri - log_Ri_one_step_forward) \
                      -(1. - Ri_one_step_forward) * (log_1mRi - log_1mRi_one_step_forward)
            reg_loss = (reg_loss * token_mask).sum() / token_mask.sum()
            
            class_loss = -labels.unsqueeze(-1) * log_Ri - (1. - labels).unsqueeze(-1) * log_1mRi
            class_loss = (class_loss * last_token_mask).sum() / last_token_mask.sum()

        if not return_dict:
            output = (lm_logits,) + outputs[2:]
            return ((class_loss, reg_loss,) + output) if class_loss is not None else output

        return CausalNADOOutputWithCrossAttentions(
            loss=class_loss,
            reg_loss=reg_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }