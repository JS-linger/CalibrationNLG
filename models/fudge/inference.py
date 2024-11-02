import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

class FudgeInference:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        fudge_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize FUDGE inference with pre-loaded models.
        
        Args:
            model: HuggingFace language model
            tokenizer: HuggingFace tokenizer
            fudge_model_path: Path to trained FUDGE model
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.fudge_model = torch.load(fudge_model_path).to(device)
        self.device = device

    def get_next_token_distribution(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Get next token distribution from base model with sampling methods."""
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :] / temperature
            
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            return torch.softmax(logits, dim=-1)

    def generate_with_fudge(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        fudge_temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        lambda_weight: float = 0.5
    ) -> str:
        """
        Generate text using the base model guided by FUDGE predictions.
        
        Args:
            prompt: Initial text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature for base model
            fudge_temperature: Temperature for FUDGE logits
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            lambda_weight: Weight for combining base and FUDGE distributions
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = input_ids.clone()

        for _ in range(max_length):
            # Get base model distribution
            base_distribution = self.get_next_token_distribution(
                generated_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Get FUDGE predictions
            with torch.no_grad():
                fudge_logits = self.fudge_model(generated_tokens).logits
                fudge_distribution = torch.softmax(fudge_logits / fudge_temperature, dim=-1)

            # Combine distributions
            combined_distribution = (
                (1 - lambda_weight) * base_distribution +
                lambda_weight * fudge_distribution
            )

            # Sample next token
            next_token = torch.multinomial(combined_distribution, num_samples=1)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

            # Stop if EOS token is generated
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)