import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding,
    PreTrainedTokenizer
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from datasets import Dataset, DatasetDict
from typing import Sequence

# Optional LoRA via PEFT
try:
    from peft import LoraConfig, get_peft_model
    _PEFT_AVAILABLE = True
except Exception:
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    _PEFT_AVAILABLE = False

class AutoregressiveFudgeClassifier(nn.Module):
    def __init__(
        self, 
        model_name: str = 'Qwen/Qwen2.5-0.5B',
        num_labels: int = 2,
        dropout: float = 0.1,
        num_layers_to_train: int = 2,
        # LoRA controls
        lora_enabled: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] | None = None
    ) -> None:
        super().__init__()
        # Load the base autoregressive model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="sdpa",
            trust_remote_code=True
        ).base_model  # Get the base model without LM head

        # Set defaults for LoRA target modules (Qwen2-like)
        if lora_target_modules is None:
            lora_target_modules = (
                "q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"
            )

        if lora_enabled:
            if not _PEFT_AVAILABLE:
                raise ImportError("peft is required for LoRA. Install with: pip install peft")
            # Train only adapters
            for param in self.base_model.parameters():
                param.requires_grad = False
            peft_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=list(lora_target_modules),
                task_type="CAUSAL_LM",
            )
            self.base_model = get_peft_model(self.base_model, peft_cfg)
            self._lora_config_dict = {
                "enabled": True,
                "r": lora_r,
                "alpha": lora_alpha,
                "dropout": lora_dropout,
                "target_modules": list(lora_target_modules),
            }
        else:
            # Freeze all parameters first
            for param in self.base_model.parameters():
                param.requires_grad = False
            # Unfreeze last n transformer layers
            for layer in self.base_model.layers[-num_layers_to_train:]:
                for param in layer.parameters():
                    param.requires_grad = True
            self._lora_config_dict = {"enabled": False}
        
        hidden_size = self.base_model.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor | None = None, 
        labels: torch.Tensor | None = None,
        **_: dict
    ) -> dict[str, torch.Tensor]:
        # Get model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get the last hidden state for classification
        last_hidden_state = outputs.last_hidden_state[:, -1, :]
        
        # Get logits from classification head
        logits = self.classifier(last_hidden_state)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            loss = loss_fct(logits, labels)
            
        return {
            "loss": loss, 
            "logits": logits,
           # "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None
        }

def train_fudge_model(
    dataset: DatasetDict, 
    model_output_dir: str,
    label_column: str,
    label_mapping: dict[str, int],
    num_labels: int | None = None,
    epochs: int = 10,
    seed: int = 42,
    model_name: str = 'Qwen/Qwen2.5-0.5B',
    # LoRA controls
    lora_enabled: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Sequence[str] | None = None
) -> tuple[AutoregressiveFudgeClassifier, PreTrainedTokenizer]:
    """
    Train the FUDGE model using an autoregressive model as the base.
    
    Args:
        dataset: Dataset containing training and validation data
        model_output_dir: Directory to save the model
        label_column: Name of the column containing labels
        label_mapping: Dictionary mapping class names to indices
        num_labels: Number of classes (if None, will be inferred from label_mapping)
        epochs: Number of training epochs
        seed: Random seed
        model_name: Name of the base model to use
    """
    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model = AutoregressiveFudgeClassifier(
        model_name=model_name,
        num_labels=num_labels,
        lora_enabled=lora_enabled,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules
    )
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Model device: {next(model.parameters()).device}")

    # Tokenize dataset with proper label handling
    def tokenize_function(examples: dict[str, str]) -> dict[str, torch.Tensor]:
        # Tokenize the text inputs
        tokenized_inputs = tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Convert labels to tensor and ensure proper shape
        labels = torch.tensor(examples[label_column])
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
            
        # Add labels to tokenized inputs
        tokenized_inputs['labels'] = labels
        
        return tokenized_inputs

    print("Tokenizing dataset...")
    # Process and verify dataset structure
    tokenized_data = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset['train'].column_names,  # Remove all original columns
        desc="Tokenizing datasets"
    )

    tokenized_data.set_format(
        type='torch', 
        columns=['input_ids', 'attention_mask', 'labels']
    )

    def compute_metrics(eval_pred):
        """Compute classification metrics"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, 
            predictions, 
            average='weighted'
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        group_by_length=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=0.1,
        remove_unused_columns=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['validation'],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # Train and save
    print("Starting training...")
    trainer.train()

    # Save model and tokenizer
    import os
    os.makedirs(model_output_dir, exist_ok=True)
    
    final_model_path = os.path.join(model_output_dir, "fudge_classifier.pt")
    print(f"Saving model to {final_model_path}")
    torch.save({
        'state_dict': model.state_dict(),
        'base_model_name': model_name,
        'num_labels': num_labels,
        'label_mapping': label_mapping,
        'lora_config': getattr(model, '_lora_config_dict', {"enabled": False})
    }, final_model_path)
    
    tokenizer.save_pretrained(model_output_dir)
    
    # Clean up checkpoints
    import shutil
    import glob
    checkpoint_dirs = glob.glob(f"{model_output_dir}/checkpoint-*")
    for dir in checkpoint_dirs:
        shutil.rmtree(dir)
    
    return model, tokenizer