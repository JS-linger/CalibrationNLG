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

class AutoregressiveFudgeClassifier(nn.Module):
    def __init__(
        self, 
        model_name: str = 'Qwen/Qwen1.5-0.5B',
        num_labels: int = 2,
        dropout: float = 0.1,
        num_layers_to_train: int = 2
    ) -> None:
        super().__init__()
        # Load the base autoregressive model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="sdpa",
            trust_remote_code=True
        ).base_model  # Get the base model without LM head

        # Freeze all parameters first
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze last n transformer layers
        for layer in self.base_model.layers[-num_layers_to_train:]:
            for param in layer.parameters():
                param.requires_grad = True
        
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
        labels: torch.Tensor | None = None
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
    num_labels: int | None = None,
    epochs: int = 10,
    seed: int = 42,
    model_name: str = 'Qwen/Qwen1.5-0.5B'
) -> tuple[AutoregressiveFudgeClassifier, PreTrainedTokenizer]:
    """
    Train the FUDGE model using an autoregressive model as the base.
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
        num_labels=num_labels
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
        evaluation_strategy="steps",     # Evaluate du ring training
        eval_steps=0.1,                  # Evaluate every 100 steps
        save_strategy="steps",           # Save based on evaluation
        save_steps=0.1,                  # Save every 100 steps
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=0.1,
        fp16=True,
        group_by_length=True,
        load_best_model_at_end=True,
        save_total_limit=1,              # Keep only the best model
        metric_for_best_model="eval_loss",  # Use validation loss for best model
        greater_is_better=False,         # Lower loss is better
        warmup_ratio=0.1,
        remove_unused_columns=True
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
        'num_labels': num_labels
    }, final_model_path)
    
    tokenizer.save_pretrained(model_output_dir)
    
    # Clean up checkpoints
    import shutil
    import glob
    checkpoint_dirs = glob.glob(f"{model_output_dir}/checkpoint-*")
    for dir in checkpoint_dirs:
        shutil.rmtree(dir)
    
    return model, tokenizer