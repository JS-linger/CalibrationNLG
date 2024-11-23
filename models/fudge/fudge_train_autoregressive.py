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
from datasets import Dataset, DatasetDict

class AutoregressiveFudgeClassifier(nn.Module):
    def __init__(
        self, 
        model_name: str = 'Qwen/Qwen1.5-0.5B',
        num_labels: int = 2,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        # Load the base autoregressive model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True
        ).base_model  # Get the base model without LM head
        
        hidden_size = self.base_model.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
    
    # Add gradient checkpointing
    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)
        
    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()

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
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def train_fudge_model(
    dataset: DatasetDict, 
    model_output_dir: str,
    label_column: str,
    num_labels: int | None = None,
    epochs: int = 3,
    test_size: float = 0.2,
    seed: int = 42,
    model_name: str = 'Qwen/Qwen1.5-0.5B'
) -> tuple[AutoregressiveFudgeClassifier, PreTrainedTokenizer]:
    """
    Train the FUDGE model using an autoregressive model as the base.
    
    Args:
        dataset: HuggingFace dataset dictionary containing train and validation splits
        model_output_dir: Directory to save the model
        label_column: Name of the label column
        num_labels: Number of classification labels
        epochs: Number of training epochs
        test_size: Fraction of data for validation
        seed: Random seed
        model_name: Name of the base model
    
    Returns:
        Tuple of (trained model, tokenizer)
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize the model
    model = AutoregressiveFudgeClassifier(
        model_name=model_name,
        num_labels=num_labels
    )
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Model device: {next(model.parameters()).device}")

    # Tokenize the dataset
    def tokenize_function(examples: dict[str, str]) -> dict[str, torch.Tensor]:
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )

    print("Tokenizing dataset...")
    tokenized_data = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['text']
    )

    tokenized_data.set_format(
        type='torch', 
        columns=['input_ids', 'attention_mask', 'labels']
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        fp16=True,
        group_by_length=True,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        warmup_steps=500,
        gradient_checkpointing=True
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['validation'],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the model
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    
    return model, tokenizer