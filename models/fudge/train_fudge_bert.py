import torch
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding,
    PreTrainedTokenizer
)
from datasets import Dataset, DatasetDict

def train_fudge_model(
    dataset: DatasetDict, 
    model_output_dir: str, 
    label_column: str, 
    num_labels: int | None = None, 
    epochs: int = 3, 
    test_size: float = 0.2, 
    seed: int = 42
) -> tuple[BertForSequenceClassification, PreTrainedTokenizer]:
    """
    Train the FUDGE model using a BERT base uncased classifier on the prefix enumerated data.

    Args:
        dataset: Hugging Face Dataset containing the preprocessed data
        model_output_dir: Directory to save the fine-tuned model
        label_column: The name of the label column to use for training
        num_labels: Number of output labels for the classifier
        epochs: Number of training epochs
        test_size: Fraction of the dataset to use as validation data
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (trained model, tokenizer)
    """

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=num_labels
    )
    model.to('cuda')  # Use GPU for training
    print(f"Model device: {next(model.parameters()).device}")

    # Freeze the lower 6 layers of the BERT model (BERT-base has 12 layers in total)
    for layer in model.bert.encoder.layer[:6]:
        for param in layer.parameters():
            param.requires_grad = False

    # Print to verify that the lower layers are frozen
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Frozen layer: {name}")
        else:
            print(f"Tunable layer: {name}")

    # Tokenize the dataset
    def tokenize_function(examples: dict[str, str]) -> dict[str, torch.Tensor]:
        tokenized_inputs = tokenizer(examples['prefixes'], truncation=True)
        tokenized_inputs['labels'] = examples['labels']
        return tokenized_inputs

    print("Parallelizing tokenization with 4 processes")
    tokenized_data = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4
    )

    # Set the format to PyTorch tensors
    tokenized_data.set_format("torch")

    # Use DataCollatorWithPadding to apply dynamic padding during batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=0.01,
        fp16=True,
        group_by_length=True
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    
    return model, tokenizer