# Fine-tune a model for defamation detection
# Using a small pretrained model suitable for sequence classification

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
import torch
from datasets import Dataset
import evaluate
import wandb
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Fine-tune a model for defamation detection')
parser.add_argument('--wandb_api_key', type=str, default=None, help='Weights & Biases API key')
parser.add_argument('--model_name', type=str, default="roberta-large", 
                    help='Pretrained model to fine-tune')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
args = parser.parse_args()

# Configure Weights & Biases logging if API key is provided
if args.wandb_api_key:
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.environ["WANDB_PROJECT"] = "defamation-detection"
    wandb.login()
else:
    # Disable wandb
    os.environ["WANDB_DISABLED"] = "true"

# Load the dataset
try:
    df = pd.read_csv("defamation_dataset.csv")
    print(f"Loaded dataset with {len(df)} examples")
except FileNotFoundError:
    print("Dataset file not found. Make sure defamation_dataset.csv exists in the current directory.")
    exit(1)

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Convert to Hugging Face dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load tokenizer and model
model_name = args.model_name
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2  # Binary classification (defamatory or not)
)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Evaluation metric (accuracy and F1 score)
metric = evaluate.combine(["accuracy", "f1"])

# Compute metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    report_to="wandb" if args.wandb_api_key else "none",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train the model
print("Starting training...")
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the fine-tuned model
model_save_path = f"defamation-detector-model_{model_name}"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

# Display model details and statistics
print("\nTraining complete!")
print(f"Model: {model_name}")
print(f"Dataset size: {len(df)} examples")
print(f"Training examples: {len(train_df)}")
print(f"Validation examples: {len(val_df)}")
print(f"Final validation accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Final validation F1 score: {eval_results['eval_f1']:.4f}")
