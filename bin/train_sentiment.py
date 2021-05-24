import fire
import torch
from glob import glob
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    Trainer, TrainingArguments, set_seed
)
import pandas as pd
from pysentimiento import compute_metrics
from pysentimiento.tass import (
    load_datasets, load_model,
)


def train(
    base_model, output_path, epochs=5, batch_size=32, eval_batch_size=16
):
    """
    """
    print("Loading dataset")
    train_dataset, dev_dataset, test_dataset = load_datasets()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model(base_model)

    model = model.to(device)
    model.train()

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)


    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=batch_size)
    dev_dataset = dev_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)


    def format_dataset(dataset):
        dataset = dataset.map(lambda examples: {'labels': examples['label']})
        dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        return dataset

    train_dataset = format_dataset(train_dataset)
    dev_dataset = format_dataset(dev_dataset)
    test_dataset = format_dataset(test_dataset)


    total_steps = (epochs * len(train_dataset)) // batch_size
    warmup_steps = total_steps // 10
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        do_eval=False,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )

    trainer.train()

    test_results = trainer.evaluate(test_dataset)
    print("\n\n")
    print("Test results")
    print("============")
    for k, v in test_results.items():
        print(f"{k:<16} : {v:.3f}")

    path = "../models/beto-sentiment-analysis"

    print(f"Saving model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    fire.Fire(train)
