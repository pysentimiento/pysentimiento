import fire
import torch
from glob import glob
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    Trainer, TrainingArguments, set_seed
)
import pandas as pd
from pysentimiento.tass import load_datasets, id2label, label2id, load_model
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(pred):
    """
    Compute metrics for Trainer
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }





def run_sentiment_experiments(
    base_model, times=5, epochs=5, batch_size=64, eval_batch_size=16
):
    """
    """
    print("Loading dataset")
    train_dataset, dev_dataset, test_dataset = load_datasets()

    device = "cuda" if torch.cuda.is_available() else "cpu"


    model, tokenizer = load_model(base_model, device)

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

    results = []

    for run in range(times):
        set_seed(2020 + run)
        print("="*80)
        print(f"Run {run+1}")
        model, _ = load_model(base_model, device)

        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
        )

        trainer.train()
        run_results = trainer.evaluate()
        run_results["run"] = run + 1
        results.append(run_results)

        print(f"Macro F1: {run_results['eval_f1']}")

        f1_scores = torch.Tensor([r["eval_f1"] for r in results])
        print(f"Results so far -- Macro F1: {f1_scores.mean():.3f} +- {f1_scores.std():.3f}")

    f1_scores = torch.Tensor([r["eval_f1"] for r in results])
    print(f"Macro F1: {f1_scores.mean():.3f} +- {f1_scores.std():.3f}")


if __name__ == "__main__":
    fire.Fire(run_sentiment_experiments)
