import os
from glob import glob
import torch
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    Trainer, TrainingArguments
)
from datasets import Dataset, Value, ClassLabel, Features
import pandas as pd
from pysentimiento.preprocessing import preprocess_tweet
import fire
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

def get_lang(file):
    return os.path.splitext(os.path.basename(file))[0]

"""
Lo pongo así por huggingface
"""
id2label = {0: 'N', 1: 'NEU', 2: 'P'}
label2id = {v:k for k,v in id2label.items()}

def load_df(file):
    #dialect = get_lang(file)

    df = pd.read_table(file, names=["id", "text", "polarity"], index_col=0)
    #df["dialect"] = dialect

    for label, idx in label2id.items():
        df.loc[df["polarity"] == label, "label"] = idx
    return df




def load_datasets():
    """
    Return train, dev, test datasets
    """
    train_files = glob("data/tass2020/train/*.tsv")
    dev_files = glob("data/tass2020/dev/*.tsv")
    test_files = glob("data/tass2020/test1.1/*.tsv")

    train_dfs = {get_lang(file):load_df(file) for file in train_files}
    dev_dfs = {get_lang(file):load_df(file) for file in dev_files}
    test_dfs = {get_lang(file):load_df(file) for file in test_files}

    train_df = pd.concat(train_dfs.values())
    dev_df = pd.concat(dev_dfs.values())
    test_df = pd.concat(test_dfs.values())

    print(len(train_df), len(dev_df), len(test_df))


    train_df["text"] = train_df["text"].apply(preprocess_tweet)
    dev_df["text"] = dev_df["text"].apply(preprocess_tweet)
    test_df["text"] = test_df["text"].apply(preprocess_tweet)

    features = Features({
        'text': Value('string'),
        'label': ClassLabel(num_classes=3, names=["neg", "neu", "pos"])
    })

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]], features=features)
    dev_dataset = Dataset.from_pandas(dev_df[["text", "label"]], features=features)
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]], features=features)

    return train_dataset, dev_dataset, test_dataset

def load_model(base_model, device):
    """
    Loads model and tokenizer
    """
    print(f"Loading model {base_model}")
    model = BertForSequenceClassification.from_pretrained(base_model, return_dict=True, num_labels=3)

    tokenizer = BertTokenizer.from_pretrained(base_model)
    tokenizer.model_max_length = 128

    model.config.hidden_dropout_prob = 0.20
    model.config.id2label = id2label
    model.config.label2id = label2id

    model = model.to(device)
    model.train()

    return model, tokenizer


def run_sentiment_experiments(base_model, times=5, epochs=5):
    """
    """
    print("Loading dataset")
    train_dataset, dev_dataset, test_dataset = load_datasets()

    device = "cuda" if torch.cuda.is_available() else "cpu"


    model, tokenizer = load_model(base_model, device)

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    batch_size = 64
    eval_batch_size = 16

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


    epochs = 5

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
    )

    results = []

    for run in range(times):
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

        results.append({
            **trainer.evaluate(),
            **{"run": run+1}
        })

    f1_scores = torch.Tensor([r["eval_f1"] for r in results])
    print(f"Macro F1: {f1_scores.mean():.3f} +- {f1_scores.std():.3f}")


if __name__ == "__main__":
    fire.Fire(run_sentiment_experiments)