import json
import sys
import fire
import torch
from pysentimiento.emotion.datasets import load_datasets
from glob import glob
from transformers import (
    Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
)
import pandas as pd
from pysentimiento import compute_metrics
from pysentimiento.tass import (
    load_datasets as load_tass_datasets, id2label as id2labeltass, label2id as label2idtass,
    load_model,
)

from pysentimiento.semeval import (
    load_datasets as load_semeval_datasets,
    id2label as id2labelsemeval, label2id as label2idsemeval
)



lang_conf = {
    "es": {
        "load_datasets": load_tass_datasets,
        "id2label": id2labeltass,
        "label2id": label2idtass,
    },
    "en": {
        "load_datasets": load_semeval_datasets,
        "id2label": id2labelsemeval,
        "label2id": label2idsemeval,
    }
}

extra_args = {
    "BertweetTokenizer": {
        "preprocessing_args": {"user_token": "@USER", "url_token": "HTTPURL"}
    }
}



def eval_sentiment(
    model_name, output_path, lang="es", eval_batch_size=16, warmup_proportion=0.1, limit=None,
):
    """
    """
    print("="*80 + '\n', "="*80 + '\n')
    print(f"Evaluating {model_name} in language {lang}", "\n" * 2)
    print("Loading dataset")
    if lang not in lang_conf.keys():
        print("lang must be one of ", lang_conf.keys())
        sys.exit(1)

    load_datasets = lang_conf[lang]["load_datasets"]
    id2label = lang_conf[lang]["id2label"]
    label2id = lang_conf[lang]["label2id"]

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 128
    model.eval()

    tokenizer_class_name = model.config.tokenizer_class

    load_extra_args = extra_args[tokenizer_class_name] if tokenizer_class_name in extra_args else {}

    _, _, test_dataset = load_datasets(**load_extra_args)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("Tokenizing and formatting \n\n")

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    def format_dataset(dataset):
        dataset = dataset.map(lambda examples: {'labels': examples['label']})
        columns = ['input_ids', 'attention_mask', 'labels']
        if 'token_type_ids' in dataset.features:
            columns.append('token_type_ids')
        dataset.set_format(type='torch', columns=columns)
        print(columns)
        return dataset

    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)
    test_dataset = format_dataset(test_dataset)


    print("Sanity check\n\n")

    print(tokenizer.decode(test_dataset[0]["input_ids"]), "\n\n")

    print("\n\nEvaluating\n")


    training_args = TrainingArguments(
        output_dir='.',
        per_device_eval_batch_size=eval_batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=lambda x: compute_metrics(x, id2label=id2label),
    )

    preds = trainer.predict(test_dataset)

    serialized = {
        "lang": lang,
        "model": model_name,
        "predictions": preds.predictions.tolist(),
        "labels": preds.label_ids.tolist(),
        "metrics": preds.metrics
    }

    print(f"Saving at {output_path}")

    with open(output_path, "w+") as f:
        json.dump(serialized, f, indent=4)

if __name__ == "__main__":
    fire.Fire(eval_sentiment)
