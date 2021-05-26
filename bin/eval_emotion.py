import json
import sys
import fire
import torch
from transformers import (
    Trainer, TrainingArguments,
    AutoModelForSequenceClassification, AutoTokenizer
)
import pandas as pd
from pysentimiento.tass import load_model
from pysentimiento.emotion import load_datasets
from pysentimiento.emotion.datasets import id2label, label2id
from pysentimiento.metrics import compute_metrics



extra_args = {
    "BertweetTokenizer": {
        "preprocessing_args": {"user_token": "@USER", "url_token": "HTTPURL"}
    }
}



def eval_emotion(
    model_name, output_path, lang="es", eval_batch_size=16, warmup_proportion=0.1, limit=None,
):
    """
    """
    print("="*80 + '\n', "="*80 + '\n')
    print(f"Evaluating {model_name} in language {lang}", "\n" * 2)
    print("Loading dataset")
    if lang not in ["es", "en"]:
        print("lang must be one of ", ["es", "en"])
        sys.exit(1)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 128
    model.eval()

    tokenizer_class_name = model.config.tokenizer_class

    load_extra_args = extra_args[tokenizer_class_name] if tokenizer_class_name in extra_args else {}

    _, _, test_dataset = load_datasets(lang=lang, **load_extra_args)


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

    print("\n\nTraining\n")


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
        "model": model_name,
        "lang": lang,
        "predictions": preds.predictions.tolist(),
        "labels": preds.label_ids.tolist(),
        "metrics": preds.metrics
    }

    print(f"Saving at {output_path}")

    with open(output_path, "w+") as f:
        json.dump(serialized, f, indent=4)

if __name__ == "__main__":
    fire.Fire(eval_emotion)
