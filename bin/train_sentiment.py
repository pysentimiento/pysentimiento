from pysentimiento.emotion.datasets import load_datasets
import sys
import fire
import torch
from glob import glob
from transformers import (
    Trainer, TrainingArguments, set_seed
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
    "vinai/bertweet-base": {
        "preprocessing_args": {"user_token": "@USER", "url_token": "HTTPURL"}
    }
}



def train(
    base_model, output_path, lang="es", epochs=5, batch_size=32, eval_batch_size=16, warmup_proportion=0.1, limit=None,
):
    """
    """
    print("="*80 + '\n', "="*80 + '\n')
    print(f"Training {base_model} in language {lang}", "\n" * 2)
    print("Loading dataset")
    if lang not in lang_conf.keys():
        print("lang must be one of ", lang_conf.keys())
        sys.exit(1)

    load_datasets = lang_conf[lang]["load_datasets"]
    id2label = lang_conf[lang]["id2label"]
    label2id = lang_conf[lang]["label2id"]


    load_extra_args = extra_args[base_model] if base_model in extra_args else {}

    train_dataset, dev_dataset, test_dataset = load_datasets(**load_extra_args)

    if limit:
        """
        Smoke test
        """
        print("\n\n", f"Limiting to {limit} instances")
        train_dataset = train_dataset.select(range(limit))
        dev_dataset = dev_dataset.select(range(limit))
        test_dataset = test_dataset.select(range(limit))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model(base_model, label2id=label2id, id2label=id2label)

    model = model.to(device)
    model.train()

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)


    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=batch_size)
    dev_dataset = dev_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=eval_batch_size)

    def format_dataset(dataset):
        dataset = dataset.map(lambda examples: {'labels': examples['label']})
        columns = ['input_ids', 'attention_mask', 'labels']
        if 'token_type_ids' in dataset.features:
            columns.append('token_type_ids')
        dataset.set_format(type='torch', columns=columns)
        print(columns)
        return dataset

    train_dataset = format_dataset(train_dataset)
    dev_dataset = format_dataset(dev_dataset)
    test_dataset = format_dataset(test_dataset)


    print("\n\nTraining\n")


    total_steps = (epochs * len(train_dataset)) // batch_size
    warmup_steps = int(warmup_proportion * total_steps)
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
        metric_for_best_model="macro_f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=lambda x: compute_metrics(x, id2label=id2label),
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


    print(f"Saving model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    fire.Fire(train)
