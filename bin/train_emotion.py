import sys
import fire
import torch
from transformers import (
    Trainer, TrainingArguments,
)
import pandas as pd
from pysentimiento.training import train_model, load_model
from pysentimiento.emotion import load_datasets
from pysentimiento.emotion.datasets import id2label, label2id
from pysentimiento.metrics import compute_metrics
from sklearn.utils.class_weight import compute_class_weight

extra_args = {
    "vinai/bertweet-base": {
        "preprocessing_args": {"user_token": "@USER", "url_token": "HTTPURL"}
    }
}



def train(
    base_model, output_path, lang="es", epochs=5, batch_size=32, eval_batch_size=16,
    warmup_proportion=.1, limit=None
):
    """
    """

    print("="*80 + '\n', "="*80 + '\n')
    print(f"Training {base_model} in language {lang}", "\n" * 2)
    print("Loading dataset")
    if lang not in ["es", "en"]:
        print("lang must be one of 'es', 'en'")
        sys.exit(1)



    load_extra_args = extra_args[base_model] if base_model in extra_args else {}

    train_dataset, dev_dataset, test_dataset = load_datasets(lang=lang, **load_extra_args)


    if limit:
        """
        Smoke test
        """
        print("\n\n", f"Limiting to {limit} instances")
        train_dataset = train_dataset.select(range(limit))
        dev_dataset = dev_dataset.select(range(limit))
        test_dataset = test_dataset.select(range(limit))


    class_weight = torch.Tensor(
        compute_class_weight('balanced', list(id2label), y=train_dataset["label"])
    )

    model, tokenizer = load_model(base_model,
        id2label=id2label,
        label2id=label2id
    )


    _, test_results = train_model(
        model, tokenizer, train_dataset, dev_dataset, test_dataset, id2label,
        epochs=epochs, batch_size=batch_size, class_weight=class_weight
    )

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
