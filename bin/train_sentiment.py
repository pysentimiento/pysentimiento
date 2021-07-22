import sys
import fire
import torch
from pysentimiento.tass import (
    load_datasets as load_tass_datasets, id2label as id2labeltass, label2id as label2idtass,
)
from pysentimiento.training import load_model, train_model
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
    base_model, output_path, lang="es", epochs=5, batch_size=32, eval_batch_size=32, warmup_proportion=0.1, limit=None,
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
        return tokenizer(batch['text'], padding=False, truncation=True)




    _, test_results = train_model(
        model, tokenizer, train_dataset, dev_dataset, test_dataset, id2label,
        epochs=epochs, batch_size=batch_size
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
