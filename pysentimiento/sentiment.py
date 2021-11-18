import torch
from pysentimiento.tass import (
    load_datasets as load_tass_datasets, id2label as id2labeltass, label2id as label2idtass,
)
from pysentimiento.training import load_model, train_model
from pysentimiento.baselines.training import train_rnn_model
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
    base_model, lang="es", epochs=5, batch_size=32,
    limit=None, **kwargs
):
    """
    """
    load_datasets = lang_conf[lang]["load_datasets"]
    id2label = lang_conf[lang]["id2label"]

    load_extra_args = extra_args[base_model] if base_model in extra_args else {}

    train_dataset, dev_dataset, test_dataset = load_datasets(**load_extra_args)

    kwargs = {
        **kwargs,
        **{
            "id2label" : id2label,
            "epochs": epochs,
            "batch_size": batch_size,
            "limit": limit,
            "lang": lang,
        }
    }

    return train_model(base_model, train_dataset, dev_dataset, test_dataset, **kwargs)
