import wandb
from .tass import (
    load_datasets as load_tass_datasets, id2label as id2labeltass, label2id as label2idtass,
)
from .training import train_model, load_model
from .tuning import hyperparameter_sweep
from .semeval import (
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


def load_datasets(lang, **kwargs):
    """
    """
    return lang_conf[lang]["load_datasets"](lang=lang, **kwargs)


def train(
    base_model, lang="es", epochs=5, batch_size=32,
    limit=None, **kwargs
):
    """
    """
    load_datasets = lang_conf[lang]["load_datasets"]
    id2label = lang_conf[lang]["id2label"]

    load_extra_args = extra_args[base_model] if base_model in extra_args else {
    }

    ds = load_datasets(**load_extra_args)

    kwargs = {
        **kwargs,
        **{
            "id2label": id2label,
            "epochs": epochs,
            "batch_size": batch_size,
            "limit": limit,
            "lang": lang,
        }
    }

    return train_model(base_model, ds["train"], ds["dev"], ds["test"], **kwargs)


def hp_tune(model_name, lang):
    """
    Hyperparameter tuning with wandb
    """
    task_name = "sentiment"

    id2label = lang_conf[lang]["id2label"]

    ds = load_datasets(lang=lang)

    def model_init():
        model, _ = load_model(model_name, id2label)
        return model

    _, tokenizer = load_model(model_name, id2label)

    config_info = {
        "model": model_name,
        "task": task_name,
        "lang": lang,
    }

    return hyperparameter_sweep(
        name=f"swp-{task_name}-{lang}-{model_name}",
        group_name=f"swp-{task_name}-{lang}",
        model_init=model_init,
        tokenizer=tokenizer,
        datasets=ds,
        id2label=id2label,
        config_info=config_info,
    )
