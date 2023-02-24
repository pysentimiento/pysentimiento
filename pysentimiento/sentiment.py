from .semeval import (
    load_datasets as load_semeval_datasets,
    id2label as id2labelsemeval, label2id as label2idsemeval
)
from .tuning import hyperparameter_sweep, get_training_arguments
from .training import train_and_eval, load_model
from .tass import (
    load_datasets as load_tass_datasets, id2label as id2labeltass, label2id as label2idtass,
)
from .sentipolc import (
    load_datasets as load_sentipolc_datasets, id2label as id2labelsentipolc, label2id as label2idsentipolc
)

from .sentiment_pt import load_datasets as load_sentiment_pt_datasets, id2label as id2labelpt, label2id as label2idpt
from .preprocessing import get_preprocessing_args

task_name = "sentiment"

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
    },

    "it": {
        "load_datasets": load_sentipolc_datasets,
        "id2label": id2labelsentipolc,
        "label2id": label2idsentipolc,
    },

    "pt": {
        "load_datasets": load_sentiment_pt_datasets,
        "id2label": id2labelpt,
        "label2id": label2idpt,
    }
}


def accepts(lang):
    """
    Check if a language is supported by this task
    """
    return lang in lang_conf


def load_datasets(lang, **kwargs):
    """
    """
    return lang_conf[lang]["load_datasets"](lang=lang, **kwargs)


def train(
    base_model, lang="es", use_defaults_if_not_tuned=False,
    **kwargs
):
    """

    """

    load_datasets = lang_conf[lang]["load_datasets"]
    id2label = lang_conf[lang]["id2label"]

    ds = load_datasets(
        lang=lang, preprocessing_args=get_preprocessing_args(base_model, lang=lang))

    training_args = get_training_arguments(
        base_model, task_name=task_name, lang=lang,
        metric_for_best_model="eval/macro_f1", use_defaults_if_not_tuned=use_defaults_if_not_tuned
    )

    return train_and_eval(
        base_model=base_model, dataset=ds, id2label=id2label,
        training_args=training_args, lang=lang, **kwargs
    )


def hp_tune(model_name, lang, **kwargs):
    """
    Hyperparameter tuning with wandb
    """
    task_name = "sentiment"

    id2label = lang_conf[lang]["id2label"]

    ds = load_datasets(lang=lang)

    def model_init():
        model, _ = load_model(model_name, id2label, lang=lang)
        return model

    _, tokenizer = load_model(model_name, id2label, lang=lang)

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
        **kwargs,
    )
