import pandas as pd
import os
import pathlib
import torch
from datasets import ClassLabel, load_dataset
from .training import train_and_eval, load_model
from .tuning import hyperparameter_sweep, get_training_arguments
from .preprocessing import preprocess_tweet, get_preprocessing_args

"""
Lo pongo así por huggingface
"""

task_name = "emotion"


project_dir = pathlib.Path(os.path.dirname(__file__)).parent
data_dir = os.path.join(project_dir, "data")
emotion_dir = os.path.join(data_dir, "emotion")


paths = {
    "es": {
        "train": os.path.join(emotion_dir, "train_es.csv"),
        "test": os.path.join(emotion_dir, "test_es.csv"),
    },
    "en": {
        "train": os.path.join(emotion_dir, "train_en.csv"),
        "test": os.path.join(emotion_dir, "test_en.csv"),
    }
}


def load_df(path):
    """
    Load TASS dataset
    """
    df = pd.read_csv(path)
    return df


def accepts(lang, **kwargs):
    """
    Returns True if the task supports the language
    """
    return lang in ["es", "en", "it", "pt"]


def load_datasets(lang, preprocess=True, preprocessing_args={}):
    """
    Load sentiment datasets
    """

    if lang in {"es", "en", "it"}:
        ds = load_dataset(f"pysentimiento/{lang}_emotion")
    elif lang == "pt":
        ds = load_dataset("pysentimiento/pt_emotion")

        label_names = [
            k for k, v in ds["train"].features.items()
            if isinstance(v, ClassLabel)
        ]

        ds = ds.map(
            lambda ex: {"labels": torch.Tensor(
                [ex[label] for label in label_names])},
            batched=False
        )
    else:
        raise ValueError(f"Language {lang} not supported for irony detection")

    if preprocess:
        def preprocess_fn(ex):
            return {"text": preprocess_tweet(ex["text"], lang=lang, **preprocessing_args)}
        ds = ds.map(preprocess_fn, batched=False)

    return ds


def train(
    base_model, lang="es", use_defaults_if_not_tuned=False,
    **kwargs
):
    """
    """
    ds = load_datasets(
        lang=lang,
        preprocessing_args=get_preprocessing_args(base_model, lang=lang)
    )

    if lang != "pt":

        id2label = {k: v for k, v in enumerate(
            ds["train"].features["label"].names)}
    else:
        label_names = [
            k for k, v in ds["train"].features.items()
            if isinstance(v, ClassLabel)
        ]

        id2label = {k: v for k, v in enumerate(label_names)}

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
    ds = load_datasets(
        lang=lang,
        preprocessing_args=get_preprocessing_args(model_name, lang=lang)
    )

    if lang != "pt":

        id2label = {k: v for k, v in enumerate(
            ds["train"].features["label"].names)}
    else:
        label_names = [
            k for k, v in ds["train"].features.items()
            if isinstance(v, ClassLabel)
        ]

        id2label = {k: v for k, v in enumerate(label_names)}

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
