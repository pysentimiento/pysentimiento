import pandas as pd
import os
import torch
from datasets import ClassLabel, load_dataset, VerificationMode
from .training import train_and_eval, load_model
from .tuning import hyperparameter_sweep, get_training_arguments
from .preprocessing import preprocess_tweet, get_preprocessing_args

"""
Lo pongo así por huggingface
"""

task_name = "emotion"


def accepts(lang, **kwargs):
    """
    Returns True if the task supports the language
    """
    return lang in ["es", "en", "it", "pt"]


def load_datasets(lang, preprocess=True, preprocessing_args={}):
    """
    Load sentiment datasets
    """

    if lang in {"es", "en", "it", "pt"}:
        # I add VerificationMode.NO_CHECKS to avoid the error
        ds = load_dataset(
            f"pysentimiento/{lang}_emotion",
            verification_mode=VerificationMode.NO_CHECKS,
        )
    else:
        raise ValueError(f"Language {lang} not supported for irony detection")

    if preprocess:

        def preprocess_fn(ex):
            return {
                "text": preprocess_tweet(ex["text"], lang=lang, **preprocessing_args)
            }

        ds = ds.map(preprocess_fn, batched=False)

    return ds


def train(base_model, lang="es", use_defaults_if_not_tuned=False, **kwargs):
    """ """
    ds = load_datasets(
        lang=lang, preprocessing_args=get_preprocessing_args(base_model, lang=lang)
    )

    if "labels" in ds["train"].features:
        emotions = ds["train"].features["labels"].feature.names
        id2label = {i: label for i, label in enumerate(emotions)}

        def _convert_labels(ex):
            labels = [0.0] * len(emotions)
            for l in ex["labels"]:
                labels[l] = 1.0
            return {"labels": labels, "foo": labels}

        ds = ds.map(_convert_labels, batched=False)
        ds = ds.remove_columns("labels")
        ds = ds.rename_column("foo", "labels")
        problem_type = "multi_label_classification"
    else:
        id2label = {k: v for k, v in enumerate(ds["train"].features["label"].names)}

    training_args = get_training_arguments(
        base_model,
        task_name=task_name,
        lang=lang,
        metric_for_best_model="eval/macro_f1",
        use_defaults_if_not_tuned=use_defaults_if_not_tuned,
    )

    return train_and_eval(
        base_model=base_model,
        dataset=ds,
        id2label=id2label,
        training_args=training_args,
        lang=lang,
        problem_type=problem_type,
        **kwargs,
    )


def hp_tune(base_model, lang, **kwargs):
    """
    Hyperparameter tuning with wandb
    """
    ds = load_datasets(
        lang=lang, preprocessing_args=get_preprocessing_args(base_model, lang=lang)
    )

    if "labels" in ds["train"].features:
        emotions = ds["train"].features["labels"].feature.names
        id2label = {i: label for i, label in enumerate(emotions)}

        def _convert_labels(ex):
            labels = [0.0] * len(emotions)
            for l in ex["labels"]:
                labels[l] = 1.0
            return {"labels": labels, "foo": labels}

        ds = ds.map(_convert_labels, batched=False)
        ds = ds.remove_columns("labels")
        ds = ds.rename_column("foo", "labels")
        problem_type = "multi_label_classification"
    else:
        id2label = {k: v for k, v in enumerate(ds["train"].features["label"].names)}

    def model_init():
        model, _ = load_model(
            base_model, id2label, lang=lang, problem_type=problem_type
        )
        return model

    _, tokenizer = load_model(base_model, id2label, lang=lang)

    config_info = {
        "model": base_model,
        "task": task_name,
        "lang": lang,
    }

    return hyperparameter_sweep(
        name=f"swp-{task_name}-{lang}-{base_model}",
        group_name=f"swp-{task_name}-{lang}",
        model_init=model_init,
        tokenizer=tokenizer,
        datasets=ds,
        id2label=id2label,
        config_info=config_info,
        **kwargs,
    )
