import torch
from .config import config
from datasets import load_dataset
from .preprocessing import get_preprocessing_args, preprocess_tweet
from .training import train_and_eval, load_model
from .tuning import hyperparameter_sweep, get_training_arguments
import logging


logger = logging.getLogger("pysentimiento")
logger.setLevel(logging.INFO)


task_name = "context_hate_speech"

id2label = [
    "CALLS",
    "WOMEN",
    "LGBTI",
    "RACISM",
    "CLASS",
    "POLITICS",
    "DISABLED",
    "APPEARANCE",
    "CRIMINAL",
]

id2label = {i: label for i, label in enumerate(id2label)}


def accepts(lang, **kwargs):
    """
    Returns True if the task supports the language
    """
    return lang in ["es"]


def load_datasets(lang, preprocess=True, preprocessing_args={}):
    """ "
    Load targeted sentiment datasets

    Args:
        lang (str): Language of the dataset
        preprocess (bool, optional): Whether to preprocess the dataset. Defaults to True.
        preprocessing_args (dict, optional): Arguments for the preprocessing. Defaults to {}.
    Returns:


    """
    if lang != "es":
        raise ValueError(
            f"Language {lang} not supported for Targeted Sentiment Analysis detection"
        )

    ds = load_dataset("piuba-bigdata/contextualized_hate_speech")

    if preprocess:

        def preprocess_fn(x):
            return {
                "text": preprocess_tweet(x["text"], lang=lang, **preprocessing_args),
                "context_tweet": preprocess_tweet(
                    x["context_tweet"], lang=lang, **preprocessing_args
                ),
            }

        ds = ds.map(preprocess_fn, batched=False)

    ds = ds.map(
        lambda x: {
            "labels": torch.Tensor([x[id2label[i]] for i in range(len(id2label))])
        },
        batched=False,
    )
    return ds


def hp_tune(
    model_name,
    lang,
    context=True,
    **kwargs,
):
    """
    Train function

    Arguments:
    ---------

    """

    ds = load_datasets(
        lang=lang, preprocessing_args=get_preprocessing_args(model_name, lang=lang)
    )

    logger.info(ds)

    _, tokenizer = load_model(model_name, id2label, lang=lang)

    def tokenize_fun(batch):
        max_len = tokenizer.model_max_length
        if context:
            return tokenizer(
                batch["text"],
                batch["context_tweet"],
                padding=False,
                truncation=True,
                max_length=min(max_len, 256),
            )
        return tokenizer(
            batch["text"],
            batch["target"],
            padding=False,
            truncation=True,
            max_length=min(max_len, 128),
        )

    def model_init():
        model, _ = load_model(model_name, id2label, lang=lang)
        return model

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
        tokenize_fun=tokenize_fun,
        **kwargs,
    )


def train(
    model_name, lang="es", use_defaults_if_not_tuned=False, context=True, **kwargs
):
    ds = load_datasets(
        lang=lang,
        preprocessing_args=get_preprocessing_args(model_name, lang=lang),
    )

    logger.info(ds)

    def tokenize_fun(batch):
        max_len = tokenizer.model_max_length
        if context:
            return tokenizer(
                batch["text"],
                batch["context_tweet"],
                padding=False,
                truncation=True,
                max_length=min(max_len, 256),
            )
        return tokenizer(
            batch["text"],
            batch["target"],
            padding=False,
            truncation=True,
            max_length=min(max_len, 128),
        )

    training_args = get_training_arguments(
        model_name,
        task_name=task_name,
        lang=lang,
        metric_for_best_model="eval/macro_f1",
        use_defaults_if_not_tuned=use_defaults_if_not_tuned,
    )

    _, tokenizer = load_model(model_name, id2label, lang=lang)

    return train_and_eval(
        base_model=model_name,
        dataset=ds,
        id2label=id2label,
        tokenize_fun=tokenize_fun,
        training_args=training_args,
        lang=lang,
        **kwargs,
    )
