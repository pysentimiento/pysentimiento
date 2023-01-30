"""
NER for LinCE dataset
"""
import numpy as np
from ..preprocessing import preprocess_tweet
from datasets import load_dataset, DatasetDict
from ..training import train_and_eval


id2label = [
    "negative",
    "neutral",
    "positive",
]

label2id = {v: k for k, v in enumerate(id2label)}


def load_datasets(lang="es", preprocess=True, preprocessing_args={}):
    """
    Load NER datasets
    """

    lince = load_dataset("lince", "sa_spaeng")
    # This hack is because of seqeval only working with BIO tags
    lince = lince.map(
        lambda x: {
            "text": " ".join(x["words"]),
            "label": label2id.get(x["sa"], None),
        }
    )

    if preprocess:
        def preprocess_fn(x): return preprocess_tweet(
            x, lang=lang, **preprocessing_args)

        lince = lince.map(
            lambda x: {"text": preprocess_fn(x["text"])}
        )

    return DatasetDict(
        train=lince["train"],
        dev=lince["validation"],
        test=lince["test"]
    )


def train(
        base_model, lang, epochs=5,
        metric_for_best_model="macro_f1",
        **kwargs):

    ds = load_datasets(
        lang=lang
    )

    return train_and_eval(
        base_model,
        train_dataset=ds["train"], dev_dataset=ds["dev"], test_dataset=ds["dev"],
        id2label=id2label, lang=lang, epochs=epochs,
        # Custom stuff for this thing to work
        metric_for_best_model=metric_for_best_model,
        **kwargs
    )
