"""
NER for LinCE dataset
"""
import numpy as np
from .ner import tokenize_and_align_labels
from ..preprocessing import preprocess_tweet
from datasets import load_dataset, load_metric
from seqeval.metrics import f1_score
from ..training import train_model


id2label =[
   "negative",
   "neutral",
   "positive",
]

label2id = {v:k for k,v in enumerate(id2label)}


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
        preprocess_fn = lambda x: preprocess_tweet(x, lang=lang, **preprocessing_args)

        lince = lince.map(
            lambda x: {"text": preprocess_fn(x["text"])}
        )

    return lince["train"], lince["validation"], lince["test"]


def train(
    base_model, lang, epochs=5,
    metric_for_best_model="macro_f1",
    **kwargs):

    train_dataset, dev_dataset, test_dataset = load_datasets(
        lang=lang
    )

    return train_model(
        base_model,
        train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=dev_dataset,
        id2label=id2label, lang=lang, epochs=epochs,
        # Custom stuff for this thing to work
        metric_for_best_model=metric_for_best_model,
        **kwargs
    )