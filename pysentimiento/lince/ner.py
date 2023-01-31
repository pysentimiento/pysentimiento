"""
NER for LinCE dataset
"""
from emoji import emoji_lis

from seqeval.metrics import f1_score
from datasets import load_dataset, Dataset, DatasetDict
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification
from ..preprocessing import preprocess_tweet
from ..training import train_and_eval
from ..ner import tokenize_and_align_labels, compute_metrics

id2label = [
    'O',
    'B-EVENT',
    'I-EVENT',
    'B-GROUP',
    'I-GROUP',
    'B-LOC',
    'I-LOC',
    'B-ORG',
    'I-ORG',
    'B-OTHER',
    'I-OTHER',
    'B-PER',
    'I-PER',
    'B-PROD',
    'I-PROD',
    'B-TIME',
    'I-TIME',
    'B-TITLE',
    'I-TITLE',
]

label2id = {v: k for k, v in enumerate(id2label)}


def preprocess_token(t, lang, demoji=True, preprocess_hashtags=False, **kwargs):
    """
    Preprocess each token
    """
    token = None
    if t.startswith("http") and "t.co" in t:
        """
        TODO: this is a patch for preprocess_tweet, but it should be fixed in the future
        """
        token = "url"
    else:
        if demoji:
            emojis = emoji_lis(t, language=lang)
            if emojis:
                """
                Put special token
                """
                token = "emoji"

        if not token:
            if len(t) == 1 and not t.isascii():
                token = "."
            else:
                token = preprocess_tweet(
                    t, lang=lang, demoji=False, preprocess_hashtags=preprocess_hashtags,
                    char_replace=False, **kwargs
                )

    if not token:
        """
        token is empty or None => put a placeholder
        """
        token = "."

    return token


def load_conll(path, lang="es"):
    """
    Loads CoNLL-2003 dataset
    """
    with open(path) as f:
        lines = f.read().splitlines()
    data = []
    current_line = []
    for line in lines:
        line = line.strip()
        if line.startswith("# "):
            continue
        elif line == "":
            data.append(current_line)
            current_line = []
        else:
            current_line.append(line.split("\t"))
    return data


def load_dataset_from_conll(path):
    data = load_conll(path)
    words = [[x[0] for x in sentence] for sentence in data]
    langs = [[x[1] for x in sentence] for sentence in data]
    if len(data[0][0]) == 2:
        ner = [[None] * len(x) for x in data]
    else:
        ner = [[x[2] for x in sentence] for sentence in data]

    # Sanity Check
    for w, l in zip(words, ner):
        assert len(w) == len(l)

    return Dataset.from_dict({
        "words": words,
        "lang": langs,
        "ner": ner,
    })


def load_datasets(lang="es", preprocess=True):
    """
    Load NER datasets
    """

    lince_ner = load_dataset("lince", "ner_spaeng")

    """
    TODO: None is for test labels which are not available
    """

    lince_ner = lince_ner.map(
        lambda x: {"labels": [label2id.get(x, None) for x in x["ner"]]}
    )

    if preprocess:
        lince_ner = lince_ner.map(
            lambda x: {
                "words": [preprocess_token(word, lang) for word in x["words"]]
            }
        )

    return DatasetDict(
        train=lince_ner["train"],
        dev=lince_ner["validation"],
        test=lince_ner["test"],
    )


def train(base_model, lang, **kwargs):

    ds = load_datasets(
        lang=lang
    )

    return train_and_eval(
        base_model,
        dataset=ds,
        id2label=id2label, lang=lang,
        # Custom stuff for this thing to work
        tokenize_fun=tokenize_and_align_labels,
        auto_class=AutoModelForTokenClassification,
        data_collator_class=DataCollatorForTokenClassification,
        metrics_fun=lambda *args, **kwargs: compute_metrics(
            *args, id2label=id2label, **kwargs),
        metric_for_best_model="micro_f1",
        **kwargs
    )
