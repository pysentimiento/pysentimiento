"""
NER for LinCE dataset
"""
from emoji import emoji_lis, demojize
import numpy as np
from seqeval.metrics import f1_score
from datasets import load_dataset, load_metric, Dataset, DatasetDict
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification
from ..preprocessing import preprocess_tweet
from ..training import train_and_eval

metric = load_metric("seqeval")

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


def align_labels_with_tokens(labels, word_ids, ignore_label=-100, label_subwords=False):
    """
    Taken from https://huggingface.co/course/chapter7/2?fw=pt

    Arguments:

    labels: List[int]
        list of NER labels ()

    word_ids: List[int]
        list of word ids (one for each token)

    ignore_label: int (default: -100)
        Value to set to ignored labels ([CLS], [SEP], [PAD] and subwords if label_subwords=False)

    label_subwords: bool (default: True)
        If True, then labels are assigned to the subword that contains the token.

        For instance, if "Bigsubword" (split as Big ##sub ##word )
        is labeled as "B-PER", then "Big" is labeled as "B-PER",
        ##sub ##word as "I-PER" and "##word" is labeled as "I-PER".

        If False, sets to ignore_id the subwords


    """
    new_labels = []
    current_word = None

    if all(l is None for l in labels):
        # All labels are none => test dataset
        return [None] * len(word_ids)

    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = ignore_label if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(ignore_label)
        else:
            """
            Same word as previous token
            """
            if label_subwords:
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                # WARNING: this strategy is only valid for BIO labels
                if label % 2 == 1:
                    label += 1
            else:
                label = ignore_label
            new_labels.append(label)

    return new_labels


def compute_metrics(eval_preds):
    """
    Compute metrics for NER
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100]
                   for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(
        predictions=true_predictions, references=true_labels)
    ret = {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "macro_f1": f1_score(true_labels, true_predictions, average="macro"),
        "micro_f1": f1_score(true_labels, true_predictions, average="micro"),
        "accuracy": all_metrics["overall_accuracy"],
    }

    for k, v in all_metrics.items():
        if not k.startswith("overall"):
            ret[k + "_f1"] = v["f1"]
            ret[k + "_precision"] = v["precision"]
            ret[k + "_recall"] = v["recall"]

    return ret


def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenize examples and also realign labels
    """
    tokenized_inputs = tokenizer(
        examples["words"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["labels"]
    new_labels = []
    word_ids = [tokenized_inputs.word_ids(i) for i in range(len(all_labels))]
    for (labels, wids) in zip(all_labels, word_ids):
        new_labels.append(
            align_labels_with_tokens(labels, wids)
        )

    tokenized_inputs["labels"] = new_labels
    tokenized_inputs["word_ids"] = word_ids

    return tokenized_inputs


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


def train(
        base_model, lang, epochs=5,
        metric_for_best_model="micro_f1",
        **kwargs):

    ds = load_datasets(
        lang=lang
    )

    return train_model(
        base_model,
        train_dataset=ds["train"], dev_dataset=ds["dev"], test_dataset=ds["dev"],
        id2label=id2label, lang=lang, epochs=epochs,
        # Custom stuff for this thing to work
        tokenize_fun=tokenize_and_align_labels,
        auto_class=AutoModelForTokenClassification,
        data_collator_class=DataCollatorForTokenClassification,
        metrics_fun=compute_metrics,
        metric_for_best_model=metric_for_best_model,
        **kwargs
    )
