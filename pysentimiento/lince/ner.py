"""
NER for LinCE dataset
"""
from ..preprocessing import preprocess_tweet
from ..training import load_model
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForTokenClassification

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

label2id = {v:k for k,v in enumerate(id2label)}


def align_labels_with_tokens(labels, word_ids):
    """
    Tomado de https://huggingface.co/course/chapter7/2?fw=pt
    """
    new_labels = []
    current_word = None

    if all(l is None for l in labels):
        """
        All labels are none => test dataset
        """
        return [None] * len(word_ids)

    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            label = labels[word_id]
            # Same word as previous token
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenize examples and also realign labels
    """
    tokenized_inputs = tokenizer(
        examples["words"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(
            align_labels_with_tokens(labels, word_ids)
        )

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def load_datasets(lang="es", preprocess=True):
    """
    Load NER datasets
    """
    def preprocess_token(t, lang):
        """
        Seguro podemos hacerlo mejor
        """
        return preprocess_tweet(
            t, lang=lang, demoji=False, preprocess_hashtags=False
        )

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

    return lince_ner["train"], lince_ner["validation"], lince_ner["test"]

def train(
    base_model, lang, epochs=5, batch_size=32,
    warmup_ratio=.1, limit=None, accumulation_steps=1, task_b=True, class_weight=None,
    max_length=128, dev=False, metric_for_best_model="macro_f1",
    **kwargs):

    train_dataset, dev_dataset, test_dataset = load_datasets(
        lang=lang
    )


    if limit:
        """
        Smoke test
        """
        print("\n\n", f"Limiting to {limit} instances")
        train_dataset = train_dataset.select(range(limit))
        dev_dataset = dev_dataset.select(range(limit))
        test_dataset = test_dataset.select(range(limit))

    lince_ner = DatasetDict(
        train=train_dataset,
        validation=dev_dataset,
        test=test_dataset,
    )

    model, tokenizer = load_model(
        base_model, label2id=label2id, id2label=id2label,
        max_length=max_length,
        auto_class=AutoModelForTokenClassification
    )

    tokenized_datasets = lince_ner.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=lince_ner["train"].column_names,
    )
