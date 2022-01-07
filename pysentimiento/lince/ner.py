"""
NER for LinCE dataset
"""
import os
import tempfile
import numpy as np
from seqeval.metrics import f1_score
from datasets import load_dataset, DatasetDict, load_metric
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

from ..preprocessing import preprocess_tweet
from ..training import load_model

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

def compute_metrics(eval_preds):
    """
    Compute metrics for NER
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    ret = {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "macro_f1": all_metrics["overall_f1"],
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
    base_model, lang, epochs=5, batch_size=32, eval_batch_size=32,
    warmup_ratio=.1, limit=None, accumulation_steps=1, max_length=128,
    load_best_model_at_end=True, metric_for_best_model="micro_f1",
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


    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    output_path = tempfile.mkdtemp()
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        warmup_ratio=warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        do_eval=False,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        group_by_length=True,
        **kwargs,
    )

    trainer_args = {
        "model": model,
        "args": training_args,
        "compute_metrics": compute_metrics,
        "train_dataset": tokenized_datasets["train"],
        "eval_dataset": tokenized_datasets["validation"],
        "data_collator": data_collator,
        "tokenizer": tokenizer,
    }

    trainer = Trainer(**trainer_args)

    trainer.train()


    test_results = trainer.predict(tokenized_datasets["validation"])

    os.system(f"rm -Rf {output_path}")

    return trainer, test_results