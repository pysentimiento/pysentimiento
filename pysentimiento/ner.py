import numpy as np
from seqeval.metrics import f1_score
from datasets import load_metric

metric = load_metric("seqeval")


def compute_metrics(eval_preds, id2label):
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


def tokenize_and_align_labels(examples, tokenizer, tokens_column="tokens", label_column="ner_tags"):
    """
    Tokenize examples and also realign labels

    Arguments:

        examples: List[Dict[str, List[str]]]
            List of examples

        tokenizer: transformers.PreTrainedTokenizer
            Tokenizer to use

        label_column: str (default: "ner_tags")
            Name of the column containing the labels
    """
    tokenized_inputs = tokenizer(
        examples[tokens_column], truncation=True, is_split_into_words=True
    )
    all_labels = examples[label_column]
    new_labels = []
    word_ids = [tokenized_inputs.word_ids(i) for i in range(len(all_labels))]
    for (labels, wids) in zip(all_labels, word_ids):
        new_labels.append(
            align_labels_with_tokens(labels, wids)
        )

    tokenized_inputs["labels"] = new_labels
    tokenized_inputs["word_ids"] = word_ids

    return tokenized_inputs
