import torch
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, ClassLabel
from pysentimiento.preprocessing import preprocess_tweet

id2label = {0: "pos", 1: "neg"}

label2id = {label: i for i, label in id2label.items()}

def load_datasets(seed=2021, preprocess=True, preprocessing_args={}, **kwargs):
    """
    Return train, dev, test datasets
    """
    ds = load_dataset("pysentimiento/it_sentipolc16")

    """
    Tokenize tweets
    """

    if preprocess:
        def it_preprocess(x):
            return {"text": preprocess_tweet(
                x["text"], lang="it", **preprocessing_args
            )}
        ds = ds.map(it_preprocess, batched=False)

    # Set label

    ds = ds.map(lambda ex: {"labels": torch.Tensor(
        [ex["opos"], ex["oneg"]])}, batched=False)

    return ds
