import torch
from datasets import load_dataset
from pysentimiento.preprocessing import preprocess_tweet


id2label = {0: 'NEG', 1: 'NEU', 2: 'POS'}
label2id = {v: k for k, v in id2label.items()}


def load_datasets(preprocess=True, preprocessing_args={}, **kwargs):
    """
    Return train, dev, test datasets
    """
    ds = load_dataset("pysentimiento/pt_sentiment")

    """
    Tokenize tweets
    """

    if preprocess:
        def pt_preprocess(x):
            return {"text": preprocess_tweet(
                x["text"], lang="pt", **preprocessing_args

            )}
        ds = ds.map(pt_preprocess, batched=False)

    # Re-preprocess
    ds = ds.map(lambda ex: {"text": ex["text"].replace("USERNAME", "@USER")})
    ds = ds.map(lambda ex: {"text": ex["text"].replace("URL", "HTTPURL")})

    return ds
