from datasets import load_dataset, Dataset, DatasetDict, Features, Value, ClassLabel
from pysentimiento.preprocessing import preprocess_tweet


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

    return ds
