from pysentimiento.irony import load_datasets
from pysentimiento.preprocessing import preprocess_tweet


def test_it_can_load_es_datasets():
    ds = load_datasets(lang="es", preprocess=False)

    assert [k in ds for k in ["train", "dev", "test"]]


def test_it_can_load_en_datasets():
    ds = load_datasets(lang="en", preprocess=False)

    assert [k in ds for k in ["train", "dev", "test"]]


def test_it_can_load_en_datasets():
    ds_no_prepro = load_datasets(lang="en", preprocess=False)
    ds = load_datasets(lang="en", preprocess=True)

    raw_text = ds_no_prepro["train"][0]["text"]
    prepro_text = ds["train"][0]["text"]

    assert prepro_text == preprocess_tweet(raw_text, lang="en")
