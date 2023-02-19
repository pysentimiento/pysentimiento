from .hate import load_datasets as load_hate_datasets
from .tass import load_datasets as load_tass_datasets
from .semeval import load_datasets as load_semeval_datasets
from .emotion import load_datasets as load_emotion_datasets
from .irony import load_datasets as load_irony_datasets
from .lince.ner import load_datasets as load_ner_datasets
from .lince.pos import load_datasets as load_pos_datasets
from .lince.sentiment import load_datasets as load_sentiment_datasets
from .sentipolc import load_datasets as load_sentipolc_datasets

load_fun = {
    "hate_speech": {
        "es": load_hate_datasets,
        "en": load_hate_datasets,
    },
    "sentiment": {
        "es": load_tass_datasets,
        "en": load_semeval_datasets,
        "it": load_sentipolc_datasets,
    },
    "emotion": {
        "es": load_emotion_datasets,
        "en": load_emotion_datasets,
    },

    "irony": {
        "es": load_irony_datasets,
        "en": load_irony_datasets,
        "it": load_irony_datasets,
    },

    # We use multilingual LinCE dataset here
    "ner": {
        "es": load_ner_datasets,
        "en": load_ner_datasets,
    },

    # We use multilingual LinCE dataset here
    "pos": {
        "es": load_pos_datasets,
        "en": load_pos_datasets,
    },

    "lince_sentiment": {
        "es": load_sentiment_datasets,
        "en": load_sentiment_datasets,
    },
}

tasks = list(load_fun.keys())


def load_datasets(task, lang, **kwargs):
    """
    Load datasets for a given task and language
    """
    return load_fun[task][lang](lang=lang, **kwargs)
