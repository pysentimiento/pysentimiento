import torch
from .preprocessing import preprocess_tweet
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import functional as F

models = {
    "es": {
        "sentiment": {
            "model_name": "finiteautomata/beto-sentiment-analysis",
        },
        "emotion": {
            "model_name": "finiteautomata/beto-emotion-analysis",
        }
    },
    "en": {
        "sentiment": {
            "model_name": "finiteautomata/bertweet-base-sentiment-analysis",
            # BerTweet uses different preprocessing args
            "preprocessing_args": {"user_token": "@USER", "url_token": "HTTPURL"}
        },
        "emotion": {
            "model_name": "finiteautomata/bertweet-base-emotion-analysis",
            "preprocessing_args": {"user_token": "@USER", "url_token": "HTTPURL"}
        }
    },
}

class BaseOutput:
    """
    Base class for classification output
    """
    def __init__(self, sentence, idx, probas):
        """
        Constructor
        """
        self.sentence = sentence
        self.idx = idx
        self.probas = probas
        self.output = max(probas.items(), key=lambda x: x[1])[0]

    def __repr__(self):
        ret = f"{self.__class__.__name__}"
        formatted_probas = sorted(self.probas.items(), key=lambda x: -x[1])
        formatted_probas = [f"{k}: {v:.3f}" for k, v in formatted_probas]
        formatted_probas = "{" + ", ".join(formatted_probas) + "}"
        ret += f"(output={self.output}, probas={formatted_probas})"

        return ret

class SentimentOutput(BaseOutput):
    pass

class EmotionOutput(BaseOutput):
    pass


class Analyzer:
    """
    Wrapper to use sentiment analysis models as black-box
    """
    def __init__(self, model_name, preprocessing_args={}):
        """
        Constructor for SentimentAnalyzer class
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.id2label = self.model.config.id2label
        self.preprocessing_args = preprocessing_args

    def predict(self, sentence):
        """
        Return most likely class for the sentence
        """
        sentence = preprocess_tweet(sentence, **self.preprocessing_args)
        idx = torch.LongTensor(self.tokenizer.encode(sentence)).view(1, -1)
        output = self.model(idx)
        probs = F.softmax(output.logits, dim=1).view(-1)
        probas = {self.id2label[i]:probs[i].item() for i in self.id2label}

        return self.__class__.output_class(sentence, idx=idx, probas=probas)



class SentimentAnalyzer(Analyzer):
    """
    Dummy class for sentiment analyzer
    """
    output_class = SentimentOutput
    def __init__(self, lang, model_name=None, preprocessing_args={}):
        """
        Class used to apply Sentiment Analysis to a tweet.

        Currently supports 'es' (Spanish) and 'en' (English)

        Arguments
        ---------

        lang: str
            Language used for the classifier. Must be one of 'es', 'en'

        model_name: str (default None)
            If given, overrides default model_name for the language. Must be a local path or a huggingface's hub model name

        preprocessing_args: dict (default {})
            arguments to `preprocessing` function
        """
        if lang not in models:
            raise ValueError(f"{lang} must be in {list(models.keys())}")

        preprocessing_args["lang"] = lang

        if not model_name:
            model_info = models[lang]["sentiment"]
            model_name = model_info["model_name"]
            preprocessing_args.update(model_info.get("preprocessing_args", {}))

        super().__init__(model_name, preprocessing_args=preprocessing_args)

class EmotionAnalyzer(Analyzer):
    """
    Dummy class for emotion analyzer
    """
    output_class = EmotionOutput
    def __init__(self, lang, model_name=None):
        """
        Class used to apply Emotion Analysis to a tweet.

        Currently supports 'es' (Spanish) and 'en' (English)

        Arguments
        ---------

        lang: str
            Language used for the classifier. Must be one of 'es', 'en'

        model_name: str (default None)
            If given, overrides default model_name for the language. Must be a local path or a huggingface's hub model name
        """
        if lang not in models:
            raise ValueError(f"{lang} must be in {list(models.keys())}")
        if not model_name:
            model_info = models[lang]["emotion"]
            model_name = model_info["model_name"]
            preprocessing_args = model_info.get("preprocessing_args", {})

        super().__init__(model_name, preprocessing_args=preprocessing_args)
