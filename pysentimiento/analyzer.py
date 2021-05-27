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
        max_idx = torch.argmax(output.logits).item()
        return self.id2label[max_idx]

    def predict_probas(self, sentence):
        """
        Return softmax probabilities for each class
        """
        sentence = preprocess_tweet(sentence, **self.preprocessing_args)
        idx = torch.LongTensor(self.tokenizer.encode(sentence)).view(1, -1)
        output = self.model(idx)
        probs = F.softmax(output.logits, dim=1).view(-1)
        return {self.id2label[i]:probs[i].item() for i in self.id2label}



class SentimentAnalyzer(Analyzer):
    """
    Dummy class for sentiment analyzer
    """
    def __init__(self, lang, model_name=None):
        """
        Class used to apply Sentiment Analysis to a tweet.

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
            model_info = models["es"]["sentiment"]
            model_name = model_info["model_name"]
            preprocessing_args = model_info.get("preprocessing_args", {})

        super().__init__(model_name, preprocessing_args=preprocessing_args)

class EmotionAnalyzer(Analyzer):
    """
    Dummy class for emotion analyzer
    """
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
            model_info = models["es"]["emotion"]
            model_name = model_info["model_name"]
            preprocessing_args = model_info.get("preprocessing_args", {})

        super().__init__(model_name, preprocessing_args=preprocessing_args)
