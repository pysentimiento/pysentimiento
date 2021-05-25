import torch
from .preprocessing import preprocess_tweet
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import functional as F

class Analyzer:
    """
    Wrapper to use sentiment analysis models as black-box
    """
    def __init__(self, model_name):
        """
        Constructor for SentimentAnalyzer class
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.id2label = self.model.config.id2label

    def predict(self, sentence):
        """
        Return most likely class for the sentence
        """
        sentence = preprocess_tweet(sentence)
        idx = torch.LongTensor(self.tokenizer.encode(sentence)).view(1, -1)
        output = self.model(idx)
        max_idx = torch.argmax(output.logits).item()
        return self.id2label[max_idx]

    def predict_probas(self, sentence):
        """
        Return softmax probabilities for each class
        """
        sentence = preprocess_tweet(sentence)
        idx = torch.LongTensor(self.tokenizer.encode(sentence)).view(1, -1)
        output = self.model(idx)
        probs = F.softmax(output.logits, dim=1).view(-1)
        return {self.id2label[i]:probs[i].item() for i in self.id2label}

class SentimentAnalyzer(Analyzer):
    """
    Dummy class for sentiment analyzer
    """
    def __init__(self, model_name="finiteautomata/beto-sentiment-analysis"):
        super().__init__(model_name)

class EmotionAnalyzer(Analyzer):
    """
    Dummy class for emotion analyzer
    """
    def __init__(self, model_name="finiteautomata/beto-emotion-analysis"):
        super().__init__(model_name)

