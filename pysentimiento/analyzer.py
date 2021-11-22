import torch
from .preprocessing import preprocess_tweet
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
    Trainer, TrainingArguments
)
from datasets import Dataset
from torch.nn import functional as F


models = {
    "es": {
        "sentiment": {
            "model_name": "pysentimiento/robertuito-sentiment-analysis",
        },
        "emotion": {
            "model_name": "pysentimiento/robertuito-emotion-analysis",
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

class AnalyzerOutput:
    """
    Base class for classification output
    """
    def __init__(self, sentence, probas):
        """
        Constructor
        """
        self.sentence = sentence
        self.probas = probas
        self.output = max(probas.items(), key=lambda x: x[1])[0]

    def __repr__(self):
        ret = f"{self.__class__.__name__}"
        formatted_probas = sorted(self.probas.items(), key=lambda x: -x[1])
        formatted_probas = [f"{k}: {v:.3f}" for k, v in formatted_probas]
        formatted_probas = "{" + ", ".join(formatted_probas) + "}"
        ret += f"(output={self.output}, probas={formatted_probas})"

        return ret


class Analyzer:
    """
    Wrapper to use sentiment analysis models as black-box
    """
    def __init__(self, model_name, task, preprocessing_args={}, batch_size=32):
        """
        Constructor for SentimentAnalyzer class

        Arguments:

        model_name: str or path
            Model name or
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length=128
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.id2label = self.model.config.id2label
        self.preprocessing_args = preprocessing_args
        self.batch_size = batch_size
        self.task = task

        self.eval_trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=".",
                per_device_eval_batch_size=batch_size,
            ),
            data_collator=DataCollatorWithPadding(self.tokenizer, padding="longest"),
        )

    def _tokenize(self, batch):
        return self.tokenizer(
            batch["text"], padding=False, truncation=True
        )

    def _predict_single(self, sentence):
        """
        Predict single

        Do it this way (without creating dataset) to make it faster
        """
        device = self.eval_trainer.args.device
        sentence = preprocess_tweet(sentence, **self.preprocessing_args)
        idx = torch.LongTensor(
            self.tokenizer.encode(
                sentence, truncation=True,
                max_length=self.tokenizer.model_max_length,
            )
        ).view(1, -1).to(device)
        output = self.model(idx)
        probs = F.softmax(output.logits, dim=1).view(-1)
        probas = {self.id2label[i]:probs[i].item() for i in self.id2label}

        return AnalyzerOutput(sentence, probas=probas)


    def predict(self, inputs):
        """
        Return most likely class for the sentence

        Arguments:
        ----------
        inputs: string or list of strings
            A single or a list of sentences to be predicted

        Returns:
        --------
            List or single output with probabilities
        """
        if isinstance(inputs, str):
            return self._predict_single(inputs)


        sentences = [
            preprocess_tweet(sent, **self.preprocessing_args) for sent in inputs
        ]
        dataset = Dataset.from_dict({"text": sentences})
        dataset = dataset.map(self._tokenize, batched=True, batch_size=self.batch_size)


        output = self.eval_trainer.predict(dataset)
        logits = torch.tensor(output.predictions)

        probs = F.softmax(logits, dim=1)

        probas = [{
            self.id2label[i]: probs[j][i].item()
            for i in self.id2label
        } for j in range(probs.shape[0])]


        rets = [AnalyzerOutput(sent, prob) for sent, prob in zip(sentences, probas) ]

        return rets


def create_analyzer(task, lang, model_name=None, preprocessing_args={}):
    """
    Create analyzer for the given task

    Arguments:
    ----------
    task: str
        Task name (sentiment or emotion)
    lang: str
        Language code (es or en)
    model_name: str
        Model name or path
    preprocessing_args: dict
        Preprocessing arguments

    Returns:
    --------
        SentimentAnalyzer or EmotionAnalyzer
    """
    if lang not in models:
        raise ValueError(f"Language {lang} not supported -- only supports {models.keys()}")

    if task not in models[lang]:
        raise ValueError(f"Task {task} not supported for {lang} -- only supports {models[lang].keys()}")

    preprocessing_args = preprocessing_args or {}

    if not model_name:
        model_info = models[lang][task]
        model_name = model_info["model_name"]
        preprocessing_args.update(model_info.get("preprocessing_args", {}))

    preprocessing_args["lang"] = lang
    return Analyzer(model_name, task, preprocessing_args)
