import torch
from .preprocessing import preprocess_tweet
import transformers
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification,
    DataCollatorWithPadding,
    Trainer, TrainingArguments,
    pipeline
)
from datasets import Dataset
from torch.nn import functional as F


transformers.logging.set_verbosity_error()

models = {
    "es": {
        "sentiment": {
            "model_name": "pysentimiento/robertuito-sentiment-analysis",
        },
        "emotion": {
            "model_name": "pysentimiento/robertuito-emotion-analysis",
        },
        "hate_speech": {
            "model_name": "pysentimiento/robertuito-hate-speech",
        },

        "irony": {
            "model_name": "pysentimiento/robertuito-irony",
        },
        "ner": {
            "model_name": "pysentimiento/robertuito-ner",
        },
        "pos": {
            "model_name": "pysentimiento/robertuito-pos",
        },

        "targeted_sentiment": {
            "model_name": "pysentimiento/roberta-targeted-sentiment-analysis"
        },

        # TODO: this model has no benchmark
        "context_hate_speech": {
            "model_name": "piuba-bigdata/beto-contextualized-hate-speech"
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
        },
        "hate_speech": {
            "model_name": "pysentimiento/bertweet-hate-speech",
            "preprocessing_args": {"user_token": "@USER", "url_token": "HTTPURL"}
        },

        "irony": {
            "model_name": "pysentimiento/bertweet-irony",
        },
        "ner": {
            "model_name": "pysentimiento/robertuito-ner",
        },
        "pos": {
            "model_name": "pysentimiento/robertuito-pos",
        }
    },

    "it": {
        "sentiment": {
            "model_name": "pysentimiento/bert-it-sentiment",
        },
        "emotion": {
            "model_name": "pysentimiento/bert-it-emotion",
        },

        "hate_speech": {
            "model_name": "pysentimiento/bert-it-hate-speech",
        },
        "irony": {
            "model_name": "pysentimiento/bert-it-irony",
        },
    },

    "pt": {
        "sentiment": {
            "model_name": "pysentimiento/bertweet-pt-sentiment",
        },

        "hate_speech": {
            "model_name": "pysentimiento/bertabaporu-pt-hate-speech",
        },

        "irony": {
            "model_name": "pysentimiento/bertabaporu-pt-irony",
        }
    }
}


class AnalyzerOutput:
    """
    Base class for classification output
    """

    def __init__(self, sentence, context, probas, is_multilabel=False):
        """
        Constructor
        """
        self.sentence = sentence
        self.probas = probas
        self.context = context
        self.is_multilabel = is_multilabel
        if not is_multilabel:
            self.output = max(probas.items(), key=lambda x: x[1])[0]
        else:
            self.output = [
                k for k, v in probas.items() if v > 0.5
            ]

    def __repr__(self):
        ret = f"{self.__class__.__name__}"
        if not self.is_multilabel:
            formatted_probas = sorted(self.probas.items(), key=lambda x: -x[1])
        else:
            formatted_probas = list(self.probas.items())
        formatted_probas = [f"{k}: {v:.3f}" for k, v in formatted_probas]
        formatted_probas = "{" + ", ".join(formatted_probas) + "}"
        ret += f"(output={self.output}, probas={formatted_probas})"

        return ret


class TokenClassificationOutput:
    """
    Output for token classification
    """

    def __init__(self, sentence, tokens, labels, probas, entities=None):
        """
        Constructor
        """
        self.sentence = sentence
        self.tokens = tokens
        self.labels = labels
        self.entities = entities
        self.probas = probas

    def __repr__(self):
        ret = f"{self.__class__.__name__}"

        if self.entities:
            formatted_entities = ", ".join(
                f'{entity["text"]} ({entity["type"]})' for entity in self.entities
            )
            ret += f"(entities=[{formatted_entities}], tokens={self.tokens}, labels={self.labels})"
        else:
            ret += f"(tokens={self.tokens}, labels={self.labels})"

        return ret


class BaseAnalyzer:
    def __init__(self, model, tokenizer, task, preprocessing_args={}, batch_size=32, **kwargs):
        """
        Constructor for SentimentAnalyzer class

        Arguments:

        model_name: str or path
            Model name or
        """
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessing_args = preprocessing_args
        self.batch_size = batch_size
        self.task = task

        self.tokenizer.model_max_length = 128
        self.problem_type = self.model.config.problem_type
        self.id2label = self.model.config.id2label

        self.eval_trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=".",
                per_device_eval_batch_size=batch_size,
            ),
            data_collator=DataCollatorWithPadding(
                self.tokenizer, padding="longest"),
        )

    def _tokenize(self, batch):
        # If context is present, use it
        if "context" in batch:
            inputs = [batch["text"], batch["context"]]
        else:
            inputs = [batch["text"]]
        return self.tokenizer(
            *inputs, padding=False, truncation=True,
            max_length=self.tokenizer.model_max_length
        )


class AnalyzerForSequenceClassification(BaseAnalyzer):
    """
    Wrapper to use sentiment analysis models as black-box
    """

    @classmethod
    def from_model_name(cls, model_name, task, preprocessing_args={}, batch_size=32, **kwargs):
        """
        Constructor for SentimentAnalyzer class

        Arguments:

        model_name: str or path
            Model name or
        """
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls(model, tokenizer, task, preprocessing_args, batch_size, **kwargs)

    def _get_output(self, sentence, logits, context=None):
        """
        Get output from logits

        It takes care of the type of problem: single or multi label classification
        """
        if self.problem_type == "multi_label_classification":
            is_multilabel = True
            probs = torch.sigmoid(logits).view(-1)
        else:
            is_multilabel = False
            probs = torch.softmax(logits, dim=1).view(-1)

        probas = {self.id2label[i]: probs[i].item() for i in self.id2label}
        return AnalyzerOutput(sentence, probas=probas, is_multilabel=is_multilabel, context=context)

    def _predict_single(self, sentence, context, preprocess_context):
        """
        Predict single

        Do it this way (without creating dataset) to make it faster
        """
        device = self.eval_trainer.args.device
        sentence = preprocess_tweet(sentence, **self.preprocessing_args)
        inputs = [sentence]

        if context:
            if preprocess_context:
                context = preprocess_tweet(context, **self.preprocessing_args)
            inputs.append(context)
        idx = torch.LongTensor(
            self.tokenizer.encode(
                *inputs,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            )
        ).view(1, -1).to(device)
        output = self.model(idx)
        logits = output.logits
        return self._get_output(sentence, logits)

    def predict(self, inputs, context=None, target=None, preprocess_context=True):
        """
        Return most likely class for the sentence

        Arguments:
        ----------
        inputs: string or list of strings
            A single or a list of sentences to be predicted

        context: string or list of strings
            A single or a list of context to be used for the prediction

        target: string or list of strings
            A rename of context

        preprocess_context: bool
            Whether to preprocess the context or not

        Returns:
        --------
            List or single AnalyzerOutput
        """

        context = context or target

        # If single string => predict it single
        if isinstance(inputs, str):
            if context and not isinstance(context, str):
                raise ValueError("Context must be a string")
            return self._predict_single(inputs, context=context, preprocess_context=preprocess_context)
        elif context and not isinstance(context, list):
            raise ValueError("Context must be a list of strings")
        elif context and len(context) != len(inputs):
            raise ValueError("Context and inputs must have the same length")

        data = {
            "text": [preprocess_tweet(sent, **self.preprocessing_args) for sent in inputs]
        }

        if context:
            data["context"] = [
                preprocess_tweet(
                    sent, **self.preprocessing_args
                ) if preprocess_context else sent
                for sent in context
            ]

        dataset = Dataset.from_dict(data)
        dataset = dataset.map(self._tokenize, batched=True,
                              batch_size=self.batch_size)

        output = self.eval_trainer.predict(dataset)
        logits = torch.tensor(output.predictions)

        if context is None:
            # Just to make this clearer
            data["context"] = [None] * len(data["text"])

        rets = [
            self._get_output(sent, logits_row.view(1, -1), context=context)
            for sent, context, logits_row in zip(data["text"], data["context"], logits)]

        return rets


class AnalyzerForTokenClassification(BaseAnalyzer):
    @classmethod
    def from_model_name(cls, model_name, task, chunk, preprocessing_args={}, batch_size=32, **kwargs):
        """
        Constructor for AnalyzerForTokenClassification class

        Arguments:

        model_name: str or path
            Model name or
        """
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls(model, tokenizer, task, preprocessing_args=preprocessing_args, batch_size=batch_size, chunk=chunk, **kwargs)

    def __init__(self, model, tokenizer, task, lang, chunk, preprocessing_args={}, batch_size=32):
        # Update preprocessing args if not provided

        preprocessing_args["preprocess_handles"] = preprocessing_args.get(
            "preprocess_handles", False)
        super().__init__(model, tokenizer, task, preprocessing_args, batch_size)

        if lang == "en":
            from spacy.lang.en import English
            nlp = English()
        elif lang == "es":
            from spacy.lang.es import Spanish
            nlp = Spanish()

        self.spacy_tokenizer = nlp.tokenizer
        self.chunk = chunk

    def decode(self, words, labels):
        """
        Convert BIO labels to segments
        Arguments:
        ----------
        words: list of str

        labels: list of str
            BIO labels


        Returns:
        --------
        entities: list of dict
            Each dict has keys "tokens" and "type"
        """
        entities = []
        current_words = None
        current_type = None
        for token, label in zip(words, labels):
            if label is None:
                # Perhaps it is too long
                continue
            elif label == 'O':
                if current_type == "O":
                    pass
                else:
                    # There was something
                    if current_words:
                        entities.append({
                            "tokens": current_words,
                            "type": current_type
                        })
                    current_words = []
                    current_type = "O"
            elif label.startswith('B-'):
                if current_words:
                    entities.append({
                        "tokens": current_words,
                        "type": current_type
                    })
                current_words = [token]
                current_type = label[2:]
            elif label.startswith('I-'):
                # If we are in the same type, add the word
                if current_type == label[2:]:
                    current_words.append(token)
                else:
                    if current_words:
                        entities.append({
                            "tokens": current_words,
                            "type": current_type
                        })
                    current_words = [token]
                    current_type = label[2:]
            else:
                # Handling non BIO labels
                if label != current_type:
                    if current_words:
                        entities.append({
                            "tokens": current_words,
                            "type": current_type
                        })
                    current_words = [token]
                    current_type = label
                else:
                    current_words.append(token)
                    current_type = label

        if current_words:
            entities.append({
                "tokens": current_words,
                "type": current_type
            })

        for segment in entities:
            segment["text"] = "".join(
                t.text + t.whitespace_ for t in segment["tokens"]
            ).strip()

            first_token = segment["tokens"][0]
            last_token = segment["tokens"][-1]

            segment["start"] = first_token.idx
            segment["end"] = last_token.idx + len(last_token.text)
            segment.pop("tokens")

        return entities

    def predict(self, inputs):
        """
        Predict token classification
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        sentences = [
            preprocess_tweet(sent, **self.preprocessing_args) for sent in inputs
        ]

        # First, tokenize with spacy
        # This is because the model seems to be working better with this tokenizer

        spacy_tokens = [
            [token for token in self.spacy_tokenizer(sentence)] for sentence in sentences
        ]

        tokens = [[token.text for token in sentence]
                  for sentence in spacy_tokens]

        tokenized_inputs = self.tokenizer(
            tokens, is_split_into_words=True, padding=True, truncation=True)

        model_device = next(self.model.parameters()).device

        model_output = self.model(**{k: torch.tensor(v).to(model_device)
                                     for k, v in tokenized_inputs.items()})

        outs = torch.argmax(model_output.logits, dim=2)
        id2label = self.model.config.id2label

        labels = []
        # Ignore subword tokens in the middle of the word
        # Just use the first token of each word
        for i, (sentence, output) in enumerate(zip(tokens, outs)):

            sentence_labels = [None for _ in sentence]
            word_ids = tokenized_inputs.word_ids(i)

            for word_id, label in zip(word_ids, output):
                if word_id is None:
                    continue
                token = sentence[word_id]
                # If it starts with "@" => it is a user
                if token.startswith("@"):
                    sentence_labels[word_id] = "B-USER"
                elif sentence_labels[word_id] is None:
                    sentence_labels[word_id] = id2label[label.item()]

            labels.append(sentence_labels)

        if self.chunk:
            entities = [self.decode(sentence, sentence_labels)
                        for sentence, sentence_labels in zip(spacy_tokens, labels)]
        else:
            # TODO: fix this to avoid issues for other tasks
            # Remove "B-" from labels
            labels = [[label[2:] if label.startswith(
                "B-") else label for label in sent_labels] for sent_labels in labels]
            entities = [None for _ in labels]

        outputs = []

        for sentence, sent_tokens, sent_labels, sent_entities, sent_outs in zip(inputs, tokens, labels, entities, model_output.logits):
            outputs.append(
                TokenClassificationOutput(
                    sentence=sentence,
                    tokens=sent_tokens,
                    labels=sent_labels,
                    entities=sent_entities if self.chunk else None,
                    probas=torch.softmax(
                        sent_outs, dim=-1).detach().cpu().numpy(),
                ))

        if len(sentences) == 1:
            return outputs[0]
        return outputs


def create_analyzer(task=None, lang=None, model_name=None, preprocessing_args={}, **kwargs):
    """
    Create analyzer for the given task

    Arguments:
    ----------
    task: str
        Task name ("sentiment", "emotion", "hate_speech", "irony", "ner", "pos")
    lang: str
        Language code (accepts "en", "es", "it", "pt". See documentation for further information)
    model_name: str
        Model name or path
    preprocessing_args: dict
        Preprocessing arguments for `preprocess_tweet` function

    Returns:
    --------
        Analyzer object for the given task and language
    """
    if not (model_name or (lang and task)):
        raise ValueError("model_name or (lang and task) must be provided")

    preprocessing_args = preprocessing_args or {}
    if task == "ner":
        analyzer_class = AnalyzerForTokenClassification
        kwargs["chunk"] = True
    elif task == "pos":
        analyzer_class = AnalyzerForTokenClassification
        kwargs["chunk"] = False
    else:
        analyzer_class = AnalyzerForSequenceClassification

    if not model_name:
        # Default models
        if lang not in models:
            raise ValueError(
                f"Language {lang} not supported -- only supports {models.keys()}")

        if task not in models[lang]:
            raise ValueError(
                f"Task {task} not supported for {lang} -- only supports {models[lang].keys()}")

        model_info = models[lang][task]
        model_name = model_info["model_name"]
        preprocessing_args.update(model_info.get("preprocessing_args", {}))

    preprocessing_args["lang"] = lang
    return analyzer_class.from_model_name(
        model_name=model_name,
        task=task,
        preprocessing_args=preprocessing_args,
        lang=lang,
        **kwargs)
