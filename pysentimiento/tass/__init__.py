from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .datasets import load_datasets, id2label, label2id


def load_model(base_model, device):
    """
    Loads model and tokenizer
    """
    print(f"Loading model {base_model}")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, return_dict=True, num_labels=3
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.model_max_length = 128

    #model.config.hidden_dropout_prob = 0.20
    model.config.id2label = id2label
    model.config.label2id = label2id

    model = model.to(device)
    model.train()

    return model, tokenizer