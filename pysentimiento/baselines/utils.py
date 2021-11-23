import torch
from torchtext.vocab import build_vocab_from_iterator
import unidecode

def build_vocab(dataset, tokenizer):
    """
    Build vocabulary 

    Arguments:

    dataset: datasets.Dataset
        The dataset to extract tokens from

    tokenizer: torchtext.tokenizer
        Tokenizer
    """
    def get_tokens():
        for example in dataset:
            yield tokenizer(unidecode.unidecode(example["text"].lower()))


    vocab = build_vocab_from_iterator(get_tokens(), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])

    return vocab

def build_embedding_matrix(vocab, fasttext_model):
    """
    Builds embedding matrix from a fasttext model

    """
    DIM = fasttext_model.get_word_vector("random").shape[0]
    emb_matrix = torch.randn(len(vocab), DIM)
    stoi = vocab.get_stoi()
    itos = vocab.get_itos()
    UNK_IDX = stoi["<unk>"]
    PAD_IDX = stoi["<pad>"]

    itos = vocab.get_itos()

    # emb_matrix[UNK_IDX] = 0
    emb_matrix[PAD_IDX] = 0

    for i, word in enumerate(itos):
        if i == UNK_IDX or i == PAD_IDX:
            # Let them unmodified
            pass
        else:
            emb_matrix[i] = torch.tensor(fasttext_model.get_word_vector(word))
    return emb_matrix
