import pytest
from pysentimiento.lince.ner import align_labels_with_tokens, preprocess_token

"""
Test for preprocessing of tokens
"""

def test_preprocessing_emoji():
    assert preprocess_token("🤔", "es") == "emoji"


def test_preprocessing_multiple_emojis():
    assert preprocess_token("🤔🤔🤔", "es") == "emoji"


def test_unicode_strange_char():
    assert preprocess_token('️', "es") == "."

"""
Tests for alignment

align_labels_with_tokens takes a list of labels and a list of word_ids for each token
"""
def test_align_for_just_one_word():
    assert align_labels_with_tokens([1], [None, 0, None]) == [-100, 1, -100]

def test_align_for_one_word_with_two_subtokens():
    aligned_labels = align_labels_with_tokens([1], [None, 0, 0, None], label_subwords=True)
    assert aligned_labels == [-100, 1, 2, -100]

def test_align_for_one_word_with_two_subtokens_no_subword():
    aligned_labels = align_labels_with_tokens([1], [None, 0, 0, None], label_subwords=False)
    assert aligned_labels == [-100, 1, -100, -100]


def test_align_with_subword():
    aligned_labels = align_labels_with_tokens(
        [0,1,0,3,4],
        [None, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4, None],
        label_subwords=True
    )
    assert aligned_labels == [-100, 0, 0, 1, 2, 0, 0, 3, 4, 4, 4, -100]

def test_align_no_subword():
    aligned_labels = align_labels_with_tokens(
        [0,1,0,3,4],
        [None, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4, None],
        label_subwords=False
    )
    assert aligned_labels == [-100, 0, -100, 1, -100, 0, -100, 3, 4, -100, -100, -100]

def test_align_complex_entity():
    aligned_labels = align_labels_with_tokens(
        [0,1,2,2],
        [None, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, None],
        label_subwords=False
    )
    assert aligned_labels == [-100, 0, -100, 1, -100, -100, 2, -100, 2, -100, -100, -100]