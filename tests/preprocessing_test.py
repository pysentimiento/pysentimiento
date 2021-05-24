import pytest
from pysentimiento.preprocessing import preprocess_tweet, camel_to_human

def test_preprocessing_replaces_users():
    """
    Replaces handles with special token for user
    """
    text = "@perezjotaeme debería cambiar esto"

    assert preprocess_tweet(text) == "[USER] debería cambiar esto"

def test_preprocessing_replaces_users_twice():
    """
    Replaces handles with special token for user
    """
    text = "@perezjotaeme @perezjotaeme debería cambiar esto"

    assert preprocess_tweet(text) == "[USER] [USER] debería cambiar esto"

def test_preprocessing_replaces_urls():
    """
    Replaces urls with special token for url
    """
    text = "esto es muy bueno http://bit.ly/sarasa"

    assert preprocess_tweet(text) == "esto es muy bueno [URL]"

def test_shortens_repeated_characters():
    """
    Replaces urls with special token for url
    """
    text = "no entiendo naaaaaaaadaaaaaaaa"

    assert preprocess_tweet(text, shorten=2) == "no entiendo naadaa"

def test_shortens_laughters():
    """
    Replaces laughters
    """

    text = "jajajajaajjajaajajaja no lo puedo creer ajajaj"
    assert preprocess_tweet(text) == "jaja no lo puedo creer jaja"

def test_replaces_odd_quotation_marks():
    """

    Replaces “ -> "

    """
    text = "Pero pará un poco, “loquita”"

    assert preprocess_tweet(text) == 'Pero pará un poco, "loquita"'


def test_preprocessing_handles_hashtags():
    """
    Replaces hashtags with text
    """
    text = "esto es #UnaGenialidad"

    assert preprocess_tweet(text) == "esto es una genialidad"

def test_camel_to_human_on_simple_camel():
    """
    Test camel to human
    """

    assert camel_to_human("CamelToHuman") == "camel to human"

def test_camel_to_human_with_numbers():
    """
    Test camel to human
    """

    assert camel_to_human("1stToDie") == "1st to die"


def test_camel_to_human_no_upper():
    """
    Test camel to human
    """

    assert camel_to_human("thisisatest") == "thisisatest"