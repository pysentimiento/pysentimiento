import emoji
import re

special_tokens = [
    "[USER]",
    "[HASHTAG]",
    "[EMOJI]",
]


user_regex = re.compile(r"@[a-zA-Z0-9_]{0,15}")
url_regex = re.compile(
    r"((?<=[^a-zA-Z0-9])(?:https?\:\/\/|[a-zA-Z0-9]{1,}\.{1}|\b)(?:\w{1,}\.{1}){1,5}(?:com|co|org|edu|gov|uk|net|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|mil|iq|io|ac|ly|sm){1}(?:\/[a-zA-Z0-9]{1,})*)"
)

hashtag_regex = re.compile(r'\B#(\w*[a-zA-Z]+\w*)')
start_of_camel = re.compile(r'([A-Z]+)')

def camel_to_human(s, lower=True):
    """
    Converts camel case to 'human' case

    Arguments:
    ----------

    lower: bool (default: False)
        Convert output to lower
    """

    ret = start_of_camel.sub(r' \1', s).strip()

    if lower:
        ret = ret.lower()

    return ret

emoji_regex = re.compile(r"\|([^\|]+)\|")
laughter_regex = re.compile("[ja][ja]+aj[ja]+")

def convert_emoji_to_text(x):
    """
    """
    return "[EMOJI] " + " ".join(x.groups()[0].split("_")) + " [EMOJI]"


replacements = {
    "~": None,
    "£": None,
    "¥": None,
    "¨": '"',
    "©": None,
    "«": '"',
    "°": None,
    "´": None,
    "¶": None,
    "·": None,
    "º": None,
    "»": '"',
    "×": None,
    "ا": None,
    "–": None,
    "—": None,
    "‘": None,
    "’": None,
    "“": '"',
    "”": '"',
    "•": None,
    "…": None,
    "‼": "!",
    "€": "$",
    "™": None,
    "●": None,
    "☠": None,
    "☹": None,
    "☺": None,
    "☻": "[EMOJI] carita feliz [EMOJI]",
    "♀": None,
    "♂": None,
    "♡": "corazón",
    "♥": "corazón",
    "⚰": None,
    "⛱": None,
    "⛹": None,
    "✈": None,
    "✓": None,
    "❤": None,
    "ー": None,
    "🕯": None,
    "🛰": None,
}


def preprocess_tweet(
    text, user_token="[USER]", url_token="[URL]", preprocess_hashtags=True, hashtag_token=None,
    demoji=True, shorten=3, normalize_laughter=True):
    """
    Basic preprocessing

    Arguments:
    ---------
    shorten: int (default: 3)

    Convert all occurrences of a character *shorten* or more times to just *shorten* times
    """
    text = user_regex.sub(user_token, text)
    text = url_regex.sub(url_token, text)

    if shorten:
        repeated_regex = re.compile(r"(.)"+ r"\1" * (shorten-1) + "+")
        text = repeated_regex.sub(r"\1"*shorten, text)

    if demoji:
        text = emoji.demojize(text, language="es", delimiters=("|", "|"))


        text = emoji_regex.sub(
            convert_emoji_to_text,
            text
        )

    if normalize_laughter:
        text = laughter_regex.sub(
            "jaja",
            text
        )

    def process_hashtags(x):
        """
        Hashtag preprocessing function

        Take first group and decamelize
        """


        text = x.groups()[0]

        text = camel_to_human(text)

        if hashtag_token:
            text = hashtag_token + " " + text

        return text

    if preprocess_hashtags:
        text = hashtag_regex.sub(
            process_hashtags,
            text
        )

    ret = ""
    for char in text:
        if char in replacements:
            replacement = replacements[char]
            if replacement:
                ret += replacement
        else:
            ret += char
    text = ret

    return text