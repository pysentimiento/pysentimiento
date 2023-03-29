import emoji
import re


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


def convert_emoji_to_text(x, emoji_wrapper="emoji"):
    """
    Converts emoji to text
    """
    wrapper = f" {emoji_wrapper} ".replace("  ", " ")
    return wrapper + " ".join(x.groups()[0].split("_")) + wrapper


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
    "☻": "😃",
    "♀": None,
    "♂": None,
    "♡": "❤️",
    "♥": "❤️",
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

laughter_conf = {
    "es": {
        "regex": re.compile("[ja][ja]+aj[ja]+"),
        "replacement": "jaja",
    },
    "en": {
        "regex": re.compile("[ha][ha]+ah[ha]+"),
        "replacement": "haha",
    },
    "it": {
        "regex": re.compile("[ha][ha]+ah[ha]+"),
        "replacement": "haha",
    },

    "pt": {
        "regex": re.compile("[ha][ha]+ah[ha]+|kk+"),
        "replacement": "kk",
    }
}

default_args = {
    "es": {
        "user_token": "@usuario",
        "url_token": "url",
        "hashtag_token": "hashtag",
    },

    "en": {
        "user_token": "@USER",
        "url_token": "HTTPURL",
        "hashtag_token": "hashtag",
    },

    "it": {
        "user_token": "##user",
        "url_token": "##url",
        "hashtag_token": "##hashtag",
    },

    "pt": {
        "user_token": "@USER",
        "url_token": "HTTPURL",
        "hashtag_token": "hashtag",
    }
}


special_tokens = {
    k: list(v.values()) for k, v in default_args.items()
}

model_preprocessing_args = {
    "pysentimiento/robertuito-base-uncased": default_args["es"],
    "pysentimiento/robertuito-base-cased": default_args["es"],
    "pysentimiento/robertuito-base-deacc": default_args["es"],
    "melll-uff/bertweetbr": default_args["en"],  # Same as BERTweet
}


def get_preprocessing_args(model_name, lang):
    """
    Gets preprocessing args for model_name and lang

    Returns default_args[lang] if model_name has not special params
    """
    return model_preprocessing_args.get(model_name, default_args[lang])


def preprocess_tweet(
        text, lang="es", user_token=None, url_token=None, preprocess_hashtags=True, hashtag_token=None, char_replace=True,
        demoji=True, shorten=3, normalize_laughter=True, emoji_wrapper="emoji", preprocess_handles=True):
    """
    Basic preprocessing

    Arguments:
    ---------

    text: str
        Text to preprocess

    lang: str (default 'es')
        Language used in the preprocessing. This is used for the demoji functionality and laughter preprocessing

    user_token: str (default "[USER]")
        Token used to replace user handles

    url_token: str (default "[URL]")
        Token used to replace urls

    preprocess_hashtags: boolean (default True)
        If true, applies preprocessing to hashtag, trying to split camel cases

    hashtag_token: str (default None)
        If preprocess_hashtags is True, adds hashtag_token before the preprocessed content of the hashtag

    shorten: int (default: 3)
        If not none, all occurrences of shorten or more characters are cut to this number

    char_replace: bool (default: True)
        If true, replaces or removes special characters to equivalent ones.

    demoji: boolean (default True)
        If true, converts emoji to text representations using `emoji` library, and wraps this with "emoji" tokens

    normalize_laughter: boolean (default True)
        Normalizes laughters. Uses different regular expressions depending on the lang argument.

    preprocess_handles: boolean (default True)
        If true, replaces user handles with user_token
    """

    user_token = user_token or default_args[lang]["user_token"]
    url_token = url_token or default_args[lang]["url_token"]
    hashtag_token = hashtag_token or default_args[lang]["hashtag_token"]

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

    if char_replace:
        ret = ""
        for char in text:
            if char in replacements:
                replacement = replacements[char]
                if replacement:
                    ret += replacement
            else:
                ret += char
        text = ret

    if preprocess_handles:
        text = user_regex.sub(user_token, text)

    text = url_regex.sub(url_token, text)

    if shorten:
        repeated_regex = re.compile(r"(.)" + r"\1" * (shorten-1) + "+")
        text = repeated_regex.sub(r"\1"*shorten, text)

    if demoji:
        text = emoji.demojize(text, language=lang, delimiters=("|", "|"))
        text = emoji_regex.sub(
            lambda x: convert_emoji_to_text(x, emoji_wrapper=emoji_wrapper),
            text
        )

    if normalize_laughter:
        laughter_regex = laughter_conf[lang]["regex"]
        replacement = laughter_conf[lang]["replacement"]

        text = laughter_regex.sub(
            replacement,
            text
        )

    return text.strip()
