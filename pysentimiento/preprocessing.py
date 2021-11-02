import emoji
import re

extra_args = {
    "vinai/bertweet-base":  {
        "user_token": "@USER",
        "url_token": "HTTPURL",
    }
}



special_tokens = ["@usuario", "url", "hashtag", "emoji"]


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

def convert_emoji_to_text(x, emoji_wrapper="[EMOJI]"):
    """
    """
    return f" {emoji_wrapper} " + " ".join(x.groups()[0].split("_")) + f" {emoji_wrapper} "


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
    }
}



def preprocess_tweet(
    text, lang="es", user_token="@usuario", url_token="url", preprocess_hashtags=True, hashtag_token=None,
    demoji=True, shorten=3, normalize_laughter=True, emoji_wrapper="emoji"):
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

    demoji: boolean (default True)
        If true, converts emoji to text representations using `emoji` library, and wraps this with "[EMOJI]" strings

    normalize_laughter: boolean (default True)
        Normalizes laughters. Uses different regular expressions depending on the lang argument.
    """
    if lang == "en" and user_token == "@usuario":
        """
        If it is english and we didn't set any defaults, we set the vinai/bertweet-base defaults
        """
        user_token = "@USER"
        url_token = "HTTPURL"


    ret = ""
    for char in text:
        if char in replacements:
            replacement = replacements[char]
            if replacement:
                ret += replacement
        else:
            ret += char
    text = ret

    text = user_regex.sub(user_token, text)
    text = url_regex.sub(url_token, text)

    if shorten:
        repeated_regex = re.compile(r"(.)"+ r"\1" * (shorten-1) + "+")
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

    return text.strip()
