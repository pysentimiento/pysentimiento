from hashformers.segmenter.auto import FastWordSegmenter
from hashformers.segmenter import TweetSegmenter, TwitterTextMatcher

def create_segmenter(lang='en'):
    """
    Create a word segmenter.

    Arguments:
    ----------
    lang: str
        Language code.
        For a full list of the supported languages and their language codes, 
        check the `wordfreq library documentation <https://pypi.org/project/wordfreq/>`_.

    Returns:
    --------
        
    """

    word_segmenter = FastWordSegmenter(
        unigram_lang=lang,
        reranker_model_name_or_path=None
    )

    twitter_matcher = TwitterTextMatcher()

    tweet_segmenter = TweetSegmenter(
        matcher=twitter_matcher,
        word_segmenter=word_segmenter
    )
    
    return tweet_segmenter