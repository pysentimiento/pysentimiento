from hashformers.segmenter.auto import TransformerWordSegmenter
from hashformers.segmenter import TweetSegmenter, TwitterTextMatcher

models = {
    "es": "DeepESP/gpt2-spanish",
    "en": "distilgpt2"
}

def create_segmenter(lang='en', model_name=None, batch_size=1000):
    """
    Create a word segmenter.

    Arguments:
    ----------
    lang: str
        Language code (es or en)
    model_name: str
        Model name or path

    Returns:
    --------
        
    """
    if lang not in models:
        raise ValueError(f"Language {lang} not supported -- only supports {models.keys()}")

    if not model_name:
        segmenter_model_name_or_path = models[lang]
    else:
        segmenter_model_name_or_path = model_name
    
    word_segmenter = TransformerWordSegmenter(
        segmenter_model_name_or_path = segmenter_model_name_or_path,
        segmenter_model_type = "gpt2",
        segmenter_device = "cuda",
        segmenter_gpu_batch_size = batch_size,
        reranker_gpu_batch_size = None,
        reranker_model_name_or_path = None,
        reranker_model_type = None
    )

    twitter_matcher = TwitterTextMatcher()

    tweet_segmenter = TweetSegmenter(
        matcher=twitter_matcher,
        word_segmenter=word_segmenter
    )
    
    return tweet_segmenter