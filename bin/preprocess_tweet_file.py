import os
import glob
import fire
import subprocess
from tqdm.auto import tqdm
import multiprocessing
from pysentimiento.preprocessing import preprocess_tweet

def my_preprocess(*args):
    return preprocess_tweet(*args, **{
        "user_token": "USUARIO",
        "url_token": "URL",
        "hashtag_token": "hashtag",
        "emoji_wrapper": "",
    })

def preprocess_tweet_files(path, output, workers=1):
    """
    Apply preprocessing to all tweets in file
    """

    batch = int(1e6)

    num = subprocess.check_output(['wc', '-l', path])
    num = int(num.decode("utf-8").split(' ')[0])


    pool = multiprocessing.Pool(processes=workers)

    pbar = tqdm(total=num)



    with open(output, "w+") as output_file:
        with open(path) as f:
            for processed_tweet in pool.imap(my_preprocess, f):
                output_file.write(processed_tweet)
                pbar.update()



if __name__ == "__main__":
    fire.Fire(preprocess_tweet_files)
