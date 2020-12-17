import os
import glob
import asyncio
import fire
import aiofiles
import tqdm.asyncio
from pysentimiento.preprocessing import preprocess_tweet

async def process_file(file):
    """
    Process each file
    """
    async with aiofiles.open(file, "r") as f:
        tweets = await f.read()
    tweets = tweets.split("\n")
    tweets = [preprocess_tweet(t) for t in tweets]

    async with aiofiles.open(file, "w+") as f:
        for t in tweets:
            await f.write(t + "\n")
    return

async def main_loop(files):
    """
    Asyncio event loop
    """
    flist = [
        process_file(file) for file in files
    ]

    for f in tqdm.asyncio.tqdm.as_completed(flist):
        await f

def preprocess_tweet_files(path):
    """
    Apply preprocessing to all tweets in file
    """
    texts = glob.glob(os.path.join(path, "*.txt"))

    print(f"{len(texts)} files found")

    asyncio.run(main_loop(texts))


if __name__ == "__main__":
    fire.Fire(preprocess_tweet_files)
