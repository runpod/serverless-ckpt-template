'''
RunPod | serverless-ckpt-template | model_fetcher.py

Downloads the model from the URL passed in.
'''

import os
import re
import shutil
import requests
import argparse

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)


SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"
MODEL_CACHE_DIR = "diffusers-cache"


def download_model(model_url: str):
    '''
    Downloads the model from the URL passed in.
    '''
    if os.path.exists(MODEL_CACHE_DIR):
        shutil.rmtree(MODEL_CACHE_DIR)
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    # Check if the URL is from huggingface.co, if so, grab the model repo id.
    if re.match(r"huggingface.co", model_url):
        url_parts = model_url.split("/")
        model_id = f"{url_parts[-2]}/{url_parts[-1]}"
    else:
        downloaded_model = requests.get(model_url, stream=True, timeout=600)
        with open(f"{MODEL_CACHE_DIR}/model.zip", "wb") as f:
            for chunk in downloaded_model.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    saftey_checker = StableDiffusionSafetyChecker.from_pretrained(
        SAFETY_MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        cache_dir=MODEL_CACHE_DIR,
    )


# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model_url", type=str,
                    default="https://huggingface.co/stabilityai/stable-diffusion-2-1", help="URL of the model to download.")


if __name__ == "__main__":
    args = parser.parse_args()
    download_model(args.model_url)
