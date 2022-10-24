# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
import os
import json
from io import BytesIO
from PIL import Image
from pathlib import Path

TESTS = "tests"
FIXTURES = TESTS + os.sep + "fixtures"
OUTPUT = TESTS + os.sep + "output"
Path(OUTPUT).mkdir(parents=True, exist_ok=True)


def b64encode_file(filename: str):
    with open(os.path.join(FIXTURES, filename), "rb") as file:
        return base64.b64encode(file.read())


def output_path(filename: str):
    return os.path.join(OUTPUT, filename)


def decode_and_save(image_byte_string: str, name: str):
    image_encoded = image_byte_string.encode("utf-8")
    image_bytes = BytesIO(base64.b64decode(image_encoded))
    image = Image.open(image_bytes)
    fp = output_path(name + ".png")
    image.save(fp)
    print("Saved " + fp)


def test(name, inputs):
    print("Running test: " + name)
    response = requests.post("http://localhost:8000/", json=inputs)
    result = response.json()

    if result.get("images_base64", None) == "None":
        print(json.dumps(result, indent=4))
        print()
        return

    images_base64 = result.get("images_base64", None)
    if images_base64:
        for idx, image_byte_string in enumerate(images_base64):
            decode_and_save(image_byte_string, f"{name}_{idx}")
    else:
        decode_and_save(result["image_base64"], name)

    print()


test(
    "outpaint",
    {
        "modelInputs": {
            "prompt": "girl with a pearl earing standing in a big room",
            "init_image": b64encode_file("girl_with_pearl_earing_outpainting_in.png"),
        },
        "callInputs": {
            "MODEL_ID": "CompVis/stable-diffusion-v1-4",
            "PIPELINE": "StableDiffusionInpaintPipelineLegacy",
            "SCHEDULER": "DDIM",  # Note, as of diffusers 0.3.0, no LMS yet
            "FILL_MODE": "patchmatch",
        },
    },
)