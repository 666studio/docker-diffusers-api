import os
from utils import Storage

CHECKPOINT_URL = os.environ.get("CHECKPOINT_URL", None)

if __name__ == "__main__":
    CHECKPOINT_DIR = "/root/.cache/checkpoints"
    if not os.path.isdir(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    if CHECKPOINT_URL:
        fname = CHECKPOINT_DIR + "/" + CHECKPOINT_URL.split("/").pop()
        if not os.path.isfile(fname):
            storage = Storage(CHECKPOINT_URL)
            storage.download_file(fname)

    storage = Storage("https://huggingface.co/sd-concepts-library/one-line-drawing/blob/main/learned_embeds.bin")
    storage.download_file(CHECKPOINT_DIR + "/lineart.bin")