import os
from utils import Storage
import torch

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

    print("download checkpoint", os.stat("/root/.cache/checkpoints/lineart.bin").st_size, torch.__version__)
    loaded_learned_embeds = torch.load("/root/.cache/checkpoints/lineart.bin", map_location="cpu")
    print("download checkpoint success", loaded_learned_embeds)
