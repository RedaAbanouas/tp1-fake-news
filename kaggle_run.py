import subprocess
from pathlib import Path
import os
import json

username = os.getenv("KAGGLE_USERNAME")

if not username:
    raise RuntimeError("KAGGLE_USERNAME is not set")

KERNEL = f"{username}/tp1-fake-news"

PROJECT_DIR = Path("/app")
OUTPUT_DIR = PROJECT_DIR / "output"


# --------------------------------------------------
# Generate Kaggle metadata
# --------------------------------------------------

metadata = {
    "id": KERNEL,
    "title": "tp1-fake-news",
    "code_file": "main.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [
        "clmentbisaillon/fake-and-real-news-dataset"
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}

metadata_path = PROJECT_DIR / "kernel-metadata.json"

with metadata_path.open("w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print(f"Generated: {metadata_path}")
print(f"Kernel: {KERNEL}")


# --------------------------------------------------
# Push notebook
# --------------------------------------------------

print("Pushing notebook to Kaggle...")

subprocess.run(
    [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(PROJECT_DIR),
    ],
    check=True,
)


# --------------------------------------------------
# Follow Kaggle logs
# --------------------------------------------------

print("Following Kaggle notebook logs...")

subprocess.run(
    [
        "kaggle",
        "kernels",
        "logs",
        KERNEL,
        "--follow",
    ],
    check=True,
)