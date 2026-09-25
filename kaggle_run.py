import subprocess
from pathlib import Path
import os
import json

username = os.getenv("KAGGLE_USERNAME")

if not username:
    raise RuntimeError("KAGGLE_USERNAME is not set")

KERNEL = f"{username}/tp1-fake-news"

PROJECT_DIR = Path("/app")
OUTPUT_DIR = Path("output")

metadata = {
    "id": f"{username}/tp1-fake-news",
    "title": "tp1-fake-news",
    "code_file": "tp1-fake-news.ipynb",
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
print(f"Kaggle kernel: {metadata['id']}")


# --------------------------------------------------
# Push notebook and start Kaggle execution
# --------------------------------------------------

print("Pushing notebook to Kaggle...")

subprocess.run(
    ["kaggle", "kernels", "push", "-p", "."],
    check=True,
)


# --------------------------------------------------
# Follow Kaggle execution logs
# --------------------------------------------------

print("Following Kaggle notebook logs...")

subprocess.run(
    ["kaggle", "kernels", "logs", KERNEL, "--follow"],
    check=True,
)


# --------------------------------------------------
# Download Kaggle outputs
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading Kaggle outputs...")

subprocess.run(
    [
        "kaggle",
        "kernels",
        "output",
        KERNEL,
        "-p",
        str(OUTPUT_DIR),
        "--force",
    ],
    check=True,
)

print(f"Outputs downloaded to: {OUTPUT_DIR}")