# TP1 — Fake News Classification

Comparing three approaches to binary fake/real news classification on the
[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset):

- **Part A** — TF-IDF + Logistic Regression (classical baseline)
- **Part B** — Frozen DistilBERT embeddings + Logistic Regression
- **Part C** — Zero-shot classification with a Gemini LLM (`llm_prompt.py`)

The heavy training (Parts A & B) runs as a Kaggle notebook (`main.ipynb`) so it
gets free GPU access; the LLM part runs locally against the test split that
notebook produces.

## Project layout

```
.
├── Dockerfile          # Reproducible dev environment (uv + Python 3.12)
├── main.ipynb          # Preprocessing, TF-IDF+LogReg, DistilBERT+LogReg
├── kaggle_run.py        # Pushes main.ipynb to Kaggle and streams its logs
├── llm_prompt.py        # Gemini zero-shot classification + metrics/plot
├── pyproject.toml / uv.lock
└── .env.example
```

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- A [Kaggle](https://www.kaggle.com/) account with an API token (needed to
  push and run `main.ipynb` on Kaggle's infrastructure)
- A [Google Gemini API key](https://aistudio.google.com/apikey) (needed for
  the LLM classification step)

## 1. Configure credentials

Create a new `.env` file in the project root, using `.env.example` as the
template:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_API_KEY=your_kaggle_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## 2. Build and start the container

```bash
docker build -t tp1-fake-news .
docker run --rm -it --env-file .env --name fake-news-container tp1-fake-news
```

This drops you into a shell inside `/app`, with dependencies already
installed via `uv sync` at build time.

## 3. Run the notebook on Kaggle (Parts A & B)

From inside the container:

```bash
uv run python kaggle_run.py
```

This script:

1. Generates a `kernel-metadata.json` pointing at the
   `clmentbisaillon/fake-and-real-news-dataset` Kaggle dataset, with GPU and
   internet access enabled.
2. Pushes `main.ipynb` to Kaggle as a private kernel
   (`<KAGGLE_USERNAME>/tp1-fake-news`).
3. Follows the kernel's logs until it finishes.

`main.ipynb` itself:

- Loads `Fake.csv` / `True.csv`, labels them (`0` = fake, `1` = real),
  merges the title and body into a single `content` column, and drops
  duplicates.
- Splits the data 80/20 (stratified, `random_state=42`) and writes the test
  split to `/kaggle/working/test.csv`.
- **Part A**: fits a `TfidfVectorizer` (unigrams+bigrams, 768 features) with
  a `LogisticRegression` classifier.
- **Part B**: extracts 768-dim `distilbert-base-uncased` embeddings for
  train/test and fits a `LogisticRegression` on top.

Once the Kaggle run finishes, pull its output (including `test.csv`) locally:

```bash
kaggle kernels output $KAGGLE_USERNAME/tp1-fake-news -p ./output --force
```

## 4. Run the LLM baseline (Part C)

Still inside the container, with `./output/test.csv` present:

```bash
uv run python llm_prompt.py
```

This samples a balanced 128-article test subset, sends it to Gemini
(`gemini-3.5-flash-lite`) in 4 batches of 32 for zero-shot fake/real
classification (with retry/backoff on transient `503`s), and writes:

- `./output/llm_confusion_matrix.png`
- `./output/llm_metrics.json` (accuracy, F1, inference time, model name)

## 5. Retrieve results from the container

If you ran step 4 without a mounted volume, copy the results out to the host
(from a separate terminal, outside the container, in the project directory):

```bash
docker cp fake-news-container:/app/output .
```

## Notes

- The three approaches are meant to be compared on accuracy/F1 and training
  or inference time — see each part's printed metrics / `llm_metrics.json`.
- `main.ipynb` can also be run directly in a Kaggle notebook environment
  without `kaggle_run.py`, as long as the dataset is attached and
  `RUN_BASELINE` / `RUN_PLM` flags at the top are set as desired.
