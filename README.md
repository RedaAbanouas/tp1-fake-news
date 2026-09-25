docker build -t tp1-fake-news .
docker run --rm -it --env-file .env --name fake-news-container tp1-fake-news

uv run python kaggle_run.py
kaggle kernels output $KAGGLE_USERNAME/tp1-fake-news -p ./output --force

uv run python llm_prompt.py

(outside the container in the project directory)
docker cp fake-news-container:/app/output .