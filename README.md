curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12
uv python pin 3.12

uv init && uv venv
source .venv/bin/activate

uv add kaggle pandas groq scikit-learn matplotlib python-dotenv