import os
import json
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    confusion_matrix,
)


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

MODEL_NAME = "gemini-3.5-flash-lite"


# --------------------------------------------------
# 2. Zero-shot prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a binary news classification system.

Your task is to classify a news article as either:

- 0 = Fake
- 1 = Real

Definitions:

- Fake: fabricated, false, or misleading news.
- Real: genuine news reporting.

Classify the article using only the information provided.

The label must be exactly:
0 for Fake
1 for Real

Do not provide an explanation.
"""


# --------------------------------------------------
# 3. Classify a batch of 8 articles
# --------------------------------------------------

def classify_batch(articles, max_retries=5):
    """
    Sends one batch to Gemini.

    If Gemini temporarily returns HTTP 503, retry the same
    request with exponential backoff.

    The function ultimately returns integer predictions:
    0 = Fake
    1 = Real.
    """

    articles_for_prompt = []

    for i, article in enumerate(articles):
        articles_for_prompt.append(
            f"ARTICLE {i}:\n{article}"
        )

    user_prompt = """
Classify each of the following news articles.

Return exactly one prediction for each article:

0 = Fake
1 = Real

Do not explain your decisions.

Articles:

""" + "\n\n".join(articles_for_prompt)

    for attempt in range(max_retries + 1):

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "predictions": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "id": {
                                            "type": "INTEGER"
                                        },
                                        "label": {
                                            "type": "STRING",
                                            "enum": ["0", "1"]
                                        }
                                    },
                                    "required": ["id", "label"]
                                }
                            }
                        },
                        "required": ["predictions"]
                    }
                )
            )

            result = json.loads(response.text)

            sorted_predictions = sorted(
                result["predictions"],
                key=lambda x: x["id"]
            )

            predictions = [
                int(item["label"])
                for item in sorted_predictions
            ]

            return predictions

        except errors.ServerError as e:

            if e.code != 503 or attempt == max_retries:
                raise

            wait_time = 5 * (2 ** attempt)

            print(
                f"Gemini returned 503. "
                f"Retry {attempt + 1}/{max_retries} "
                f"in {wait_time}s..."
            )

            time.sleep(wait_time)

test_df = pd.read_csv("./output/test.csv")

fake_test = test_df[test_df["label"] == 0].sample(
    n=64,
    random_state=42
)

real_test = test_df[test_df["label"] == 1].sample(
    n=64,
    random_state=42
)

# Combine Fake and Real examples
test_balanced = pd.concat(
    [fake_test, real_test],
    ignore_index=True
)

# Shuffle the final test set
test_balanced = test_balanced.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X_test = test_balanced["content"]
y_test = test_balanced["label"]


BATCH_SIZE = 32
NUM_BATCHES = 4

NUM_ARTICLES = BATCH_SIZE * NUM_BATCHES

X_test_subset = X_test.iloc[:NUM_ARTICLES]
y_test_subset = y_test.iloc[:NUM_ARTICLES]

predictions = []

start_time = time.perf_counter()

for batch_number in range(NUM_BATCHES):

    start = batch_number * BATCH_SIZE
    end = start + BATCH_SIZE

    batch = X_test_subset.iloc[start:end]

    print(
        f"Batch {batch_number + 1}/{NUM_BATCHES} "
        f"({start + 1}-{end})"
    )

    batch_predictions = classify_batch(
        batch.tolist()
    )

    predictions.extend(batch_predictions)

    print(
        f"Predictions: {batch_predictions}"
    )

    # Wait between requests
    time.sleep(4)


inference_time = time.perf_counter() - start_time

accuracy = accuracy_score(y_test_subset, predictions)
f1 = f1_score(y_test_subset, predictions)
cm = confusion_matrix(y_test_subset, predictions,)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    
disp.plot(cmap="Greens")

plt.title("LLM Model Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "./output/llm_confusion_matrix.png", dpi=200)

plt.close()

print(f"LLM Accuracy ({MODEL_NAME}): {accuracy:.4f}")
print(f"LLM F1 ({MODEL_NAME}):       {f1:.4f}")

llm_metrics = {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "inference_time_seconds": round(inference_time, 4),
        "model_name": MODEL_NAME
    }
    
with open("./output/llm_metrics.json", "w") as f:
    json.dump(llm_metrics, f, indent=4)