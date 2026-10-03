"""
evaluate_nlp.py
---------------
Evaluates FinBERT predictions against 60 hand-labeled headlines.

Outputs
-------
  - Accuracy
  - Per-class precision / recall / F1
  - Confusion matrix (text + saved to data/processed/confusion_matrices.json)
  - Honest commentary on where the model struggles

Hand-labeling methodology
--------------------------
Labels in data/raw/headlines_labeled.csv were assigned by the analyst
based on clear positive/negative financial framing in each headline.
Headlines about record growth, deal wins, profit beats → Positive.
Headlines about falling margins, lawsuits, guidance misses → Negative.
Ambiguous headlines (e.g., leadership restructuring with no valence
signal) → Neutral.

Run this after sentiment.py has generated news_sentiment.csv.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_PATH   = os.path.join(BASE_DIR, "data", "raw",       "headlines_labeled.csv")
SENT_PATH     = os.path.join(BASE_DIR, "data", "processed", "news_sentiment.csv")
OUT_PATH      = os.path.join(BASE_DIR, "data", "processed", "confusion_matrices.json")

LABEL_ORDER   = ["Positive", "Neutral", "Negative"]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _cm_to_dict(cm: np.ndarray, labels: list) -> dict:
    return {
        "labels": labels,
        "matrix": cm.tolist(),
    }


def _print_cm(cm: np.ndarray, labels: list, title: str):
    print(f"\n{'─'*50}")
    print(f"  Confusion Matrix — {title}")
    print(f"{'─'*50}")
    header = f"{'':>12}" + "".join(f"{l:>12}" for l in labels)
    print(f"  {header}")
    for i, row_label in enumerate(labels):
        row = f"  {row_label:>12}" + "".join(f"{cm[i, j]:>12}" for j in range(len(labels)))
        print(row)
    print(f"  (rows = true, columns = predicted)")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation() -> dict:
    # ---- load hand labels ----
    labels_df = pd.read_csv(LABELS_PATH)
    labels_df.columns = labels_df.columns.str.strip()

    print(f"Loaded {len(labels_df)} hand-labeled headlines.")
    print(f"Label distribution:\n{labels_df['true_label'].value_counts().to_string()}\n")

    results = {}

    # ---- pull pre-computed FinBERT labels from processed CSV ----
    if not os.path.exists(SENT_PATH):
        print("news_sentiment.csv not found — run sentiment.py first.")
        return results

    sent_df = pd.read_csv(SENT_PATH)
    merged  = pd.merge(
        labels_df,
        sent_df[["Headline", "fb_label"]],
        on="Headline",
        how="left",
    )

    finbert_preds = merged["fb_label"]
    if not finbert_preds.notna().all():
        print("Some headlines could not be matched to the sentiment CSV. Check news_sentiment.csv.")
        return results

    true_labels = labels_df["true_label"]

    # ----------------------------------------------------------------
    # FinBERT evaluation
    # ----------------------------------------------------------------
    print("=" * 50)
    print("  FinBERT EVALUATION")
    print("=" * 50)

    fb_acc = accuracy_score(true_labels, finbert_preds)
    print(f"\n  Accuracy : {fb_acc:.1%}")
    print("\n  Classification Report:")
    print(classification_report(true_labels, finbert_preds, labels=LABEL_ORDER, zero_division=0))

    cm_fb = confusion_matrix(true_labels, finbert_preds, labels=LABEL_ORDER)
    _print_cm(cm_fb, LABEL_ORDER, "FinBERT")

    results["finbert"] = {
        "accuracy": round(fb_acc, 4),
        "report":   classification_report(
                        true_labels, finbert_preds,
                        labels=LABEL_ORDER, output_dict=True, zero_division=0
                    ),
        "confusion_matrix": _cm_to_dict(cm_fb, LABEL_ORDER),
    }

    # ---- honest commentary ----
    print("\n" + "=" * 50)
    print("  NOTES ON RESULTS")
    print("=" * 50)
    print("""
  These 60 headlines were written for this project and are relatively
  unambiguous in their financial framing, so accuracy here will likely
  overestimate real-world performance on messier news data.

  FinBERT was fine-tuned on financial text (Financial PhraseBank) and
  handles domain-specific phrasing well, but may still confuse neutral
  corporate announcements with positive ones.

  The model has not been calibrated on live NSE news — treat accuracy
  figures as indicative, not production-grade.
""")

    # ---- save ----
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Confusion matrices saved → {OUT_PATH}")

    return results


if __name__ == "__main__":
    run_evaluation()
