"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import unicodedata

import pandas as pd
import spacy
from transformers import pipeline as hf_pipeline


# ---------------------------------------------------------------------------
# Task 1: Load and Explore the Data
# ---------------------------------------------------------------------------

def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    return pd.read_csv(filepath)


def explore_data(df):
    """Summarize basic corpus statistics.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        Dictionary with keys:
          'shape': tuple (n_rows, n_cols)
          'lang_counts': dict mapping language code -> row count
          'category_counts': dict mapping category -> row count
          'text_length_stats': dict with 'mean', 'min', 'max' word counts
    """
    word_counts = df["text"].str.split().str.len()

    return {
        "shape": df.shape,
        "lang_counts": df["language"].value_counts().to_dict(),
        "category_counts": df["category"].value_counts().to_dict(),
        "text_length_stats": {
            "mean": round(word_counts.mean(), 2),
            "min": int(word_counts.min()),
            "max": int(word_counts.max()),
        },
    }


# ---------------------------------------------------------------------------
# Task 2: Preprocess Text
# ---------------------------------------------------------------------------

def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using the injected spaCy pipeline.

    Args:
        text: Raw text string.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        List of cleaned, lemmatized token strings.
    """
    # Step 1: Unicode NFC normalization (handles accented chars, ligatures, etc.)
    normalized = unicodedata.normalize("NFC", text)

    # Step 2: Run through spaCy
    doc = nlp(normalized)

    # Step 3: Drop punctuation and whitespace tokens, return lowercased lemmas
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_punct and not token.is_space
    ]

    return tokens


# ---------------------------------------------------------------------------
# Task 3: spaCy NER Extraction
# ---------------------------------------------------------------------------

def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER.

    Args:
        df: DataFrame with columns id, text, language, ...
        nlp: A loaded spaCy Language object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # Filter to English only — en_core_web_sm does not support Arabic
    english_df = df[df["language"] == "en"]

    rows = []
    for _, row in english_df.iterrows():
        doc = nlp(row["text"])
        for ent in doc.ents:
            rows.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })

    return pd.DataFrame(rows, columns=["text_id", "entity_text", "entity_label",
                                        "start_char", "end_char"])


# ---------------------------------------------------------------------------
# Task 4: Hugging Face NER Extraction
# ---------------------------------------------------------------------------

def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER.

    Uses the injected HF pipeline (expected: dslim/bert-base-NER).

    Args:
        df: DataFrame with columns id, text, language, ...
        ner_pipeline: A loaded Hugging Face `pipeline('ner', ...)` object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # Filter to English only
    english_df = df[df["language"] == "en"]

    rows = []
    for _, row in english_df.iterrows():
        raw_entities = ner_pipeline(row["text"])

        # --- Merge WordPiece subword tokens ---
        # BERT splits "IPCC" -> ["IP", "##CC"]; we stitch them back together.
        merged = []
        for token in raw_entities:
            if token["word"].startswith("##") and merged:
                # Continuation token: append to previous entity's text
                merged[-1]["word"] += token["word"][2:]
                merged[-1]["end"] = token["end"]
            else:
                merged.append(dict(token))  # copy so we can mutate safely

        # --- Strip IOB prefix and record entities ---
        for token in merged:
            # Strip "B-" or "I-" prefix (e.g. "B-ORG" -> "ORG")
            label = token["entity"]
            if label.startswith("B-") or label.startswith("I-"):
                label = label[2:]

            rows.append({
                "text_id": row["id"],
                "entity_text": token["word"],
                "entity_label": label,
                "start_char": token["start"],
                "end_char": token["end"],
            })

    return pd.DataFrame(rows, columns=["text_id", "entity_text", "entity_label",
                                        "start_char", "end_char"])


# ---------------------------------------------------------------------------
# Task 5: Compare NER Outputs
# ---------------------------------------------------------------------------

def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label -> count for spaCy
          'hf_counts': dict of entity_label -> count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
          'both': set of (text_id, entity_text) tuples found by both systems
          'spacy_only': set of (text_id, entity_text) tuples found only by spaCy
          'hf_only': set of (text_id, entity_text) tuples found only by HF
    """
    # Per-label counts
    spacy_counts = spacy_df["entity_label"].value_counts().to_dict()
    hf_counts = hf_df["entity_label"].value_counts().to_dict()

    # Build sets of (text_id, entity_text) tuples for overlap analysis
    spacy_set = set(zip(spacy_df["text_id"], spacy_df["entity_text"]))
    hf_set = set(zip(hf_df["text_id"], hf_df["entity_text"]))

    both = spacy_set & hf_set
    spacy_only = spacy_set - hf_set
    hf_only = hf_set - spacy_set

    # Print a human-readable summary
    print("\n--- NER Comparison Summary ---")
    print(f"  Total spaCy entities : {len(spacy_df)}")
    print(f"  Total HF entities    : {len(hf_df)}")
    print(f"  Agreed on            : {len(both)}")
    print(f"  spaCy-only           : {len(spacy_only)}")
    print(f"  HF-only              : {len(hf_only)}")
    print(f"\n  spaCy label counts   : {spacy_counts}")
    print(f"  HF label counts      : {hf_counts}")

    return {
        "spacy_counts": spacy_counts,
        "hf_counts": hf_counts,
        "total_spacy": len(spacy_df),
        "total_hf": len(hf_df),
        "both": both,
        "spacy_only": spacy_only,
        "hf_only": hf_only,
    }


# ---------------------------------------------------------------------------
# Task 6: Evaluate Against Gold Standard
# ---------------------------------------------------------------------------

def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations.

    Computes entity-level precision, recall, and F1. An entity is a
    true positive if both the entity text and label match a gold entry
    for the same text_id.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label.
        gold_df: DataFrame with columns text_id, entity_text,
                 entity_label.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0-1).
    """
    # Build sets of (text_id, entity_text, entity_label) tuples
    predicted_set = set(
        zip(predicted_df["text_id"], predicted_df["entity_text"], predicted_df["entity_label"])
    )
    gold_set = set(
        zip(gold_df["text_id"], gold_df["entity_text"], gold_df["entity_label"])
    )

    # Only evaluate on texts that have gold annotations
    gold_text_ids = set(gold_df["text_id"].unique())
    predicted_on_gold = {
        (tid, text, label)
        for tid, text, label in predicted_set
        if tid in gold_text_ids
    }

    true_positives = len(predicted_on_gold & gold_set)
    false_positives = len(predicted_on_gold - gold_set)
    false_negatives = len(gold_set - predicted_on_gold)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load models once — expensive to initialize, reuse everywhere
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading Hugging Face NER model...")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    # ── Task 1: Load and explore ──────────────────────────────────────────
    print("\n=== Task 1: Load & Explore ===")
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape              : {summary['shape']}")
            print(f"Languages          : {summary['lang_counts']}")
            print(f"Categories         : {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")

    # ── Task 2: Preprocess a sample ───────────────────────────────────────
    print("\n=== Task 2: Preprocess Text ===")
    sample_row = df[df["language"] == "en"].iloc[0]
    sample_tokens = preprocess_text(sample_row["text"], nlp)
    if sample_tokens is not None:
        print(f"Original  : {sample_row['text'][:80]}...")
        print(f"Tokens    : {sample_tokens[:10]}")

    # ── Task 3: spaCy NER ─────────────────────────────────────────────────
    print("\n=== Task 3: spaCy NER ===")
    spacy_entities = extract_spacy_entities(df, nlp)
    if spacy_entities is not None:
        print(f"spaCy entities found: {len(spacy_entities)}")
        print(spacy_entities.head())

    # ── Task 4: HF NER ───────────────────────────────────────────────────
    print("\n=== Task 4: Hugging Face NER ===")
    hf_entities = extract_hf_entities(df, hf_ner)
    if hf_entities is not None:
        print(f"HF entities found: {len(hf_entities)}")
        print(hf_entities.head())

    # ── Task 5: Compare ──────────────────────────────────────────────────
    print("\n=== Task 5: Compare NER Outputs ===")
    if spacy_entities is not None and hf_entities is not None:
        comparison = compare_ner_outputs(spacy_entities, hf_entities)

    # ── Task 6: Evaluate against gold standard ───────────────────────────
    print("\n=== Task 6: Evaluate Against Gold Standard ===")
    gold = pd.read_csv("data/gold_entities.csv")

    if spacy_entities is not None:
        spacy_metrics = evaluate_ner(spacy_entities, gold)
        print(f"spaCy  → {spacy_metrics}")

    if hf_entities is not None:
        hf_metrics = evaluate_ner(hf_entities, gold)
        print(f"HF     → {hf_metrics}")