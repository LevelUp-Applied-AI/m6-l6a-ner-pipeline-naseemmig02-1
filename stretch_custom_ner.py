
import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from typing import List, Dict
import os

# ---------------------------------------------------------------------------
# Task: Define EntityRuler Patterns
# ---------------------------------------------------------------------------

def get_climate_patterns():
    """Returns a list of patterns for the EntityRuler."""
    return [
        # Concept 1: COP Summits
        {"label": "CLIMATE_EVENT", "pattern": [{"LOWER": "cop"}, {"IS_DIGIT": True}]},
        {"label": "CLIMATE_EVENT", "pattern": "COP28"},
        
        # Concept 2: Paris Agreement
        {"label": "AGREEMENT", "pattern": "Paris Agreement"},
        
        # Concept 3: IPCC Reports
        {"label": "REPORT", "pattern": "Sixth Assessment Report"},
        {"label": "REPORT", "pattern": "IPCC AR6"},
        
        # Concept 4: Temperature Thresholds
        {"label": "THRESHOLD", "pattern": [{"TEXT": {"REGEX": "^[12](\\.[05])?°?C$"}}, {"LOWER": "target"}]},
        {"label": "THRESHOLD", "pattern": "1.5 degrees Celsius"},
        
        # Concept 5: Climate Policies (CBAM)
        {"label": "POLICY", "pattern": "Carbon Border Adjustment Mechanism"},
        
        # Concept 6: Climate Summits
        {"label": "CLIMATE_EVENT", "pattern": "Climate Ambition Summit"},
        
        # Concept 7: NDCs
        {"label": "POLICY", "pattern": "nationally determined contributions"},
        {"label": "POLICY", "pattern": "NDCs"},
        
        # Concept 8: FAO Report
        {"label": "REPORT", "pattern": "State of Food and Agriculture"},
    ]

# ---------------------------------------------------------------------------
# Task: Integration and Extraction
# ---------------------------------------------------------------------------

def extract_entities(df, nlp):
    """Extract entities from English texts using the provided spaCy nlp object."""
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
    return pd.DataFrame(rows)

def evaluate_standard_entities(predicted_df, gold_df):
    """
    Evaluate only standard-label entities found in gold_entities.csv.
    Standard labels: ORG, GPE, DATE, LAW, MONEY, PERSON, QUANTITY, LOC, EVENT, WORK_OF_ART
    """
    standard_labels = ["ORG", "GPE", "DATE", "LAW", "MONEY", "PERSON", "QUANTITY", "LOC", "EVENT", "WORK_OF_ART"]
    
    # Filter predicted entities to only those with standard labels
    pred_standard = predicted_df[predicted_df["entity_label"].isin(standard_labels)].copy()
    
    # Build sets for comparison
    predicted_set = set(zip(pred_standard["text_id"], pred_standard["entity_text"], pred_standard["entity_label"]))
    gold_set = set(zip(gold_df["text_id"], gold_df["entity_text"], gold_df["entity_label"]))
    
    gold_text_ids = set(gold_df["text_id"].unique())
    predicted_on_gold = {item for item in predicted_set if item[0] in gold_text_ids}
    
    tp = len(predicted_on_gold & gold_set)
    fp = len(predicted_on_gold - gold_set)
    fn = len(gold_set - predicted_on_gold)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def main():
    # Load data
    df = pd.read_csv("data/climate_articles.csv")
    gold_df = pd.read_csv("data/gold_entities.csv")
    
    # Base Model
    print("Running Base Model...")
    nlp_base = spacy.load("en_core_web_sm")
    spacy_base_df = extract_entities(df, nlp_base)
    
    # Before NER Model
    print("Running EntityRuler BEFORE NER...")
    nlp_before = spacy.load("en_core_web_sm")
    ruler_before = nlp_before.add_pipe("entity_ruler", before="ner")
    ruler_before.add_patterns(get_climate_patterns())
    spacy_before_df = extract_entities(df, nlp_before)
    
    # After NER Model
    print("Running EntityRuler AFTER NER...")
    nlp_after = spacy.load("en_core_web_sm")
    ruler_after = nlp_after.add_pipe("entity_ruler", after="ner")
    ruler_after.add_patterns(get_climate_patterns())
    spacy_after_df = extract_entities(df, nlp_after)
    
    # --- 1. Before/After Comparison (Counts) ---
    print("\n=== Entity Count Comparison ===")
    base_counts = spacy_base_df["entity_label"].value_counts()
    before_counts = spacy_before_df["entity_label"].value_counts()
    after_counts = spacy_after_df["entity_label"].value_counts()
    
    summary_df = pd.DataFrame({
        "Base": base_counts,
        "Ruler_Before": before_counts,
        "Ruler_After": after_counts
    }).fillna(0).astype(int)
    print(summary_df)
    
    # --- 2. Evaluation Delta ---
    print("\n=== Evaluation on Standard Labels ===")
    base_eval = evaluate_standard_entities(spacy_base_df, gold_df)
    before_eval = evaluate_standard_entities(spacy_before_df, gold_df)
    after_eval = evaluate_standard_entities(spacy_after_df, gold_df)
    
    eval_df = pd.DataFrame({
        "Base": base_eval,
        "Ruler_Before": before_eval,
        "Ruler_After": after_eval
    })
    print(eval_df)
    
    # --- 3. Qualitative Analysis of Custom Labels ---
    print("\n=== Custom Label Examples (Ruler Before) ===")
    custom_labels = ["CLIMATE_EVENT", "POLICY", "REPORT", "THRESHOLD", "AGREEMENT"]
    custom_matches = spacy_before_df[spacy_before_df["entity_label"].isin(custom_labels)]
    
    for label in custom_labels:
        matches = custom_matches[custom_matches["entity_label"] == label].head(2)
        if not matches.empty:
            print(f"\nLabel: {label}")
            for _, row in matches.iterrows():
                text_snippet = df[df["id"] == row["text_id"]]["text"].values[0]
                # Find context around the match
                start = max(0, row["start_char"] - 20)
                end = min(len(text_snippet), row["end_char"] + 20)
                print(f"  Match: '{row['entity_text']}' in '...{text_snippet[start:end]}...'")

    # Save results for analysis
    with open("stretch_comparison_output.txt", "w", encoding="utf-8") as f:
        f.write("=== Entity Count Comparison ===\n")
        f.write(summary_df.to_string())
        f.write("\n\n=== Evaluation on Standard Labels ===\n")
        f.write(eval_df.to_string())
        f.write("\n\n=== Custom Label Examples ===\n")
        for label in custom_labels:
            matches = custom_matches[custom_matches["entity_label"] == label].head(3)
            if not matches.empty:
                f.write(f"\nLabel: {label}\n")
                for _, row in matches.iterrows():
                    f.write(f"  Match: '{row['entity_text']}' (Text ID: {row['text_id']})\n")

if __name__ == "__main__":
    main()
