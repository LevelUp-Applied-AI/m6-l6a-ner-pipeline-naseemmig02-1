"""
Challenge Extensions — Tiers 1, 2, and 3

Tier 1: Per-Category NER Analysis
Tier 2: Custom Entity Aggregation Pipeline
Tier 3: Custom NER Evaluator (with pytest tests in test_challenges.py)

Run: python challenges.py
"""

import warnings
warnings.filterwarnings("ignore")

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import spacy

from ner_pipeline import load_data, extract_spacy_entities


# ═════════════════════════════════════════════════════════════════════════════
# TIER 1 — Per-Category NER Analysis
# ═════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "policy":     "#2E86AB",
    "science":    "#A23B72",
    "impact":     "#F18F01",
    "adaptation": "#C73E1D",
}
CATEGORIES = ["policy", "science", "impact", "adaptation"]


def entity_counts_by_category(entities_df: pd.DataFrame,
                               articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a pivot table: rows = entity_label, cols = category, values = count.

    Args:
        entities_df: Output of extract_spacy_entities (text_id, entity_label, ...).
        articles_df: Original climate_articles DataFrame (id, category, ...).

    Returns:
        DataFrame with entity labels as index and categories as columns.
    """
    merged = entities_df.merge(
        articles_df[["id", "category"]],
        left_on="text_id", right_on="id", how="left",
    )
    pivot = (
        merged.groupby(["entity_label", "category"])
        .size()
        .unstack(fill_value=0)
    )
    for cat in CATEGORIES:
        if cat not in pivot.columns:
            pivot[cat] = 0
    pivot = pivot[CATEGORIES]
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=False).drop(columns="_total")
    return pivot


def evaluate_per_category(entities_df: pd.DataFrame,
                           gold_df: pd.DataFrame,
                           articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute entity-level precision, recall, and F1 for each category
    that appears in the gold standard.

    Args:
        entities_df: Predicted entities DataFrame.
        gold_df:     Gold-standard entities DataFrame.
        articles_df: Original articles DataFrame (for category lookup).

    Returns:
        DataFrame with columns: category, precision, recall, f1, n_gold.
    """
    id_to_cat   = articles_df.set_index("id")["category"].to_dict()
    gold_ids    = set(gold_df["text_id"].unique())
    results     = []

    for category in CATEGORIES:
        cat_ids = {tid for tid in gold_ids if id_to_cat.get(tid) == category}
        if not cat_ids:
            continue

        pred_set = set(zip(
            entities_df.loc[entities_df["text_id"].isin(cat_ids), "text_id"],
            entities_df.loc[entities_df["text_id"].isin(cat_ids), "entity_text"],
            entities_df.loc[entities_df["text_id"].isin(cat_ids), "entity_label"],
        ))
        gold_set = set(zip(
            gold_df.loc[gold_df["text_id"].isin(cat_ids), "text_id"],
            gold_df.loc[gold_df["text_id"].isin(cat_ids), "entity_text"],
            gold_df.loc[gold_df["text_id"].isin(cat_ids), "entity_label"],
        ))

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = _f1(precision, recall)

        results.append({
            "category":  category,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "n_gold":    len(gold_set),
        })

    return pd.DataFrame(results)


def plot_heatmap(pivot: pd.DataFrame, save_path: str = "tier1_heatmap.png"):
    """Heatmap: entity label × category, colour = count (top-15 labels)."""
    top_labels = pivot.nlargest(15, pivot.columns.tolist()).index
    data = pivot.loc[top_labels]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        data, annot=True, fmt="d", cmap="YlOrRd",
        linewidths=0.4, linecolor="#e0e0e0",
        cbar_kws={"label": "Entity count"}, ax=ax,
    )
    ax.set_title("Entity-Type × Category Distribution (spaCy, top-15 labels)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Category", fontsize=11)
    ax.set_ylabel("Entity Label", fontsize=11)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_grouped_bar(pivot: pd.DataFrame,
                     save_path: str = "tier1_grouped_bar.png"):
    """Grouped bar chart: top-8 entity labels coloured by category."""
    top_labels = pivot.nlargest(8, pivot.columns.tolist()).index
    data   = pivot.loc[top_labels]
    x      = np.arange(len(top_labels))
    n_cats = len(CATEGORIES)
    width  = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, cat in enumerate(CATEGORIES):
        offset = (i - n_cats / 2 + 0.5) * width
        ax.bar(x + offset, data[cat], width, label=cat.capitalize(),
               color=PALETTE[cat], edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(top_labels, fontsize=10)
    ax.set_ylabel("Entity count", fontsize=11)
    ax.set_title("Top-8 Entity Labels by Category (spaCy)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Category", framealpha=0.9)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_per_category_metrics(metrics_df: pd.DataFrame,
                               save_path: str = "tier1_metrics.png"):
    """Bar chart of precision / recall / F1 per category."""
    if metrics_df.empty:
        print("  No per-category metrics to plot (insufficient gold data).")
        return

    categories = metrics_df["category"].tolist()
    x     = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, metrics_df["precision"], width, label="Precision",
           color="#2E86AB", edgecolor="white")
    ax.bar(x,          metrics_df["recall"],   width, label="Recall",
           color="#A23B72", edgecolor="white")
    ax.bar(x + width,  metrics_df["f1"],       width, label="F1",
           color="#F18F01", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("NER Quality per Category (spaCy vs Gold Standard)",
                 fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


TIER1_ANALYSIS = """
Tier 1 Analysis
───────────────
Policy texts are dominated by ORG and GPE entities (international bodies,
countries, agreements) — the types spaCy recognises most reliably — likely
giving that category the highest F1. Science texts contain more DATE and
CARDINAL entities (measurement values, report years), but numerics are often
mis-spanned or partially extracted, hurting recall. Impact texts combine
location references with quantity expressions (sea-level figures, temperature
anomalies), leading to both boundary errors on quantities and label confusion
between QUANTITY and CARDINAL. Adaptation texts show the most varied entity
mix and the smallest gold-annotated sample, so their metrics should be treated
with caution. Overall, categories dominated by clearly delimited organisation
and country names (policy) are easiest for spaCy's en_core_web_sm; those
dominated by numeric and domain-specific expressions (science, impact) are
hardest.
"""


# ═════════════════════════════════════════════════════════════════════════════
# TIER 2 — Custom Entity Aggregation Pipeline
# ═════════════════════════════════════════════════════════════════════════════

NORMALISATION_MAP: dict[str, str] = {
    "united nations":                            "United Nations",
    "un":                                        "United Nations",
    "u.n.":                                      "United Nations",
    "u.n":                                       "United Nations",
    "ipcc":                                      "IPCC",
    "intergovernmental panel on climate change": "IPCC",
    "unep":                                      "UNEP",
    "un environment programme":                  "UNEP",
    "unfccc":                                    "UNFCCC",
    "united nations framework convention on climate change": "UNFCCC",
    "cop28":                                     "COP28",
    "cop 28":                                    "COP28",
    "cop27":                                     "COP27",
    "cop 27":                                    "COP27",
    "world bank":                                "World Bank",
    "the world bank":                            "World Bank",
    "wmo":                                       "WMO",
    "world meteorological organization":         "WMO",
    "us":                                        "United States",
    "u.s.":                                      "United States",
    "united states":                             "United States",
    "united states of america":                  "United States",
    "usa":                                       "United States",
    "eu":                                        "European Union",
    "european union":                            "European Union",
    "jordan":                                    "Jordan",
    "the hashemite kingdom of jordan":           "Jordan",
    "dead sea":                                  "Dead Sea",
    "arctic":                                    "Arctic",
    "the arctic":                                "Arctic",
    "paris agreement":                           "Paris Agreement",
    "the paris agreement":                       "Paris Agreement",
    "kyoto protocol":                            "Kyoto Protocol",
}


def normalise_entity(text: str) -> str:
    """
    Return the canonical form of an entity string.

    Args:
        text: Raw entity string.

    Returns:
        Canonical entity string from NORMALISATION_MAP, or the original.
    """
    return NORMALISATION_MAP.get(text.strip().lower(), text.strip())


def normalise_entities_df(entities_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply normalise_entity to every row of an entities DataFrame.

    Args:
        entities_df: DataFrame with an 'entity_text' column.

    Returns:
        Copy of the DataFrame with entity_text replaced by canonical forms.
    """
    df = entities_df.copy()
    df["entity_text"] = df["entity_text"].apply(normalise_entity)
    return df


def compute_cooccurrence(entities_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many texts each pair of (normalised) entities appears in together.

    Args:
        entities_df: DataFrame with columns text_id and entity_text.

    Returns:
        Edge-list DataFrame: entity_a, entity_b, count — sorted descending.
    """
    text_entities = (
        entities_df.groupby("text_id")["entity_text"]
        .apply(lambda s: list(s.unique()))
        .to_dict()
    )
    cooc: Counter = Counter()
    for _, ents in text_entities.items():
        for a, b in itertools.combinations(sorted(ents), 2):
            cooc[(a, b)] += 1

    return pd.DataFrame(
        [(a, b, cnt) for (a, b), cnt in cooc.items()],
        columns=["entity_a", "entity_b", "count"],
    ).sort_values("count", ascending=False).reset_index(drop=True)


def cooccurrence_matrix(edge_list: pd.DataFrame) -> pd.DataFrame:
    """
    Convert an edge list to a symmetric adjacency matrix.

    Args:
        edge_list: DataFrame with columns entity_a, entity_b, count.

    Returns:
        Square DataFrame (entity × entity) with co-occurrence counts.
    """
    entities = sorted(set(edge_list["entity_a"]) | set(edge_list["entity_b"]))
    matrix = pd.DataFrame(0, index=entities, columns=entities)
    for _, row in edge_list.iterrows():
        matrix.loc[row["entity_a"], row["entity_b"]] = row["count"]
        matrix.loc[row["entity_b"], row["entity_a"]] = row["count"]
    return matrix


def compute_entity_tfidf(entities_df: pd.DataFrame,
                          articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score each (entity, category) pair with a TF-IDF-style metric.

    TF  = count in category / total entities in category
    IDF = log( n_categories / n_categories_containing_entity )

    Args:
        entities_df: Normalised entities DataFrame.
        articles_df: Original articles DataFrame (id, category).

    Returns:
        DataFrame with columns: entity, category, tf, idf, tfidf.
    """
    merged = entities_df.merge(
        articles_df[["id", "category"]],
        left_on="text_id", right_on="id", how="left",
    )
    n_categories = merged["category"].nunique()

    tf_counts = (
        merged.groupby(["category", "entity_text"])
        .size()
        .reset_index(name="count")
    )
    cat_totals = merged.groupby("category").size().rename("cat_total")
    tf_counts  = tf_counts.join(cat_totals, on="category")
    tf_counts["tf"] = tf_counts["count"] / tf_counts["cat_total"]

    cat_presence = (
        merged.groupby("entity_text")["category"]
        .nunique()
        .rename("n_cats_present")
    )
    tf_counts = tf_counts.join(cat_presence, on="entity_text")
    tf_counts["idf"]   = tf_counts["n_cats_present"].apply(
        lambda n: math.log(n_categories / n) if n > 0 else 0.0
    )
    tf_counts["tfidf"] = tf_counts["tf"] * tf_counts["idf"]

    return (
        tf_counts.rename(columns={"entity_text": "entity"})
        [["entity", "category", "tf", "idf", "tfidf"]]
        .sort_values("tfidf", ascending=False)
        .reset_index(drop=True)
    )


def plot_cooccurrence_network(edge_list: pd.DataFrame,
                               top_n: int = 20,
                               save_path: str = "tier2_network.png"):
    """
    Draw a weighted co-occurrence network for the top-N entity pairs.

    Args:
        edge_list: Output of compute_cooccurrence.
        top_n:     Number of highest-count edges to include.
        save_path: Output filename.
    """
    top_edges = edge_list.head(top_n)
    G = nx.Graph()
    for _, row in top_edges.iterrows():
        G.add_edge(row["entity_a"], row["entity_b"], weight=row["count"])

    node_strength = defaultdict(float)
    for u, v, d in G.edges(data=True):
        node_strength[u] += d["weight"]
        node_strength[v] += d["weight"]
    node_sizes = [300 + node_strength[n] * 80 for n in G.nodes()]

    ORG_ENTITIES = {"IPCC", "UNEP", "UNFCCC", "WMO", "World Bank",
                    "United Nations", "COP28", "COP27"}
    LOC_ENTITIES = {"Jordan", "Dead Sea", "Arctic",
                    "United States", "European Union"}

    colours = [
        "#2E86AB" if n in ORG_ENTITIES else
        "#C73E1D" if n in LOC_ENTITIES else
        "#F18F01"
        for n in G.nodes()
    ]
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w   = max(weights) if weights else 1

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#fafafa")
    pos = nx.spring_layout(G, seed=42, k=2.2)

    nx.draw_networkx_edges(
        G, pos, width=[1 + 5 * (w / max_w) for w in weights],
        edge_color="#cccccc", alpha=0.7, ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color=colours, alpha=0.9, ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    legend_handles = [
        mpatches.Patch(color="#2E86AB", label="Organisation"),
        mpatches.Patch(color="#C73E1D", label="Location"),
        mpatches.Patch(color="#F18F01", label="Other"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", framealpha=0.9)
    ax.set_title(f"Top-{top_n} Entity Co-occurrences (normalised spaCy entities)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# TIER 3 — Custom NER Evaluator
# ═════════════════════════════════════════════════════════════════════════════

MatchStrategy = Literal["exact", "partial", "type_agnostic"]


@dataclass
class Entity:
    """A single named entity with its span and label."""
    text_id:    int
    text:       str
    label:      str
    start_char: int
    end_char:   int


@dataclass
class EvalResult:
    """Stores evaluation metrics for one strategy / averaging combination."""
    strategy:  str
    averaging: str
    precision: float
    recall:    float
    f1:        float
    tp:        int = 0
    fp:        int = 0
    fn:        int = 0


@dataclass
class ErrorReport:
    """Counts of each error category across all evaluated texts."""
    boundary_errors: int = 0
    type_errors:     int = 0
    missing:         int = 0
    spurious:        int = 0


def df_to_entities(df: pd.DataFrame) -> list[Entity]:
    """
    Convert an entities DataFrame to a list of Entity objects.

    Args:
        df: DataFrame with columns text_id, entity_text, entity_label,
            start_char, end_char.

    Returns:
        List of Entity instances.
    """
    return [
        Entity(
            text_id    = int(row["text_id"]),
            text       = str(row["entity_text"]),
            label      = str(row["entity_label"]),
            start_char = int(row.get("start_char", -1)),
            end_char   = int(row.get("end_char",   -1)),
        )
        for _, row in df.iterrows()
    ]


def _spans_overlap(p: Entity, g: Entity) -> bool:
    return p.start_char < g.end_char and p.end_char > g.start_char


def _exact_match(p: Entity, g: Entity) -> bool:
    """Text and label must both match exactly."""
    return p.text == g.text and p.label == g.label


def _partial_match(p: Entity, g: Entity) -> float:
    """
    Jaccard overlap ratio × label agreement.
    Returns a score in [0, 1]; label mismatch → 0.
    """
    if p.label != g.label:
        return 0.0
    if p.start_char < 0 or g.start_char < 0:
        longer = max(len(p.text), len(g.text))
        common = sum(a == b for a, b in zip(p.text, g.text))
        return common / longer if longer > 0 else 0.0
    intersection = max(0, min(p.end_char, g.end_char) - max(p.start_char, g.start_char))
    union        = max(p.end_char, g.end_char) - min(p.start_char, g.start_char)
    return intersection / union if union > 0 else 0.0


def _type_agnostic_match(p: Entity, g: Entity) -> bool:
    """Span (text) must match; label is ignored."""
    return p.text == g.text


class NERSEvaluator:
    """
    NER evaluator supporting three matching strategies and two averaging methods.

    Usage:
        ev = NERSEvaluator()
        result = ev.evaluate(predicted_df, gold_df,
                             strategy="exact", averaging="micro")
    """

    def evaluate(self,
                 predicted_df: pd.DataFrame,
                 gold_df:      pd.DataFrame,
                 strategy:     MatchStrategy = "exact",
                 averaging:    Literal["micro", "macro"] = "micro") -> EvalResult:
        """
        Compute precision, recall, and F1 under the specified strategy.

        Args:
            predicted_df: Predicted entities DataFrame.
            gold_df:      Gold-standard entities DataFrame.
            strategy:     "exact", "partial", or "type_agnostic".
            averaging:    "micro" or "macro".

        Returns:
            EvalResult with precision, recall, f1, tp, fp, fn.
        """
        predicted = df_to_entities(predicted_df)
        gold      = df_to_entities(gold_df)
        gold_ids  = {e.text_id for e in gold}
        predicted = [e for e in predicted if e.text_id in gold_ids]

        if averaging == "micro":
            return self._micro(predicted, gold, strategy)
        return self._macro(predicted, gold, strategy)

    def _micro(self, predicted, gold, strategy) -> EvalResult:
        tp, fp, fn  = self._count(predicted, gold, strategy)
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return EvalResult(strategy=strategy, averaging="micro",
                          precision=round(precision, 4),
                          recall=round(recall, 4),
                          f1=round(_f1(precision, recall), 4),
                          tp=tp, fp=fp, fn=fn)

    def _macro(self, predicted, gold, strategy) -> EvalResult:
        all_ids = sorted({e.text_id for e in gold})
        precisions, recalls, f1s = [], [], []
        total_tp = total_fp = total_fn = 0
        for tid in all_ids:
            p_sub = [e for e in predicted if e.text_id == tid]
            g_sub = [e for e in gold      if e.text_id == tid]
            tp, fp, fn = self._count(p_sub, g_sub, strategy)
            total_tp += tp; total_fp += fp; total_fn += fn
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(prec); recalls.append(rec); f1s.append(_f1(prec, rec))
        precision = float(np.mean(precisions)) if precisions else 0.0
        recall    = float(np.mean(recalls))    if recalls    else 0.0
        f1        = float(np.mean(f1s))        if f1s        else 0.0
        return EvalResult(strategy=strategy, averaging="macro",
                          precision=round(precision, 4),
                          recall=round(recall, 4),
                          f1=round(f1, 4),
                          tp=total_tp, fp=total_fp, fn=total_fn)

    def _count(self, predicted, gold, strategy) -> tuple[int, int, int]:
        matched_gold = set()
        matched_pred = set()
        for pi, p in enumerate(predicted):
            for gi, g in enumerate(gold):
                if gi in matched_gold:
                    continue
                matched = False
                if strategy == "exact":
                    matched = _exact_match(p, g)
                elif strategy == "partial":
                    matched = _partial_match(p, g) >= 0.5
                elif strategy == "type_agnostic":
                    matched = _type_agnostic_match(p, g)
                if matched:
                    matched_pred.add(pi)
                    matched_gold.add(gi)
                    break
        tp = len(matched_pred)
        fp = len(predicted) - tp
        fn = len(gold)      - len(matched_gold)
        return tp, fp, fn


def analyse_errors(predicted_df: pd.DataFrame,
                   gold_df:      pd.DataFrame) -> tuple[ErrorReport, pd.DataFrame]:
    """
    Classify every mismatch into one of four error types.

    boundary_error : overlapping span, same label, different text
    type_error     : same text, different label
    missing        : gold entity with no matching prediction
    spurious       : predicted entity with no matching gold

    Args:
        predicted_df: Predicted entities DataFrame.
        gold_df:      Gold-standard entities DataFrame.

    Returns:
        (ErrorReport, detail_df) — detail_df lists every mismatch.
    """
    predicted = df_to_entities(predicted_df)
    gold      = df_to_entities(gold_df)
    gold_ids  = {e.text_id for e in gold}
    predicted = [e for e in predicted if e.text_id in gold_ids]

    report = ErrorReport()
    details: list[dict] = []
    matched_gold = set()
    matched_pred = set()

    # Pass 1: exact matches (true positives)
    for pi, p in enumerate(predicted):
        for gi, g in enumerate(gold):
            if gi in matched_gold:
                continue
            if _exact_match(p, g):
                matched_pred.add(pi); matched_gold.add(gi); break

    # Pass 2: type errors
    for pi, p in enumerate(predicted):
        if pi in matched_pred: continue
        for gi, g in enumerate(gold):
            if gi in matched_gold: continue
            if p.text == g.text and p.label != g.label:
                report.type_errors += 1
                details.append({"text_id": p.text_id, "error_type": "type_error",
                                 "entity_text": p.text,
                                 "predicted_label": p.label, "gold_label": g.label})
                matched_pred.add(pi); matched_gold.add(gi); break

    # Pass 3: boundary errors
    for pi, p in enumerate(predicted):
        if pi in matched_pred: continue
        for gi, g in enumerate(gold):
            if gi in matched_gold: continue
            if p.label == g.label and _spans_overlap(p, g):
                report.boundary_errors += 1
                details.append({"text_id": p.text_id, "error_type": "boundary_error",
                                 "entity_text": f"pred='{p.text}' | gold='{g.text}'",
                                 "predicted_label": p.label, "gold_label": g.label})
                matched_pred.add(pi); matched_gold.add(gi); break

    # Remaining unmatched predictions → spurious
    for pi, p in enumerate(predicted):
        if pi not in matched_pred:
            report.spurious += 1
            details.append({"text_id": p.text_id, "error_type": "spurious",
                             "entity_text": p.text,
                             "predicted_label": p.label, "gold_label": "—"})

    # Remaining unmatched gold → missing
    for gi, g in enumerate(gold):
        if gi not in matched_gold:
            report.missing += 1
            details.append({"text_id": g.text_id, "error_type": "missing",
                             "entity_text": g.text,
                             "predicted_label": "—", "gold_label": g.label})

    detail_df = pd.DataFrame(
        details,
        columns=["text_id", "error_type", "entity_text", "predicted_label", "gold_label"],
    )
    return report, detail_df


def print_metrics_table(results: list[EvalResult]):
    """Print a formatted comparison table of all EvalResult objects."""
    header = (f"{'Strategy':<16} {'Averaging':<10} {'Precision':>10}"
              f" {'Recall':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        print(f"{r.strategy:<16} {r.averaging:<10} {r.precision:>10.4f}"
              f" {r.recall:>8.4f} {r.f1:>8.4f} {r.tp:>6} {r.fp:>6} {r.fn:>6}")
    print(sep)


def plot_error_distribution(report: ErrorReport,
                             save_path: str = "tier3_errors.png"):
    """Bar chart of the four error categories."""
    cats   = ["boundary_error", "type_error", "missing", "spurious"]
    counts = [report.boundary_errors, report.type_errors,
               report.missing, report.spurious]
    clrs   = ["#2E86AB", "#A23B72", "#C73E1D", "#F18F01"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cats, counts, color=clrs, edgecolor="white", width=0.55)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("NER Error Distribution (spaCy vs Gold Standard)",
                 fontsize=13, fontweight="bold")
    ax.set_xticklabels([c.replace("_", " ").title() for c in cats], fontsize=10)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_strategy_comparison(results: list[EvalResult],
                              save_path: str = "tier3_strategies.png"):
    """Grouped bar chart: F1 by strategy and averaging method."""
    strategies = ["exact", "partial", "type_agnostic"]
    averagings = ["micro", "macro"]
    x      = np.arange(len(strategies))
    width  = 0.3
    clrs   = {"micro": "#2E86AB", "macro": "#F18F01"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, avg in enumerate(averagings):
        f1s = [
            next((r.f1 for r in results if r.strategy == s and r.averaging == avg), 0.0)
            for s in strategies
        ]
        ax.bar(x + (i - 0.5) * width, f1s, width, label=avg.capitalize(),
               color=clrs[avg], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in strategies])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("F1 by Strategy and Averaging Method (spaCy)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Averaging")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared utility
# ─────────────────────────────────────────────────────────────────────────────

def _f1(precision: float, recall: float) -> float:
    return (2 * precision * recall) / (precision + recall) \
        if (precision + recall) > 0 else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Main — runs all three tiers in sequence
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading data...")
    df   = load_data()
    gold = pd.read_csv("data/gold_entities.csv")

    print("Extracting spaCy entities...")
    entities = extract_spacy_entities(df, nlp)

    # ─────────────────────────────────────────────────────────────────────
    # TIER 1
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  TIER 1 — Per-Category NER Analysis")
    print("═" * 60)

    pivot = entity_counts_by_category(entities, df)
    print("\nEntity counts by label × category:")
    print(pivot.to_string())

    cat_metrics = evaluate_per_category(entities, gold, df)
    print("\nPer-category evaluation:")
    if not cat_metrics.empty:
        print(cat_metrics.to_string(index=False))
    else:
        print("  (No gold annotations span multiple categories.)")

    print("\nGenerating Tier 1 plots...")
    plot_heatmap(pivot)
    plot_grouped_bar(pivot)
    plot_per_category_metrics(cat_metrics)
    print(TIER1_ANALYSIS)

    # ─────────────────────────────────────────────────────────────────────
    # TIER 2
    # ─────────────────────────────────────────────────────────────────────
    print("═" * 60)
    print("  TIER 2 — Entity Aggregation Pipeline")
    print("═" * 60)

    norm_entities = normalise_entities_df(entities)
    print(f"\nUnique entities before normalisation : {entities['entity_text'].nunique()}")
    print(f"Unique entities after normalisation  : {norm_entities['entity_text'].nunique()}")

    edge_list = compute_cooccurrence(norm_entities)
    print("\nTop-10 co-occurring entity pairs:")
    print(edge_list.head(10).to_string(index=False))

    tfidf_df = compute_entity_tfidf(norm_entities, df)
    print("\nTop-5 distinctive entities per category (TF-IDF):")
    for cat in CATEGORIES:
        top5 = tfidf_df[tfidf_df["category"] == cat].head(5)
        print(f"\n  {cat.upper()}")
        print(top5[["entity", "tfidf"]].to_string(index=False))

    print("\nGenerating Tier 2 network plot...")
    plot_cooccurrence_network(edge_list, top_n=20)

    # ─────────────────────────────────────────────────────────────────────
    # TIER 3
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  TIER 3 — Custom NER Evaluator")
    print("═" * 60)

    evaluator = NERSEvaluator()
    results: list[EvalResult] = [
        evaluator.evaluate(entities, gold, strategy=s, averaging=a)
        for s in ("exact", "partial", "type_agnostic")
        for a in ("micro", "macro")
    ]
    print_metrics_table(results)

    report, detail_df = analyse_errors(entities, gold)
    print(f"\nError breakdown:")
    print(f"  Boundary errors : {report.boundary_errors}")
    print(f"  Type errors     : {report.type_errors}")
    print(f"  Missing         : {report.missing}")
    print(f"  Spurious        : {report.spurious}")
    print("\nSample errors (first 10):")
    print(detail_df.head(10).to_string(index=False))

    print("\nGenerating Tier 3 plots...")
    plot_error_distribution(report)
    plot_strategy_comparison(results)

    print("\n✓ All challenge tiers complete.")