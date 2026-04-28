
# Stretch: Custom NER Rules Analysis

## Before/After Comparison: Entity Counts

The following table compares the entity counts between the base spaCy model (`en_core_web_sm`) and the two pipeline configurations using the custom `EntityRuler`.

| Entity Label | Base Model | Ruler (Before NER) | Ruler (After NER) |
| :--- | :--- | :--- | :--- |
| **AGREEMENT** (Custom) | 0 | 6 | 0 |
| **CLIMATE_EVENT** (Custom) | 0 | 5 | 3 |
| **POLICY** (Custom) | 0 | 4 | 3 |
| **REPORT** (Custom) | 0 | 2 | 0 |
| **THRESHOLD** (Custom) | 0 | 1 | 0 |
| **ORG** | 184 | 181 | 184 |
| **EVENT** | 8 | 2 | 8 |
| **DATE** | 256 | 257 | 256 |
| **QUANTITY** | 92 | 91 | 92 |

*Note: Custom labels are only present in the Ruler configurations. Some standard labels (ORG, EVENT) decreased in the "Before" configuration as the Ruler captured those spans first.*

## Evaluation Delta (Standard Labels Only)

Evaluation performed against `gold_entities.csv`, considering only standard spaCy labels (`ORG`, `GPE`, `DATE`, `LAW`, `MONEY`, `PERSON`, `QUANTITY`, `LOC`, `EVENT`, `WORK_OF_ART`).

| Metric | Base Model | Ruler (Before NER) | Ruler (After NER) |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.6567 | 0.7500 | 0.6567 |
| **Recall** | 0.6471 | 0.6618 | 0.6471 |
| **F1 Score** | 0.6519 | 0.7031 | 0.6519 |

## Qualitative Analysis of Custom Rules

The custom rules successfully captured domain-specific terminology that the base model either misclassified or missed entirely.

*   **AGREEMENT**: "Paris Agreement" was previously classified as `EVENT` by the base model. By using the `EntityRuler` before the NER, it is correctly identified as an `AGREEMENT`.
*   **CLIMATE_EVENT**: "COP28" and "Climate Ambition Summit" were often missed or labeled as `ORG`. The token pattern `[{"LOWER": "cop"}, {"IS_DIGIT": True}]` successfully caught "COP28" in Article 2.
*   **THRESHOLD**: "1.5 degrees Celsius" in Article 1 was correctly identified, providing more domain-specific context than a generic `QUANTITY` label.
*   **POLICY**: "nationally determined contributions" (NDCs) were captured as `POLICY`, whereas the base model ignored them.

## Impact Analysis

The introduction of the custom `EntityRuler` significantly improved the precision of the NER pipeline (from 0.6567 to 0.7500) when placed before the statistical NER. This improvement is primarily due to the rules "claiming" spans that were previously misclassified by the base model. For example, "Paris Agreement" was mislabeled as `EVENT` (standard label) in the base model, but because it is labeled as `AGREEMENT` (custom label) in the "Before" configuration, it no longer contributes as a false positive for `EVENT` in the standard evaluation. While this technically increases precision for standard labels, the real value lies in the addition of domain-specific categories like `REPORT` (e.g., "Sixth Assessment Report") and `POLICY` (e.g., "NDCs"), which were previously invisible to the system. However, the "After" configuration showed minimal impact because the statistical model's existing (often incorrect) labels took priority, preventing the `EntityRuler` from applying its more specific domain knowledge to already-labeled spans.
