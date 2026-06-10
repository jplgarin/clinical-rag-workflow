# ADHD / Neuraxis example

A worked adapter that turns the Neuraxis EEG classifier output into a clinical
report. It is the canonical example of specialising the pipeline for a domain
without touching `core/`.

## What it shows

- Mapping a vendor payload (probability + SHAP attributions + a normative
  comparison) onto the generic `ClinicalFindings` schema.
- Per-section prompt templates tuned for an EEG/ADHD report.
- A small knowledge base of real, published findings used for retrieval.

## Input format

```json
{
  "patient_age": 12,
  "patient_sex": "M",
  "adhd_probability": 0.73,
  "top_shap_features": [
    {"name": "F3_theta_rel", "value": 0.28, "shap_value": 0.15}
  ],
  "normative_comparison": {
    "feature": "TBR_Fz", "patient_value": 3.2,
    "norm_mean": 2.1, "norm_std": 0.4, "z_score": 2.75
  }
}
```

See `sample_findings.json` for a complete example.

## Report sections

Clinical Summary, EEG Biomarker Analysis, Normative Comparison, Clinical
Implications, Recommendations, Limitations and Disclaimers.

## Knowledge base

`knowledge/` holds short factual summaries of published work on EEG markers in
ADHD (theta/beta ratio, the decline in TBR effect size over time, EEG subtypes,
a quantitative-EEG review, and ML-on-EEG with SHAP). These are illustrative
summaries for retrieval, not a substitute for the primary sources.

> This example is decision-support tooling for demonstration. It is not a
> medical device and must not be used to diagnose.
