"""Adapter for the Neuraxis ADHD EEG output format.

Neuraxis emits a probability plus SHAP feature attributions and a normative
comparison. This adapter reshapes that into the generic schema and supplies
ADHD-specific report sections and prompts. It is the reference example for how
to specialise the pipeline without touching the core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adapters.base import BaseAdapter
from core.schema import ClinicalFinding, ClinicalFindings, FindingStatus

_KB_PATH = Path(__file__).resolve().parent / "knowledge"

# z-score magnitude above which a normative deviation is called abnormal.
_Z_ABNORMAL = 2.0
_Z_BORDERLINE = 1.5


class NeuraxisAdapter(BaseAdapter):
    def get_domain(self) -> str:
        return "adhd_neuraxis"

    def get_knowledge_base_path(self) -> Path:
        return _KB_PATH

    def format_findings(self, raw: dict) -> ClinicalFindings:
        """Map a Neuraxis payload onto generic findings.

        The classifier probability becomes a finding in its own right, each
        SHAP feature becomes a finding tagged with its attribution, and the
        normative comparison contributes a z-scored finding when present.
        """
        findings: list[ClinicalFinding] = []

        if (prob := raw.get("adhd_probability")) is not None:
            findings.append(
                ClinicalFinding(
                    name="adhd_probability",
                    value=float(prob),
                    unit=None,
                    reference_range="0.0 - 1.0",
                    status=_probability_status(float(prob)),
                    confidence=float(prob) if prob >= 0.5 else 1.0 - float(prob),
                    metadata={"kind": "classifier_output"},
                )
            )

        for feature in raw.get("top_shap_features", []):
            shap = float(feature.get("shap_value", 0.0))
            findings.append(
                ClinicalFinding(
                    name=feature["name"],
                    value=float(feature.get("value", 0.0)),
                    status=(
                        FindingStatus.ABNORMAL
                        if abs(shap) >= 0.1
                        else FindingStatus.BORDERLINE
                    ),
                    confidence=min(1.0, abs(shap) * 4),
                    metadata={"shap_value": shap, "kind": "shap_feature"},
                )
            )

        if (norm := raw.get("normative_comparison")) is not None:
            z = float(norm.get("z_score", 0.0))
            findings.append(
                ClinicalFinding(
                    name=norm.get("feature", "normative_feature"),
                    value=float(norm.get("patient_value", 0.0)),
                    reference_range=_norm_range(norm),
                    status=_zscore_status(z),
                    confidence=1.0,
                    metadata={
                        "z_score": z,
                        "norm_mean": norm.get("norm_mean"),
                        "norm_std": norm.get("norm_std"),
                        "kind": "normative_comparison",
                    },
                )
            )

        return ClinicalFindings(
            domain=self.get_domain(),
            findings=findings,
            source_model="neuraxis",
            summary_stats={
                "adhd_probability": raw.get("adhd_probability"),
                "feature_count": len(raw.get("top_shap_features", [])),
            },
        )

    def get_report_sections(self) -> list[str]:
        return [
            "Clinical Summary",
            "EEG Biomarker Analysis",
            "Normative Comparison",
            "Clinical Implications",
            "Recommendations",
            "Limitations and Disclaimers",
        ]

    def get_prompt_templates(self) -> dict[str, str]:
        common = (
            "Patient: {patient}\n\n"
            "Findings:\n{findings}\n\n"
            "Evidence:\n{evidence}\n\n"
        )
        return {
            "Clinical Summary": (
                common
                + "Write a concise clinical summary of the EEG-based ADHD "
                "assessment. State the model probability in plain terms and the "
                "one or two most influential features. Two short paragraphs. "
                "Cite evidence inline as [n]."
            ),
            "EEG Biomarker Analysis": (
                common
                + "Discuss each EEG feature and what an elevated or reduced "
                "value typically indicates, grounded in the evidence. Do not "
                "overstate certainty. Cite inline as [n]."
            ),
            "Normative Comparison": (
                common
                + "Interpret the patient's value against the normative "
                "distribution using the z-score. Explain what the deviation "
                "means without implying a diagnosis. Cite inline as [n]."
            ),
            "Clinical Implications": (
                common
                + "Describe what these findings may imply clinically, framed as "
                "supporting evidence rather than a standalone diagnosis. Cite "
                "inline as [n]."
            ),
            "Recommendations": (
                common
                + "Offer measured next steps a clinician might consider "
                "(further assessment, clinical correlation). Avoid prescribing "
                "treatment. Cite inline as [n] where relevant."
            ),
            "Limitations and Disclaimers": (
                common
                + "State the limitations of EEG-based ADHD assessment and that "
                "this report is decision support, not a diagnosis. Reference the "
                "evidence on biomarker reliability where available. Cite as [n]."
            ),
        }

    def get_report_metadata(self) -> dict[str, str]:
        return {
            "display_name": "ADHD EEG Assessment (Neuraxis)",
            "modality": "EEG",
            "disclaimer": (
                "Decision support only. Not a diagnostic device. Findings must "
                "be interpreted by a qualified clinician alongside clinical "
                "history and standardised assessment."
            ),
        }


def _probability_status(p: float) -> FindingStatus:
    if p >= 0.85:
        return FindingStatus.CRITICAL
    if p >= 0.6:
        return FindingStatus.ABNORMAL
    if p >= 0.4:
        return FindingStatus.BORDERLINE
    return FindingStatus.NORMAL


def _zscore_status(z: float) -> FindingStatus:
    az = abs(z)
    if az >= 3.0:
        return FindingStatus.CRITICAL
    if az >= _Z_ABNORMAL:
        return FindingStatus.ABNORMAL
    if az >= _Z_BORDERLINE:
        return FindingStatus.BORDERLINE
    return FindingStatus.NORMAL


def _norm_range(norm: dict[str, Any]) -> str:
    mean, std = norm.get("norm_mean"), norm.get("norm_std")
    if mean is None or std is None:
        return ""
    lo, hi = mean - 2 * std, mean + 2 * std
    return f"{lo:.2f} - {hi:.2f}"
