"""Request and response models for the HTTP API.

These wrap the core schema for the wire so the public API surface can evolve
independently of the internal models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.schema import GeneratedReport


class GenerateRequest(BaseModel):
    """Payload for ``POST /api/generate``."""

    findings: dict[str, Any] = Field(
        ..., description="Raw, domain-specific findings as the adapter expects them."
    )
    patient: dict[str, Any] = Field(
        default_factory=dict,
        description="Anonymous demographic context (age, sex, and so on).",
    )
    domain: str = Field(
        ..., description="Registered adapter domain to use, e.g. 'adhd_neuraxis'."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "domain": "adhd_neuraxis",
                "patient": {"age": 12, "sex": "M"},
                "findings": {
                    "patient_age": 12,
                    "patient_sex": "M",
                    "adhd_probability": 0.73,
                    "top_shap_features": [
                        {"name": "F3_theta_rel", "value": 0.28, "shap_value": 0.15}
                    ],
                },
            }
        }
    }


class GenerateResponse(BaseModel):
    """Response for ``POST /api/generate``."""

    domain: str = Field(..., description="Domain the report was generated for.")
    report: GeneratedReport = Field(..., description="The generated, verified report.")


class DomainInfo(BaseModel):
    """One registered adapter, as returned by ``GET /api/domains``."""

    domain: str
    metadata: dict[str, str] = Field(default_factory=dict)
    sections: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response for the health endpoints."""

    status: str = Field(..., description="'ok' when the service is ready.")
    loaded_domains: list[str] = Field(default_factory=list)
    version: str
