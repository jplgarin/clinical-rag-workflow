"""API tests.

We mount the router on a bare app and register a stub adapter with
``embed=False`` so nothing touches the embedding model or the network.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from adapters.base import BaseAdapter
from core.generator import LLMClient, ReportGenerator
from core.schema import ClinicalFinding, ClinicalFindings
from api.routes import registry, router


class ApiStubAdapter(BaseAdapter):
    def get_domain(self):
        return "stub"

    def get_knowledge_base_path(self):
        return Path("does-not-exist")

    def format_findings(self, raw):
        if raw.get("boom"):
            raise ValueError("bad findings")
        return ClinicalFindings(
            domain="stub", findings=[ClinicalFinding(name="m", value=1.0)]
        )

    def get_prompt_templates(self):
        return {"Summary": "{evidence}"}

    def get_report_sections(self):
        return ["Summary"]

    def get_report_metadata(self):
        return {"display_name": "Stub Domain"}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)

    fake_llm = MagicMock(spec=LLMClient)
    fake_llm.complete.return_value = "Section body."
    generator = ReportGenerator(model="fake", client=fake_llm)

    registry._pipelines.clear()
    registry._adapters.clear()
    registry.register(ApiStubAdapter(), generator=generator, embed=False)
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "stub" in body["loaded_domains"]


def test_list_domains(client):
    r = client.get("/api/domains")
    assert r.status_code == 200
    data = r.json()
    assert data[0]["domain"] == "stub"
    assert data[0]["metadata"]["display_name"] == "Stub Domain"
    assert data[0]["sections"] == ["Summary"]


def test_generate_success(client):
    r = client.post(
        "/api/generate",
        json={"domain": "stub", "patient": {"age": 12}, "findings": {}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["domain"] == "stub"
    assert body["report"]["sections"][0]["title"] == "Summary"


def test_generate_unknown_domain(client):
    r = client.post("/api/generate", json={"domain": "nope", "findings": {}})
    assert r.status_code == 404
    assert "unknown domain" in r.json()["detail"]


def test_generate_adapter_error_is_422(client):
    r = client.post(
        "/api/generate", json={"domain": "stub", "findings": {"boom": True}}
    )
    assert r.status_code == 422
    assert "bad findings" in r.json()["detail"]


def test_registry_missing_domain_raises():
    registry._pipelines.clear()
    with pytest.raises(KeyError):
        registry.get("absent")
