"""HTTP routes and the in-process adapter registry.

The registry owns one fully wired pipeline per domain. Building a pipeline
embeds that domain's knowledge base once at startup, so requests only pay for
retrieval and generation.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from adapters.base import BaseAdapter
from core import __version__
from core.generator import ReportGenerator
from core.pipeline import ClinicalReportPipeline
from core.retriever import Retriever, VectorStore
from core.verifier import ReportVerifier
from api.models import (
    DomainInfo,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)


class Registry:
    """Holds the adapters and pipelines available to the API."""

    def __init__(self) -> None:
        self._pipelines: dict[str, ClinicalReportPipeline] = {}
        self._adapters: dict[str, BaseAdapter] = {}

    def register(
        self,
        adapter: BaseAdapter,
        generator: Optional[ReportGenerator] = None,
        embed: bool = True,
    ) -> None:
        """Wire an adapter into a ready-to-run pipeline.

        Args:
            adapter: The domain adapter to register.
            generator: Generator to use. A default is created when omitted.
            embed: When ``True``, load and embed the domain knowledge base now.
                Set ``False`` in tests to skip the embedding model.
        """
        domain = adapter.get_domain()
        retriever = Retriever()
        if embed:
            store = VectorStore(name=domain)
            kb_path = adapter.get_knowledge_base_path()
            if kb_path.exists():
                store.load_documents(kb_path)
                store.embed_documents()
                retriever.add_store(store)
            else:
                logger.warning("knowledge path missing for %s: %s", domain, kb_path)

        pipeline = ClinicalReportPipeline(
            retriever=retriever,
            generator=generator or ReportGenerator(),
            verifier=ReportVerifier(),
            adapter=adapter,
        )
        self._pipelines[domain] = pipeline
        self._adapters[domain] = adapter
        logger.info("registered domain '%s'", domain)

    def get(self, domain: str) -> ClinicalReportPipeline:
        if domain not in self._pipelines:
            raise KeyError(domain)
        return self._pipelines[domain]

    def domains(self) -> list[str]:
        return sorted(self._pipelines)

    def adapter(self, domain: str) -> BaseAdapter:
        return self._adapters[domain]


# Single process-wide registry, populated by the app lifespan.
registry = Registry()

router = APIRouter(prefix="/api", tags=["reports"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness and readiness, including which domains are loaded."""
    return HealthResponse(
        status="ok",
        loaded_domains=registry.domains(),
        version=__version__,
    )


@router.get("/domains", response_model=list[DomainInfo])
def list_domains() -> list[DomainInfo]:
    """List every registered domain with its metadata and section layout."""
    out: list[DomainInfo] = []
    for domain in registry.domains():
        adapter = registry.adapter(domain)
        out.append(
            DomainInfo(
                domain=domain,
                metadata=adapter.get_report_metadata(),
                sections=adapter.get_report_sections(),
            )
        )
    return out


@router.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate a verified report for the requested domain."""
    try:
        pipeline = registry.get(request.domain)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"unknown domain '{request.domain}'. "
            f"Available: {registry.domains()}",
        )
    try:
        report = pipeline.run(request.findings, request.patient)
    except Exception as exc:  # surface adapter/generator errors as 422
        logger.exception("generation failed for domain %s", request.domain)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return GenerateResponse(domain=request.domain, report=report)
