"""The contract every domain adapter implements.

An adapter is the only place that knows about a specific clinical domain. It
maps a vendor or instrument payload onto the generic schema, names the report
sections, and supplies the prompts. The core pipeline depends on this
interface and nothing more, so adding a domain never means touching core code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.schema import ClinicalFindings


class BaseAdapter(ABC):
    """Abstract base for domain adapters."""

    @abstractmethod
    def get_domain(self) -> str:
        """Stable machine name for the domain, e.g. ``"adhd_neuraxis"``."""

    @abstractmethod
    def get_knowledge_base_path(self) -> Path:
        """Directory holding this domain's ``.txt``/``.md`` knowledge files."""

    @abstractmethod
    def format_findings(self, raw: dict) -> ClinicalFindings:
        """Translate a raw vendor payload into generic findings.

        Args:
            raw: The domain-specific input dictionary.

        Returns:
            A populated :class:`~core.schema.ClinicalFindings`.
        """

    @abstractmethod
    def get_prompt_templates(self) -> dict[str, str]:
        """Map each report section title to its prompt template."""

    @abstractmethod
    def get_report_sections(self) -> list[str]:
        """Ordered list of section titles the report should contain."""

    @abstractmethod
    def get_report_metadata(self) -> dict[str, str]:
        """Static metadata for the domain (display name, disclaimers, etc.)."""

    def extract_patient_context(self, raw: dict) -> dict:
        """Pull demographic fields out of a raw payload.

        Adapters with patient data nested elsewhere can override this. The
        default assumes flat ``age``/``sex`` keys and is forgiving about both.
        """
        return {
            "age": raw.get("age") or raw.get("patient_age"),
            "sex": raw.get("sex") or raw.get("patient_sex", "U"),
        }
