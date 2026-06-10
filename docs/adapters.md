# Writing an adapter

An adapter is the only code that knows about a clinical domain. It does four
things: translate the raw input, name the report sections, supply a prompt per
section, and describe itself. This guide builds one end to end against the
`examples/generic` payload.

## The contract

Every adapter subclasses `BaseAdapter` and implements six methods:

| Method | Returns | What it is for |
|--------|---------|----------------|
| `get_domain` | `str` | Stable machine name, used as the API `domain`. |
| `get_knowledge_base_path` | `Path` | Folder of `.txt`/`.md` files to retrieve from. |
| `format_findings` | `ClinicalFindings` | Map your raw payload onto the generic schema. |
| `get_report_sections` | `list[str]` | Ordered section titles. |
| `get_prompt_templates` | `dict[str, str]` | One prompt per section title. |
| `get_report_metadata` | `dict[str, str]` | Display name, disclaimers, and so on. |

There is also an optional `extract_patient_context(raw)` you can override when
demographics are nested somewhere unusual; the default reads flat `age`/`sex`.

## Step 1: translate the input

Our payload (`examples/generic/sample_findings.json`) already carries a list of
findings with status flags, so the mapping is almost one to one:

```python
from pathlib import Path
from adapters.base import BaseAdapter
from core.schema import ClinicalFinding, ClinicalFindings


class LabPanelAdapter(BaseAdapter):
    def get_domain(self) -> str:
        return "lab_panel"

    def get_knowledge_base_path(self) -> Path:
        return Path(__file__).parent / "knowledge"

    def format_findings(self, raw: dict) -> ClinicalFindings:
        findings = [
            ClinicalFinding(
                name=item["name"],
                value=float(item["value"]),
                unit=item.get("unit"),
                reference_range=item.get("reference_range"),
                status=item.get("status", "normal"),
            )
            for item in raw["findings"]
        ]
        return ClinicalFindings(
            domain=self.get_domain(),
            findings=findings,
            summary_stats={"count": len(findings)},
        )
```

`status` accepts `normal`, `borderline`, `abnormal`, or `critical`. Anything you
want to keep but the schema does not model goes in a finding's `metadata` dict.

## Step 2: name the sections and write prompts

Section titles double as retrieval queries, so make them descriptive. Each
template may use four placeholders, filled in by the generator:
`{section}`, `{findings}`, `{patient}`, and `{evidence}`. Unknown placeholders
are left alone, so a stray brace will not crash generation.

```python
    def get_report_sections(self) -> list[str]:
        return ["Summary", "Interpretation", "Recommendations"]

    def get_prompt_templates(self) -> dict[str, str]:
        base = "Patient: {patient}\n\nFindings:\n{findings}\n\nEvidence:\n{evidence}\n\n"
        return {
            "Summary": base + "Summarize the panel in two short paragraphs. Cite as [n].",
            "Interpretation": base + "Interpret each abnormal value against its range. Cite as [n].",
            "Recommendations": base + "Suggest measured next steps. Do not prescribe treatment.",
        }

    def get_report_metadata(self) -> dict[str, str]:
        return {
            "display_name": "Lab Panel",
            "disclaimer": "Decision support only. Not a diagnosis.",
        }
```

## Step 3: add a knowledge base

Drop a few `.txt` or `.md` files into the folder returned by
`get_knowledge_base_path`. Files are chunked by paragraph and embedded locally.
Keep each file focused on one topic; retrieval works best when a chunk answers a
single question. Use real, citable sources rather than invented numbers.

## Step 4: register it

Wire it into the API registry. The bundled example does this in
`api/main.py:_load_adapters`; follow the same pattern:

```python
from api.routes import registry
registry.register(LabPanelAdapter())
```

`register` loads and embeds the knowledge base once at startup. In tests, pass
`embed=False` to skip the embedding model entirely.

## Step 5: test it

You do not need a live LLM to test an adapter. Test the pure mapping directly,
and test generation with a fake client:

```python
from unittest.mock import MagicMock
from core.generator import LLMClient, ReportGenerator
from core.pipeline import ClinicalReportPipeline
from core.retriever import Retriever
from core.verifier import ReportVerifier


def test_format_findings_maps_status():
    adapter = LabPanelAdapter()
    findings = adapter.format_findings(
        {"findings": [{"name": "HbA1c", "value": 6.1, "status": "abnormal"}]}
    )
    assert findings.findings[0].status == "abnormal"


def test_pipeline_with_fake_llm():
    client = MagicMock(spec=LLMClient)
    client.complete.return_value = "Body [1]."
    pipeline = ClinicalReportPipeline(
        retriever=Retriever(),                       # empty is fine for a smoke test
        generator=ReportGenerator(model="fake", client=client),
        verifier=ReportVerifier(),
        adapter=LabPanelAdapter(),
    )
    report = pipeline.run(
        {"findings": [{"name": "HbA1c", "value": 6.1}]},
        {"age": 45, "sex": "F"},
    )
    assert len(report.sections) == 3
```

See `tests/integration/test_full_pipeline.py` for the same pattern applied to
the real Neuraxis adapter with a mocked embedder and LLM.

## Tips

- Keep all branching on the input shape inside `format_findings`. The rest of
  the adapter should be declarative.
- If two domains share knowledge, give each its own `VectorStore` and let the
  `Retriever` fan out across both; it deduplicates and re-ranks for you.
- Resist putting domain logic in `core/`. If you feel the urge, it is a sign the
  adapter interface needs another method, not that the core needs a special case.
