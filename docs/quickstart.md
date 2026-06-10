# Quickstart

## Requirements

- Python 3.10, 3.11, or 3.12
- An OpenAI-compatible LLM endpoint (hosted, or a local server such as Ollama,
  vLLM, or LM Studio)

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB). After
that, retrieval is fully offline.

## Install

```bash
git clone https://github.com/jplgarin/clinical-rag-workflow.git
cd clinical-rag-workflow
pip install -r requirements.txt
pip install -e ".[dev]"      # dev extras: pytest, flake8, mypy
```

## Configure

```bash
cp .env.example .env
```

Set at least:

```bash
LLM_API_BASE=https://api.openai.com/v1   # or http://localhost:11434/v1 for Ollama
LLM_API_KEY=...                          # blank is fine for local servers
GENERATOR_MODEL=gpt-4o-mini              # whatever your backend serves
```

## Run the service

```bash
uvicorn api.main:app --reload
```

- Web UI: <http://localhost:8000/app>
- OpenAPI docs: <http://localhost:8000/docs>
- Health: <http://localhost:8000/api/health>

The bundled ADHD/Neuraxis example registers on startup, so the UI has a domain
to play with immediately.

## Generate a report from the command line

```bash
curl -s http://localhost:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON' | python -m json.tool
{
  "domain": "adhd_neuraxis",
  "patient": {"age": 12, "sex": "M"},
  "findings": {
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
}
JSON
```

## Run the tests

```bash
pytest
```

Everything external (the LLM and the embedding model) is mocked in tests, so no
API key or network access is required to run the suite.

## Use the pipeline directly in Python

```python
from core.generator import ReportGenerator
from core.pipeline import ClinicalReportPipeline
from core.retriever import Retriever, VectorStore
from core.verifier import ReportVerifier
from examples.adhd_neuraxis.adapter import NeuraxisAdapter

adapter = NeuraxisAdapter()
store = VectorStore(name=adapter.get_domain())
store.load_documents(adapter.get_knowledge_base_path())
store.embed_documents()

pipeline = ClinicalReportPipeline(
    retriever=Retriever([store]),
    generator=ReportGenerator(),     # reads model + endpoint from env
    verifier=ReportVerifier(),
    adapter=adapter,
)

report = pipeline.run(
    {"adhd_probability": 0.73, "top_shap_features": []},
    {"age": 12, "sex": "M"},
)
print(report.sections[0].content)
```
