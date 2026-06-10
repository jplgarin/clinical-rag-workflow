"""Cover the app wiring in api.main without running the embedding lifespan.

The TestClient is used outside a context manager on purpose, so the startup
hook (which would load the embedding model) does not fire.
"""

from fastapi.testclient import TestClient

from api.main import app
from core import __version__


def test_plain_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": __version__}


def test_root_redirects_to_app():
    client = TestClient(app)
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/app/"


def test_web_static_mounted():
    client = TestClient(app)
    r = client.get("/app/")
    assert r.status_code == 200
    assert "Clinical RAG Workflow" in r.text
