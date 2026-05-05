from fastapi.testclient import TestClient

from openai_api_server_via_codex.server import create_app


def test_healthz():
    client = TestClient(create_app())
    response = client.get("/healthz")
    assert response.json() == {"status": "ok"}
