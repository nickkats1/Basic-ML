"""Tests for the Flask serving app."""

import importlib


def _client(monkeypatch, trained):
    """Reload the app pointed at the session-trained model artifact."""
    monkeypatch.setenv("MODEL_PATH", trained["config"].model_path)
    import app as app_module

    importlib.reload(app_module)
    return app_module.app.test_client()


def test_health(monkeypatch, trained):
    client = _client(monkeypatch, trained)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_home_renders_form(monkeypatch, trained):
    client = _client(monkeypatch, trained)
    resp = client.get("/")
    assert resp.status_code == 200


def test_predict_endpoint(monkeypatch, trained):
    client = _client(monkeypatch, trained)
    row = trained["X_test"].iloc[0]
    form = {name: str(row[name]) for name in trained["config"].features}
    resp = client.post("/predict", data=form)
    assert resp.status_code == 200


def test_predict_endpoint_rejects_bad_input(monkeypatch, trained):
    client = _client(monkeypatch, trained)
    resp = client.post("/predict", data={"age": "not-a-number"})
    assert resp.status_code == 400
