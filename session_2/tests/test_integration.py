import pytest
from litestar.testing import TestClient
from unittest.mock import MagicMock
from src.api import app
from src.model import RideDurationModel

@pytest.fixture(scope="module")
def client():
    mock = MagicMock(spec=RideDurationModel)
    mock.predict.return_value = 15.5
    app.dependency_overrides[RideDurationModel] = lambda: mock
    with TestClient(app=app) as c:
        yield c
    app.dependency_overrides.clear()

# ── Happy path ─────────────────────────────────────────
def test_predict_success(client):
    resp = client.post("/predict", json={"distance_km": 5.0, "passengers": 2})
    assert resp.status_code == 201
    body = resp.json()
    assert "duration_min" in body
    assert isinstance(body["duration_min"], float)

# ── Validation errors ──────────────────────────────────
@pytest.mark.parametrize("payload,expected_status", [
    ({"distance_km": -1.0, "passengers": 2},   422),   # negative distance
    ({"distance_km": 5.0,  "passengers": 0},   422),   # zero passengers
    ({"passengers": 2},                        422),   # missing required field
    ({},                                       422),   # empty body
])
def test_predict_validation(client, payload, expected_status):
    assert client.post("/predict", json=payload).status_code == expected_status

# ── Health + contract checks ───────────────────────────
def test_health(client):
    assert client.get("/health").json()["status"] == "healthy"

def test_response_schema(client):
    body = client.post("/predict", json={"distance_km": 5.0}).json()
    assert set(body.keys()) == {"duration_min", "status"}
