from fastapi.testclient import TestClient

from somahouse.app import app

client = TestClient(app)


def test_live_controller_and_disturbances():
    response = client.post("/api/start", json={"seed": 572, "controller": "feedback"})
    assert response.status_code == 200, response.text
    session = response.json()["session"]
    first = client.post("/api/advance", json={"session": session, "steps": 12})
    assert first.status_code == 200
    assert len(first.json()["trace"]) == 12
    assert first.json()["trace"][-1]["hour"] == 2
    changed = client.post("/api/advance", json={"session": session, "steps": 1,
                           "setpoint": [24, 24, 24], "people": [4, 0, 0], "outside_offset": -5,
                           "daylight_factor": 0})
    assert changed.status_code == 200
    last = changed.json()["trace"][-1]
    assert last["setpoint"] == [24, 24, 24] and last["people"] == [4, 0, 0]
    assert last["daylight"] == [0, 0, 0]
    assert changed.json()["totals"]["energy_kwh"] >= first.json()["totals"]["energy_kwh"]


def test_live_errors_are_bounded():
    assert client.post("/api/start", json={"controller": "../../private"}).status_code == 404
    assert client.post("/api/start", json={"stress": "undefined"}).status_code == 422
    assert client.post("/api/advance", json={"session": "none"}).status_code == 404
    start = client.post("/api/start", json={"controller": "balance"}).json()
    assert client.post("/api/advance", json={"session": start["session"], "steps": 2000}).status_code == 422
    assert client.post("/api/advance", json={"session": start["session"], "people": [100, 0, 0]}).status_code == 422
    assert client.get("/api/download/checkpoint/unknown").status_code == 404


def test_saved_neural_actions_and_brain_inspection():
    status = client.get("/api/status").json()
    if not status["runs"]:
        return
    ident = status["runs"][0]["id"]
    info = client.get(f"/api/brain/{ident}")
    assert info.status_code == 200
    assert len(info.json()["nodes"]) == info.json()["stats"]["width"]
    session = client.post("/api/start", json={"controller": ident}).json()["session"]
    result = client.post("/api/advance", json={"session": session, "steps": 2, "intervention": "hub"})
    assert result.status_code == 200
    assert all(v == 0 for v in result.json()["trace"][-1]["hub"])
    checkpoint = client.get(f"/api/download/checkpoint/{ident}")
    assert checkpoint.status_code == 200 and len(checkpoint.content) > 10000


def test_finite_48_hour_session_and_finished_readback():
    ident = client.post("/api/start", json={"controller": "off"}).json()["session"]
    for _ in range(24):
        r = client.post("/api/advance", json={"session": ident, "steps": 12})
        assert r.status_code == 200
    assert r.json()["done"]
    assert r.json()["trace"][-1]["hour"] == 48
    after = client.post("/api/advance", json={"session": ident, "steps": 1}).json()
    assert after["done"] and after["trace"] == []
