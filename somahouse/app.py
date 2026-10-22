"""Local experiment console. Live sessions do not retrain or modify study artifacts."""
import argparse
import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .brain import DOMAINS, ROOT, HouseBrain
from .sim import DT, ROOMS, baseline, observe, scenarios, transition
from .teacher import Teacher

torch.set_num_threads(2)
app = FastAPI(title="SomaHouse")
sessions = {}
session_lock = threading.Lock()
model_cache = {}
STATIC = ROOT / "somahouse/static"
REPORTS = ROOT / "reports"


def json_file(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def catalog():
    original = [json_file(p) for p in sorted((REPORTS / "training").glob("*.json"))]
    extension = json_file(REPORTS / "teacher/selection.json", {}).get("runs", [])
    return original + extension


def get_brain(ident):
    if ident not in {r["id"] for r in catalog()}:
        raise HTTPException(404, "Unknown trained checkpoint")
    if ident not in model_cache:
        run = next(r for r in catalog() if r["id"] == ident)
        path = run.get("checkpoint", f"checkpoints/{ident}.pt")
        model_cache[ident] = HouseBrain.load(ROOT / path).eval()
    return model_cache[ident]


def selected_id():
    extension = json_file(REPORTS / "teacher/results.json", {})
    if extension:
        return f"{extension['primary']}-s{extension['representative_seed']}"
    result = json_file(REPORTS / "results.json", {})
    selection = json_file(REPORTS / "selection.json", {})
    if selection:
        return f"{selection['primary_family']}-s{result.get('representative_seed', 19)}"
    runs = catalog()
    if not runs: raise HTTPException(409, "Training is still producing its first checkpoint")
    return runs[0]["id"]


@app.get("/api/status")
def status():
    selection = json_file(REPORTS / "selection.json", {})
    results = json_file(REPORTS / "results.json", {})
    return {"name": "SomaHouse", "runs": catalog(), "selection": selection,
            "results": results, "ready": bool(results), "rooms": ROOMS,
            "live_learning": False, "simulation_minutes_per_step": 10,
            "teacher_study": json_file(REPORTS / "teacher/results.json", {}),
            "default_controller": selected_id() if catalog() else None}


class Start(BaseModel):
    seed: int = Field(572, ge=0, le=9_999_999)
    controller: str = "selected"
    stress: str = "normal"
    intervention: str | None = None


def check_intervention(value):
    if value is not None and value not in ("hub", "recurrence", *DOMAINS):
        raise HTTPException(422, "Unknown intervention")


@app.post("/api/start")
@torch.no_grad()
def start(req: Start):
    check_intervention(req.intervention)
    name = req.controller
    if name == "selected":
        name = selected_id()
    policy = name if name in ("feedback", "balance", "off") else get_brain(name) if name != "teacher" else None
    teacher_config = json_file(REPORTS / "teacher/selection.json", {}).get("teacher", {}).get("config")
    if name == "teacher" and teacher_config is None: raise HTTPException(409, "Teacher study is not ready")
    try:
        env = scenarios([req.seed], 288, stress=req.stress)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    session = {"env": env, "state": env.initial, "time": 0, "policy": policy,
               "teacher_config": teacher_config if name == "teacher" else None,
               "controller": name, "intervention": req.intervention, "totals": {}, "updated": time.time()}
    ident = uuid.uuid4().hex
    with session_lock:
        for key in list(sessions):
            if time.time() - sessions[key]["updated"] > 3600:
                del sessions[key]
        if len(sessions) >= 32:
            del sessions[min(sessions, key=lambda k: sessions[k]["updated"])]
        sessions[ident] = session
    return {"session": ident, "controller": name, "seed": req.seed, "duration_hours": 48,
            "initial": {"temperature": env.initial.temperature[0].tolist(),
                        "co2": env.initial.co2[0].tolist(), "setpoint": env.setpoint[0, 0].tolist()},
            "stats": policy.stats() if isinstance(policy, HouseBrain) else None}


class Advance(BaseModel):
    session: str
    steps: int = Field(1, ge=1, le=12)
    setpoint: list[float] | None = None
    people: list[float] | None = None
    outside_offset: float | None = Field(None, ge=-15, le=15)
    daylight_factor: float | None = Field(None, ge=0, le=1.5)
    intervention: str | None = None


@app.post("/api/advance")
@torch.no_grad()
def advance(req: Advance):
    check_intervention(req.intervention)
    with session_lock:
        if req.session not in sessions:
            raise HTTPException(404, "Session expired; reset the house")
        session = sessions[req.session]
        env, t = session["env"], session["time"]
        for value, name, low, high in ((req.setpoint, "setpoint", 18, 26), (req.people, "people", 0, 4)):
            if value is not None:
                if len(value) != 3 or any(not np.isfinite(v) or v < low or v > high for v in value):
                    raise HTTPException(422, f"{name}: provide three values between {low} and {high}")
                getattr(env, name)[t:, 0] = torch.tensor(value)
        # Offsets are absolute relative to the original exogenous signal, not compounded.
        if "original_outside" not in session:
            session["original_outside"] = env.outside.clone()
            session["original_daylight"] = env.daylight.clone()
        if req.outside_offset is not None:
            env.outside[t:] = session["original_outside"][t:] + req.outside_offset
        if req.daylight_factor is not None:
            env.daylight[t:] = session["original_daylight"][t:] * req.daylight_factor
        session["intervention"] = req.intervention
        trace = []
        for t in range(t, min(t + req.steps, env.steps)):
            obs = observe(session["state"], env, t)
            policy = session["policy"]
            activity = None
            if session.get("teacher_config") is not None:
                action = Teacher(session["teacher_config"])(obs)
            elif isinstance(policy, str):
                action = baseline(obs, policy)
            else:
                action, activity = policy(obs, intervention=session["intervention"], return_activity=True)
            state, metrics = transition(session["state"], action, env, t)
            session["state"] = state
            session["time"] = t + 1
            for k in ("energy_kwh", "thermal_degree_hours", "co2_ppm_hours", "light_lux_hours"):
                session["totals"][k] = session["totals"].get(k, 0) + metrics[k][0].item()
            trace.append({"hour": (t + 1) * DT, "outside": env.outside[t, 0].item(),
                          "temperature": state.temperature[0].tolist(), "co2": state.co2[0].tolist(),
                          "lux": metrics["lux"][0].tolist(), "people": env.people[t, 0].tolist(),
                          "setpoint": env.setpoint[t, 0].tolist(), "daylight": env.daylight[t, 0].tolist(),
                          "action": state.action[0].tolist(), "power_kw": metrics["peak_kw"][0].item(),
                          "hub": activity["hub"][0].tolist() if activity else [],
                          "spokes": activity["spokes"][0].mean(-1).tolist() if activity else []})
        session["updated"] = time.time()
        return {"trace": trace, "totals": session["totals"], "done": session["time"] >= env.steps,
                "intervention": session["intervention"]}


@app.get("/api/brain/{ident}")
@torch.no_grad()
def brain_info(ident: str):
    brain = get_brain(ident)
    modules = []
    for i in range(brain.stats()["modules"]):
        w = (brain.recurrent[i] * brain.edge_mask[i]).numpy()
        src, dst = np.nonzero(w)
        order = np.argsort(np.abs(w[src, dst]))[-240:][::-1]
        modules.append({"index": i, "domain": DOMAINS[i % 4], "copy": i // 4,
                        "weights": w.tolist(), "edge_count": len(src),
                        "displayed_edges": [{"source": int(src[j]), "target": int(dst[j]), "weight": float(w[src[j], dst[j]])} for j in order],
                        "input_norm": float(brain.input[i].norm()), "send_norm": float(brain.send[i].norm()),
                        "receive_norm": float(brain.receive[i].norm()), "readout_norm": float(brain.readout[i].norm())})
    return {"id": ident, "stats": brain.stats(), "nodes": brain.nodes, "modules": modules,
            "training": next(r for r in catalog() if r["id"] == ident),
            "hub_weights": brain.hub_weight.tolist(), "sensor_masks": brain.sensor_mask.tolist()}


@app.get("/api/download/{kind}/{name}")
def download(kind: str, name: str):
    if kind == "checkpoint" and name in {r["id"] for r in catalog()}:
        run = next(r for r in catalog() if r["id"] == name)
        return FileResponse(ROOT / run.get("checkpoint", f"checkpoints/{name}.pt"), filename=f"somahouse-{name}.pt")
    allowed = {"results.json", "selection.json", "episodes.csv", "verification.json", "report.md", "case-study.md", "study-bundle.zip", "preview.json"}
    if kind == "report" and name in allowed and (REPORTS / name).is_file():
        return FileResponse(REPORTS / name, filename=name)
    if kind == "protocol" and name == "protocol.md":
        return FileResponse(ROOT / "research/protocol.md", filename="somahouse-protocol.md")
    if kind == "teacher" and name in {"results.json", "selection.json", "episodes.csv", "verification.json", "report.md", "teacher-protocol.md"}:
        path = ROOT / "research/teacher-protocol.md" if name == "teacher-protocol.md" else REPORTS / "teacher" / name
        if path.is_file(): return FileResponse(path, filename=f"somahouse-teacher-{name}")
    raise HTTPException(404, "Artifact not found")


@app.get("/")
def index(): return FileResponse(STATIC / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon(): return Response(status_code=204)


@app.get("/brain.html")
def inspector(): return FileResponse(STATIC / "brain.html")


@app.get("/case-study.html")
def case_study(): return FileResponse(STATIC / "case-study.html")


app.mount("/assets", StaticFiles(directory=STATIC), name="assets")
app.mount("/figures", StaticFiles(directory=REPORTS), name="figures")


def main():
    import uvicorn
    parser = argparse.ArgumentParser(); parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__": main()
