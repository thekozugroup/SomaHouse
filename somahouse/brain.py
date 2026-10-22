"""Engineered hub/spokes with traceable fly-derived private recurrent masks."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .sim import FEATURES

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("thermal", "air", "light", "context")


def graph(width=32, topology="fly", seed=0):
    counts = np.load(ROOT / "data/connectome/seed.npz")["counts"][:width, :width].copy()
    nodes = json.loads((ROOT / "data/connectome/provenance.json").read_text())["nodes"][:width]
    original = counts.copy()
    swaps = 0
    if topology == "rewired":
        rng = np.random.default_rng(seed)
        src, dst = np.nonzero(counts)
        values = counts[src, dst].copy()
        pairs = set(zip(src.tolist(), dst.tolist()))
        for _ in range(20 * len(src)):
            i, j = rng.integers(len(src), size=2)
            u, v, x, y = int(src[i]), int(dst[i]), int(src[j]), int(dst[j])
            if u == x or v == y or u == y or x == v or (u, y) in pairs or (x, v) in pairs:
                continue
            pairs.remove((u, v)); pairs.remove((x, y))
            pairs.add((u, y)); pairs.add((x, v))
            dst[i], dst[j] = y, v
            swaps += 1
        counts.fill(0)
        counts[src, dst] = values
    elif topology == "dense":
        counts = np.ones_like(counts) - np.eye(width, dtype=np.float32)
    elif topology != "fly":
        raise ValueError("Unknown graph topology")
    signs = np.array([-1 if n["neurotransmitter"] in ("gaba", "GABA", "glutamate") else 1 for n in nodes])
    w = np.log1p(counts) * signs[:, None]
    w /= np.maximum(np.abs(w).sum(0, keepdims=True), 1)
    return w.astype(np.float32), nodes, {"accepted_swaps": swaps,
             "different_edges": int(np.sum((counts > 0) != (original > 0)))}


def sensor_masks():
    mask = torch.zeros(4, FEATURES)
    # Identical widths, different observation streams. Hub is the only cross-domain path.
    groups = [list(range(0, 3)) + list(range(9, 15)) + [18, 19, 20, 30] + list(range(21, 24)),
              list(range(3, 6)) + list(range(9, 12)) + list(range(24, 27)),
              list(range(6, 12)) + list(range(15, 18)) + list(range(27, 30)),
              list(range(9, 15)) + [18, 19, 20, 30]]
    for i, indices in enumerate(groups):
        mask[i, indices] = 1
    return mask


class HouseBrain(nn.Module):
    def __init__(self, width=32, seed=7, topology="fly", hub=True, copies=1):
        super().__init__()
        torch.manual_seed(seed)
        w, nodes, graph_info = graph(width, topology, seed)
        self.config = {"width": width, "seed": seed, "topology": topology, "hub": hub, "copies": copies}
        self.nodes = nodes
        self.graph_info = graph_info
        self.lineage = []
        m = 4 * copies
        self.register_buffer("sensor_mask", sensor_masks().repeat(copies, 1))
        self.register_buffer("edge_mask", torch.from_numpy(w != 0).repeat(m, 1, 1))
        self.register_buffer("cell_mask", torch.ones(m, width))
        self.recurrent = nn.Parameter(torch.from_numpy(w).repeat(m, 1, 1))
        self.input = nn.Parameter(torch.randn(m, FEATURES, width) * .14)
        self.bias = nn.Parameter(torch.zeros(m, width))
        self.send = nn.Parameter(torch.randn(m, width, 8) * .12)
        self.hub_weight = nn.Parameter(torch.randn(4 * 8, 16) * .12)
        self.hub_bias = nn.Parameter(torch.zeros(16))
        self.receive = nn.Parameter(torch.randn(m, 16, width) * .12)
        self.readout = nn.Parameter(torch.randn(m, width, 3) * .025)
        self.output_bias = nn.Parameter(torch.tensor([0., 0., 0., -1.2, -1.2, -1.2, -.8, -.8, -.8]))

    def forward(self, obs, *, intervention=None, return_activity=False):
        m, n = self.cell_mask.shape
        gate = self.cell_mask
        if intervention and intervention.startswith("class:"):
            wanted = intervention.split(":", 1)[1]
            keep = torch.tensor([node["class"] != wanted for node in self.nodes], dtype=obs.dtype)
            gate = gate * keep[None, :]
        drive = torch.einsum("bf,mf,mfn->bmn", obs, self.sensor_mask, self.input) + self.bias
        h = torch.tanh(drive) * gate
        w = self.recurrent * self.edge_mask * gate[:, :, None] * gate[:, None, :]
        w = w / w.abs().sum(1, keepdim=True).clamp(min=1)
        if intervention == "recurrence":
            w = torch.zeros_like(w)
        hub = torch.zeros(len(obs), 16, dtype=obs.dtype, device=obs.device)
        for _ in range(2):
            ports = torch.tanh(torch.einsum("bmn,mnr->bmr", h, self.send))
            # Copies are averaged inside each functional domain; the hub retains four ports.
            pooled = ports.reshape(len(obs), self.config["copies"], 4, 8).mean(1).reshape(len(obs), 32)
            hub = torch.tanh(pooled @ self.hub_weight + self.hub_bias)
            if not self.config["hub"] or intervention == "hub":
                hub = torch.zeros_like(hub)
            incoming = torch.einsum("br,mrn->bmn", hub, self.receive)
            h = torch.tanh(drive + .65 * torch.einsum("bmn,mnk->bmk", h, w) + .35 * incoming) * gate
            if intervention in DOMAINS:
                domain = DOMAINS.index(intervention)
                lesion = torch.ones(m, dtype=obs.dtype, device=obs.device)
                lesion[domain::4] = 0
                h = h * lesion[None, :, None]
        read = torch.einsum("bmn,mna->bma", h, self.readout)
        read = read.reshape(len(obs), self.config["copies"], 4, 3).sum(1)
        logits = torch.cat((read[:, 0], read[:, 1], read[:, 2]), dim=1) + self.output_bias
        action = torch.cat((torch.tanh(logits[:, :3]), torch.sigmoid(logits[:, 3:])), dim=1)
        if return_activity:
            return action, {"spokes": h, "hub": hub, "ports": ports}
        return action

    def stats(self):
        gate = self.cell_mask.bool()
        active_edges = self.edge_mask & gate[:, :, None] & gate[:, None, :]
        allocated = sum(p.numel() for p in self.parameters())
        masked = self.recurrent.numel() - active_edges.sum().item()
        return {**self.config, "source_cells_per_spoke": int(gate[0].sum()),
                "spoke_units": int(gate.sum()), "hub_units": 16 if self.config["hub"] else 0,
                "source_unique_ids": len({n["body_id"] for i, n in enumerate(self.nodes) if gate[0, i]}),
                "recurrent_edges": int(active_edges.sum()), "allocated_parameters": allocated,
                "edge_adjusted_parameters": int(allocated - masked), "modules": 4 * self.config["copies"],
                "graph": self.graph_info, "lineage": self.lineage}

    def save(self, path):
        torch.save({"config": self.config, "nodes": self.nodes, "graph_info": self.graph_info,
                    "lineage": self.lineage, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        result = cls(**data["config"])
        result.nodes, result.lineage, result.graph_info = data["nodes"], data["lineage"], data["graph_info"]
        result.load_state_dict(data["state"])
        return result


def surgery(parent: HouseBrain, operation: str, seed: int):
    if operation == "duplicate":
        cfg = {**parent.config, "copies": parent.config["copies"] * 2}
        child = HouseBrain(**cfg)
        state = parent.state_dict()
        for name in ("sensor_mask", "edge_mask", "cell_mask", "recurrent", "input", "bias", "send", "receive", "readout"):
            reps = [2] + [1] * (state[name].ndim - 1)
            state[name] = state[name].repeat(*reps)
            if name == "readout":
                state[name] = state[name] / 2
        child.load_state_dict(state)
        # Zero perturbation at surgery preserves the initial action exactly. Independent
        # gradient noise is added after the first checkpoint, inside the training routine.
    elif operation in ("cut", "continue"):
        child = copy.deepcopy(parent)
        if operation == "cut":
            importance = (parent.input.detach().abs().mean(1) + parent.readout.detach().abs().mean(2)
                          + parent.send.detach().abs().mean(2) + parent.receive.detach().abs().mean(1)).mean(0)
            kept = torch.topk(importance, parent.config["width"] // 2).indices.sort().values
            child = HouseBrain(**{**parent.config, "width": len(kept)})
            state = parent.state_dict()
            for name in ("recurrent", "edge_mask"):
                state[name] = state[name][:, kept][:, :, kept]
            for name in ("input", "receive"):
                state[name] = state[name][:, :, kept]
            for name in ("bias", "cell_mask"):
                state[name] = state[name][:, kept]
            for name in ("send", "readout"):
                state[name] = state[name][:, kept]
            child.load_state_dict(state)
            child.nodes = [copy.deepcopy(parent.nodes[int(i)]) for i in kept]
            child.graph_info = {"selection": "learned weight salience", "parent_indices": kept.tolist()}
    else:
        raise ValueError("Unknown operation")
    child.lineage = copy.deepcopy(parent.lineage) + [{"operation": operation, "seed": seed,
                    "parent_spoke_units": parent.stats()["spoke_units"],
                    "child_spoke_units": child.stats()["spoke_units"]}]
    return child
