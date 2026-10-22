"""Reduced-order three-zone house. Units: hours, kW, kWh/K, degC, ppm, lux.

All controller actions pass through the same actuator map. No thermostat overlay.
The model is deliberately transparent and is not a calibrated building emulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

DT = 1 / 6
ROOMS = ("Living", "Bedroom", "Office")
FEATURES = 31
OBS_NAMES = ([f"temperature_{r}" for r in ROOMS] + [f"co2_{r}" for r in ROOMS]
             + [f"lux_{r}" for r in ROOMS] + [f"people_{r}" for r in ROOMS]
             + [f"setpoint_{r}" for r in ROOMS] + [f"daylight_{r}" for r in ROOMS]
             + ["outdoor_temperature", "time_sin", "time_cos"]
             + [f"previous_{a}_{r}" for a in ("heat", "vent", "light") for r in ROOMS]
             + ["electricity_price"])
COST_WEIGHTS = {"thermal_slack_squared": 4.0, "thermal_tracking_squared": 0.05,
                "co2_excess_500ppm_squared": 2.0, "light_deficit_300lux_squared": 4.0,
                "power_kw": 0.08, "action_movement": 0.012}


class State(NamedTuple):
    temperature: torch.Tensor
    co2: torch.Tensor
    action: torch.Tensor


@dataclass
class Scenario:
    seeds: list[int]
    outside: torch.Tensor
    daylight: torch.Tensor
    people: torch.Tensor
    setpoint: torch.Tensor
    hour: torch.Tensor
    price: torch.Tensor
    capacity: torch.Tensor
    conductance: torch.Tensor
    volume: torch.Tensor
    initial: State
    stress: str = "normal"
    actuator_factor: float = 1.0

    @property
    def steps(self):
        return self.outside.shape[0] - 1


def scenarios(seeds: list[int], steps: int = 288, *, training: bool = False,
              stress: str = "normal") -> Scenario:
    """Every episode has an independent RNG; batch composition changes nothing."""
    if stress not in ("normal", "weather", "temperature_bias", "co2_dropout", "capacity"):
        raise ValueError("Unknown stress scenario")
    arrays = {k: [] for k in ("outside", "daylight", "people", "setpoint", "hour", "price",
                             "capacity", "conductance", "volume", "temperature", "co2")}
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        start = rng.uniform(0, 48) if training else 0.0
        hour_abs = start + np.arange(steps + 1) * DT
        hour = hour_abs % 24
        season = int(rng.integers(0, 3))
        mean = (3.0, 17.0, 31.0)[season] + rng.uniform(-3, 3)
        if stress == "weather":
            mean += -8 if season == 0 else 8
        outside = mean + rng.uniform(3, 7) * np.sin((hour - 9) * np.pi / 12)
        outside += np.interp(hour_abs, np.arange(0, hour_abs[-1] + 3, 2),
                             rng.normal(0, 1.1, len(np.arange(0, hour_abs[-1] + 3, 2))))
        sun = np.maximum(0, np.sin((hour - 6) * np.pi / 12))
        cloud = .55 + .4 * np.sin(hour_abs * rng.uniform(.3, .8) + rng.uniform(0, 6)) ** 2
        daylight = sun[:, None] * cloud[:, None] * rng.uniform(380, 1000, (1, 3))
        people = np.zeros((steps + 1, 3), np.float32)
        shift = rng.uniform(-.7, .7)
        h = (hour + shift) % 24
        people[:, 0] = ((h >= 6.5) & (h < 8.5)) + 2 * ((h >= 17.5) & (h < 23))
        people[:, 1] = 2 * ((h < 7) | (h >= 23))
        people[:, 2] = ((h >= 8.5) & (h < 17.5)) * rng.integers(1, 3)
        visit = rng.uniform(11, 18)
        people[:, 0] += ((hour >= visit) & (hour < visit + rng.uniform(1, 3))) * rng.integers(0, 3)
        # A day's setpoint can change without the controller seeing a forecast.
        setpoint = np.broadcast_to(rng.uniform(20.5, 23.5, (1, 3)), (steps + 1, 3)).copy()
        setpoint += (hour[:, None] > rng.uniform(12, 19)) * rng.uniform(-1.5, 1.5, (1, 3))
        price = np.where((hour >= 16) & (hour < 21), 1.6, 1.0)
        arrays["outside"].append(outside)
        arrays["daylight"].append(daylight)
        arrays["people"].append(people)
        arrays["setpoint"].append(setpoint)
        arrays["hour"].append(hour)
        arrays["price"].append(price)
        arrays["capacity"].append(rng.uniform(1.3, 2.2, 3))
        arrays["conductance"].append(rng.uniform(.055, .095, 3))
        arrays["volume"].append(rng.uniform(65, 100, 3))
        arrays["temperature"].append(setpoint[0] + rng.uniform(-2.5, 2.5, 3))
        arrays["co2"].append(rng.uniform(450, 1100, 3))
    data = {}
    for k, v in arrays.items():
        a = np.asarray(v, np.float32)
        if k in ("outside", "daylight", "people", "setpoint", "hour", "price"):
            a = a.swapaxes(0, 1)
        data[k] = torch.from_numpy(a)
    initial = State(data.pop("temperature"), data.pop("co2"), torch.zeros(len(seeds), 9))
    return Scenario(seeds=list(map(int, seeds)), initial=initial, stress=stress,
                    actuator_factor=.45 if stress == "capacity" else 1.0, **data)


def observe(s: State, env: Scenario, t: int) -> torch.Tensor:
    temp = s.temperature + (2.0 if env.stress == "temperature_bias" else 0.0)
    co2 = torch.full_like(s.co2, 650) if env.stress == "co2_dropout" else s.co2
    lux = env.daylight[t] + 450 * s.action[:, 6:9]
    return torch.cat(((temp - 22) / 8, (co2 - 420) / 1000, lux / 600,
                      env.people[t] / 3, (env.setpoint[t] - 22) / 4, env.daylight[t] / 600,
                      (env.outside[t, :, None] - 15) / 20,
                      torch.sin(env.hour[t, :, None] * np.pi / 12),
                      torch.cos(env.hour[t, :, None] * np.pi / 12), s.action,
                      env.price[t, :, None] / 1.6), dim=1)


def actuate(raw: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
    heat = raw[:, :3].clamp(-1, 1) * factor
    # A 6 kW shared thermal plant, versus 3 kW maximum in each zone.
    heat = heat * torch.clamp(2.0 * factor / (heat.abs().sum(1, keepdim=True) + 1e-8), max=1)
    vent = raw[:, 3:6].clamp(0, 1) * factor
    light = raw[:, 6:9].clamp(0, 1)
    return torch.cat((heat, vent, light), dim=1)


def transition(s: State, raw: torch.Tensor, env: Scenario, t: int):
    a = actuate(raw, env.actuator_factor)
    heat, vent, light = a[:, :3], a[:, 3:6], a[:, 6:9]
    ach = .12 + 3.4 * vent
    outside = env.outside[t, :, None]
    envelope = env.conductance * (outside - s.temperature)
    ventilation = .000335 * env.volume * ach * (outside - s.temperature)
    interzone = .025 * (s.temperature.mean(1, keepdim=True) - s.temperature)
    gains = .12 * env.people[t] + .00035 * env.daylight[t] + .07 * light
    temperature = s.temperature + DT * (envelope + ventilation + interzone + gains + 3 * heat) / env.capacity
    co2 = s.co2 + DT * (18000 / env.volume * env.people[t] - ach * (s.co2 - 420))
    lux = env.daylight[t] + 450 * light
    power = (3 * heat.abs() / 3.2 + .045 * vent ** 3 + .07 * light).sum(1)
    occ = (env.people[t] > 0).float()
    error = (temperature - env.setpoint[t]).abs()
    thermal = torch.relu(error - 1)
    excess = torch.relu(co2 - 1000)
    deficit = torch.relu(300 - lux)
    # Empty rooms may drift three degrees. Occupied rooms dominate the objective.
    empty_slack = torch.relu(error - 3)
    movement = (a - s.action).abs().mean(1)
    loss = (4 * (thermal ** 2 * occ + .1 * empty_slack ** 2 * (1 - occ))
            + .05 * error ** 2 * occ + 2 * (excess / 500) ** 2 * occ
            + 4 * (deficit / 300) ** 2 * occ).mean(1)
    loss = loss + .08 * power * env.price[t] + .012 * movement
    metrics = {"cost": loss, "energy_kwh": power * DT,
               "thermal_degree_hours": (thermal * occ).sum(1) * DT / 3,
               "co2_ppm_hours": (excess * occ).sum(1) * DT / 3,
               "light_lux_hours": (deficit * occ).sum(1) * DT / 3,
               "occupied_zone_hours": occ.sum(1) * DT,
               "thermal_compliant_hours": ((error <= 1).float() * occ).sum(1) * DT,
               "co2_compliant_hours": ((co2 <= 1000).float() * occ).sum(1) * DT,
               "light_compliant_hours": ((lux >= 295).float() * occ).sum(1) * DT,
               "action_movement": movement,
               "peak_kw": power, "lux": lux, "executed": a}
    return State(temperature, co2, a), metrics


def baseline(obs: torch.Tensor, kind: str = "feedback") -> torch.Tensor:
    temp = obs[:, :3] * 8 + 22
    co2 = obs[:, 3:6] * 1000 + 420
    people = obs[:, 9:12] * 3
    target = obs[:, 12:15] * 4 + 22
    daylight = obs[:, 15:18] * 600
    outdoor = obs[:, 18:19] * 20 + 15
    occ = (people > 0).float()
    if kind == "off":
        return torch.zeros(len(obs), 9)
    if kind == "feedback":
        heat = (target - temp) * .75
        heat *= .15 + .85 * occ
        vent = ((co2 - 750) / 600).clamp(0, 1)
    elif kind == "balance":
        vent = ((18000 / 82 * people / 530 - .12) / 3.4 + (co2 - 950) / 1200).clamp(0, 1)
        ach = .12 + 3.4 * vent
        gain = .12 * people + .00035 * daylight
        feed = -( .075 + .000335 * 82 * ach) * (outdoor - temp) - gain
        error = target - temp
        # Deliberately relaxed tracking in empty rooms, same current sensors.
        heat = (feed + 1.3 * error * (.12 + .88 * occ)) / 3
    else:
        raise ValueError("Unknown baseline")
    light = ((300 - daylight) / 450).clamp(0, 1) * occ
    return torch.cat((heat.clamp(-1, 1), vent, light), dim=1)


def rollout(policy, env: Scenario, *, trace: bool = False, intervention: str | None = None):
    state = env.initial
    totals: dict[str, torch.Tensor] = {}
    history = []
    for t in range(env.steps):
        obs = observe(state, env, t)
        action = baseline(obs, policy) if isinstance(policy, str) else policy(obs, intervention=intervention)
        state, m = transition(state, action, env, t)
        for k, v in m.items():
            if k in ("lux", "executed"):
                continue
            totals[k] = torch.maximum(totals.get(k, torch.zeros_like(v)), v) if k == "peak_kw" else totals.get(k, 0) + v
        if trace:
            history.append({"hour": round((t + 1) * DT, 4), "outside": env.outside[t, 0].item(),
                            "temperature": state.temperature[0].detach().tolist(),
                            "co2": state.co2[0].detach().tolist(), "lux": m["lux"][0].detach().tolist(),
                            "people": env.people[t, 0].tolist(), "daylight": env.daylight[t, 0].tolist(),
                            "setpoint": env.setpoint[t, 0].tolist(), "action": state.action[0].detach().tolist(),
                            "power_kw": m["peak_kw"][0].item()})
    totals["cost"] = totals["cost"] / env.steps
    return totals, history
