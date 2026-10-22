"""Render actual saved-model telemetry to an honestly time-scaled preview video."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .brain import DOMAINS, ROOT, HouseBrain
from .sim import ROOMS, observe, scenarios, transition
from .study import sha, write_json

STATIC = ROOT / "somahouse/static"
FONTS = STATIC / "fonts"


def font(size, display=False):
    return ImageFont.truetype(str(FONTS / ("display.ttf" if display else "body.ttf")), size)


def clock(step):
    minutes = (step + 1) * 10
    return f"Day {1 + (minutes - 1) // 1440} / {minutes // 60 % 24:02d}:{minutes % 60:02d}"


def events(trace):
    feed, frames = [], []
    previous = None
    for i, row in enumerate(trace):
        stamp = clock(i).split(" / ")[1]
        changes = []
        if previous is None:
            changes.append(f"Episode starts. Outside {row['outside']:.1f}°C.")
        else:
            for z, room in enumerate(ROOMS):
                if row["people"][z] != previous["people"][z]:
                    changes.append(f"{room}: occupancy {int(row['people'][z])}.")
                if abs(row["setpoint"][z] - previous["setpoint"][z]) > .01:
                    changes.append(f"{room}: target {row['setpoint'][z]:.1f}°C.")
                if abs(row["action"][6 + z] - previous["action"][6 + z]) >= .12:
                    changes.append(f"{room}: lights {row['action'][6+z]*100:.0f}% / {row['daylight'][z]:.0f} daylight lux.")
            if not changes and i % 6 == 0:
                z = int(np.argmax(np.abs(row["action"][:3])))
                changes.append(f"{ROOMS[z]}: {'cool' if row['action'][z] < 0 else 'heat'} {abs(row['action'][z])*100:.0f}%; {row['temperature'][z]:.1f}°C.")
        for message in changes:
            feed.append(f"{stamp}  {message}")
        frames.append(feed[-3:].copy())
        previous = row
    return frames


def render_frame(index, trace, activity, stats, model_name, feed):
    row = trace[index]
    im = Image.new("RGB", (1280, 720), "#fafafa")
    d = ImageDraw.Draw(im)

    def text(x, y, value, size=14, color="#262626", display=False):
        d.text((x, y), str(value), font=font(size, display), fill=color)

    def box(coords, fill="#ffffff", outline="#d4d4d4", width=1):
        d.rounded_rectangle(coords, radius=4, fill=fill, outline=outline, width=width)

    text(32, 20, "SomaHouse", 38, display=True)
    text(33, 70, "A house as a body. An actual saved-model test.", 16, "#666666")
    text(987, 24, clock(index), 24)
    text(987, 64, "48 simulated hours / 72 seconds", 13, "#666666")
    d.line((32, 108, 1248, 108), fill="#c7c7c7")
    text(32, 124, "01 / HOUSE + SENSORS", 12, "#666666")
    text(650, 124, "02 / BRAIN + MESSAGE HUB", 12, "#666666")
    text(32, 151, f"Outside {row['outside']:.1f}°C     Power {row['power_kw']:.2f} kW", 14)
    text(650, 151, f"{stats['spoke_units']} spoke units + 16 hub units / fixed learned weights", 13)
    rooms = [(32, 186, 312, 510), (328, 186, 612, 340), (328, 356, 612, 510)]
    for z, (left, top, right, bottom) in enumerate(rooms):
        delta = abs(row["temperature"][z] - row["setpoint"][z])
        shade = "#eeeeee" if delta <= 1 else "#dcdcdc"
        box((left, top, right, bottom), shade, "#a3a3a3", 2)
        text(left + 14, top + 11, ROOMS[z].upper(), 11)
        text(left + 14, top + 31, f"{row['temperature'][z]:.1f}°", 37 if z == 0 else 26, display=True)
        text(left + (126 if z == 0 else 112), top + 41, f"target {row['setpoint'][z]:.1f}°C", 12, "#606060")
        y = top + (89 if z == 0 else 70)
        text(left + 14, y, f"{row['co2'][z]:.0f} ppm CO₂  /  {row['lux'][z]:.0f} lux  /  {int(row['people'][z])} people", 12)
        # Furniture in the large room leaves sensor/actuator readings unobstructed.
        if z == 0:
            box((left + 20, top + 130, left + 137, top + 200), shade, "#bcbcbc")
            d.line((left + 25, top + 177, left + 132, top + 177), fill="#bcbcbc", width=2)
            d.ellipse((left + 169, top + 146, left + 228, top + 185), outline="#bcbcbc", width=2)
        y = bottom - 47
        for a, label in enumerate(("Cool" if row["action"][z] < 0 else "Heat", "Vent", "Light")):
            val = abs(row["action"][a * 3 + z])
            x = left + 14 + a * 86
            text(x, y, f"{label} {val*100:.0f}%", 10)
            d.rectangle((x, y + 21, x + 68, y + 25), fill="#cccccc")
            d.rectangle((x, y + 21, x + max(1, 68 * val), y + 25), fill="#333333")
        # Window illumination is measured daylight, mapped to grayscale.
        bright = int(115 + min(1, row["daylight"][z] / 900) * 120)
        d.line((left + 40, top, right - 40, top), fill=(bright,) * 3, width=6)

    center = (949, 339)
    centers = []
    for m in range(stats["modules"]):
        angle = -math.pi / 2 + m / stats["modules"] * math.tau
        centers.append((center[0] + 211 * math.cos(angle), center[1] + 125 * math.sin(angle)))
    for m, point in enumerate(centers):
        magnitude = float(np.mean(np.abs(activity["ports"][m])))
        gray = int(210 - min(1, magnitude) * 150)
        d.line((center, point), fill=(gray,) * 3, width=1 + round(magnitude * 3))
    for m, (x, y) in enumerate(centers):
        box((x - 62, y - 29, x + 62, y + 29), "#f1f1f1", "#bbbbbb")
        text(x - 52, y - 26, f"{DOMAINS[m%4].upper()} {m//4+1}", 10)
        cells = activity["spokes"][m]
        for cell, value in enumerate(cells):
            cx, cy = x - 49 + cell % 8 * 14, y - 4 + cell // 8 * 7
            gray = int(225 - min(1, abs(float(value))) * 205)
            d.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=(gray,) * 3)
    box((center[0] - 40, center[1] - 35, center[0] + 40, center[1] + 35), "#262626", "#262626")
    text(center[0] - 15, center[1] - 29, "HUB", 11, "#ffffff")
    for cell, value in enumerate(activity["hub"]):
        x, y = center[0] - 22 + cell % 4 * 14, center[1] - 6 + cell // 4 * 10
        gray = int(80 + min(1, abs(float(value))) * 175)
        d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(gray,) * 3)
    text(650, 493, "Dots: |cell activation|. Links: |message|. Two exchanges per decision.", 11, "#666666")
    d.line((32, 536, 1248, 536), fill="#cccccc")
    text(32, 552, "03 / STIMULUS + ACTION FEED", 12, "#666666")
    for i, message in enumerate(feed):
        text(32, 580 + i * 24, message, 14, "#262626" if i == len(feed) - 1 else "#777777")
    text(777, 556, "ACTUAL TELEMETRY / 2,400× SIMULATION SPEED", 11)
    text(777, 583, "10 simulated minutes per decision / 4 decisions per video second", 11, "#666666")
    text(777, 607, "Schematic positions. Activations are artificial, not biological recordings.", 11, "#666666")
    d.rectangle((32, 669, 1248, 671), fill="#dddddd")
    d.rectangle((32, 669, 32 + 1216 * (index + 1) / len(trace), 671), fill="#262626")
    text(32, 686, f"Decision {index+1:03d}/{len(trace)}  /  {model_name}", 11, "#666666")
    text(935, 686, "Simulation only. No learning during playback.", 11, "#666666")
    return im


@torch.no_grad()
def build(*, study=1):
    torch.set_num_threads(2)
    reports = ROOT / ("reports/teacher" if study == 2 else "reports")
    result = json.loads((reports / "results.json").read_text())
    family, seed = result["primary"], result["representative_seed"]
    checkpoint = ROOT / "checkpoints" / ("teacher" if study == 2 else "") / f"{family}-s{seed}.pt"
    recorded = json.loads((reports / f"trace-{family}.json").read_text())
    model = HouseBrain.load(checkpoint).eval()
    env = scenarios([recorded["scenario_seed"]], 288)
    state = env.initial; activities = []; maximum = 0.
    for t, row in enumerate(recorded["trace"]):
        action, activity = model(observe(state, env, t), return_activity=True)
        state, _ = transition(state, action, env, t)
        maximum = max(maximum, float(np.max(np.abs(state.action[0].numpy() - row["action"]))),
                      float(np.max(np.abs(state.temperature[0].numpy() - row["temperature"]))))
        activities.append({k: v[0].numpy() for k, v in activity.items()})
    assert maximum < 1e-6, maximum
    trace = recorded["trace"]; feed = events(trace)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg: raise RuntimeError("FFmpeg required to render the optional video")
    output = STATIC / "preview.mp4"
    command = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", "1280x720", "-framerate", "4", "-i", "pipe:0", "-vf", "fps=24",
               "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for i in range(len(trace)):
            im = render_frame(i, trace, activities[i], model.stats(), f"{family}-s{seed}", feed[i])
            if i == 72: im.save(STATIC / "preview-poster.png")
            proc.stdin.write(im.tobytes())
        proc.stdin.close()
        if proc.wait() != 0: raise RuntimeError("Video encoding failed")
    finally:
        if proc.poll() is None: proc.kill(); proc.wait()
    write_json(ROOT / "reports/preview.json", {"study": study, "controller": family, "training_seed": seed,
               "scenario_seed": recorded["scenario_seed"], "checkpoint": str(checkpoint.relative_to(ROOT)),
               "checkpoint_sha256": sha(checkpoint), "trace_sha256": sha(reports / f"trace-{family}.json"),
               "verified_decisions": len(trace), "maximum_action_temperature_replay_error": maximum,
               "simulated_hours": 48, "duration_seconds": 72, "simulation_speed": 2400,
               "video_fps": 24, "decision_fps": 4, "resolution": [1280, 720],
               "video_sha256": sha(output), "poster_sha256": sha(STATIC / "preview-poster.png"),
               "activity": "saved-model tanh cell/hub/port values; magnitude in grayscale; schematic positions",
               "online_learning": False})
    print(f"Rendered {output}: 288 decisions verified; max replay error {maximum}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--study", type=int, choices=(1, 2), default=1)
    build(study=parser.parse_args().study)
