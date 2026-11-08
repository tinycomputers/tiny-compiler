"""Builds the full soundtrack for the TinyCompiler intro video.

Inputs : data/s1..s9.mp3 (+ .json word timings) from the voiceover, data/events.json
Outputs: public/audio/mix.mp3, src/timing.json (scene frames + word timings)

The music bed and every SFX are synthesised here (numpy only), so nothing is downloaded
and the result is deterministic. The voice ducks the music with a smooth side-chain.
Run: python3 scripts/build_audio.py   (needs numpy + ffmpeg; also `npm run audio`)
"""
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SR = 44100
FPS = 30
LEAD, TAIL = 0.15, 0.30  # seconds of air before / after each voice line
SCENES = 9

rng = np.random.default_rng(42)


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


# ---------------------------------------------------------------- timing
vo, scenes, words = [], [], []
cursor = 0
for i in range(1, SCENES + 1):
    a = decode(ROOT / "data" / f"s{i}.mp3")
    dur = len(a) / SR
    frames = math.ceil((LEAD + dur + TAIL) * FPS)
    scenes.append({"id": i, "start": cursor, "frames": frames, "voStart": LEAD, "voDur": dur})
    words.append([{**w, "s": w["s"] + LEAD} for w in json.load(open(ROOT / "data" / f"s{i}.json"))])
    vo.append(a)
    cursor += frames
TOTAL = cursor
def norm(w):
    return re.sub(r"[^a-z0-9'\-]", "", w.lower())


def resolve(spec):
    """Anchor -> scene-relative seconds, from the voice's word timings."""
    sc = spec["scene"]
    hits = [w for w in words[sc - 1] if norm(w["t"]) == norm(spec["word"])]
    if len(hits) < spec.get("n", 1):
        raise SystemExit(f"anchor word {spec['word']!r} (n={spec.get('n', 1)}) not found in scene {sc}: "
                         f"{[w['t'] for w in words[sc - 1]]}")
    return hits[spec.get("n", 1) - 1]["s"] + spec.get("offset", 0.0)


raw_events = {k: v for k, v in json.load(open(ROOT / "data" / "events.json")).items() if not k.startswith("_")}
events = {k: (v["scene"], resolve(v), v["sfx"]) for k, v in raw_events.items()}
(ROOT / "src" / "timing.json").write_text(
    json.dumps({"fps": FPS, "total": TOTAL, "scenes": scenes, "words": words,
                "events": {k: {"scene": s, "t": t} for k, (s, t, _) in events.items()}}, indent=1)
)
N = int(TOTAL / FPS * SR) + SR  # +1s tail
t_of_scene = lambda i: scenes[i - 1]["start"] / FPS  # noqa: E731


def mix_in(dst, src, at_s, gain=1.0):
    i = int(at_s * SR)
    if i < 0:
        src, i = src[-i:], 0
    if i >= len(dst):
        return
    j = min(len(dst), i + len(src))
    dst[i:j] += src[: j - i] * gain


# ---------------------------------------------------------------- voice track
voice = np.zeros(N, dtype=np.float32)
for i, a in enumerate(vo, start=1):
    mix_in(voice, a, t_of_scene(i) + LEAD)

# ---------------------------------------------------------------- synthesis helpers
def tt(d):
    return np.arange(int(d * SR)) / SR


def env(n, a, r, curve=3.0):
    e = np.ones(n)
    na, nr = int(a * SR), int(r * SR)
    if na:
        e[:na] = np.linspace(0, 1, na) ** 2
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr) ** curve
    return e


def onepole(x, cutoff):  # cutoff: array (Hz) per sample
    y = np.zeros_like(x)
    acc = 0.0
    k = 1 - np.exp(-2 * np.pi * cutoff / SR)
    for n in range(len(x)):
        acc += k[n] * (x[n] - acc)
        y[n] = acc
    return y


def whoosh(d=0.7, up=True):
    t = tt(d)
    n = rng.standard_normal(len(t))
    sweep = np.linspace(400, 5200, len(t)) if up else np.linspace(5200, 400, len(t))
    y = onepole(n, sweep)
    y = y - onepole(y, np.full(len(t), 250.0))  # high-pass-ish
    shape = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 2
    return (y * shape * 0.9).astype(np.float32)


def tick():
    t = tt(0.06)
    return (np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 90) * 0.55).astype(np.float32)


def click():
    t = tt(0.09)
    noise = rng.standard_normal(len(t)) * np.exp(-t * 140) * 0.25
    return (np.sin(2 * np.pi * 1500 * t) * np.exp(-t * 60) * 0.5 + noise).astype(np.float32)


def pop():
    t = tt(0.16)
    f = 520 * np.exp(-t * 11) + 140
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 20) * 0.75).astype(np.float32)


def chime():
    t = tt(1.1)
    y = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * d)
            for f, a, d in [(988, 0.5, 3.2), (1480, 0.32, 4.2), (1976, 0.16, 6.0), (2960, 0.07, 8.0)])
    return (y * 0.55).astype(np.float32)


def shimmer():
    t = tt(1.6)
    y = sum(np.sin(2 * np.pi * f * t) * 0.18 for f in (523.25, 659.25, 783.99, 1046.5, 1318.5))
    return (y * env(len(t), 0.25, 1.2, 2) * 0.55).astype(np.float32)


def thump():
    t = tt(0.9)
    f = 90 * np.exp(-t * 6) + 38
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5.5)
    return (y * 0.9 + chime()[: len(t)] * 0.0).astype(np.float32)


KINDS = {"tick": tick, "click": click, "pop": pop, "chime": chime, "shimmer": shimmer, "thump": thump}

sfx = np.zeros(N, dtype=np.float32)
for i in range(1, SCENES + 1):  # a whoosh into every scene cut
    if i > 1:
        mix_in(sfx, whoosh(0.75), t_of_scene(i) - 0.45, 0.42)
mix_in(sfx, whoosh(1.4, up=True), t_of_scene(9) - 1.0, 0.0)
for name, (scene, sec, kind) in events.items():
    if kind:
        g = {"tick": 0.35, "click": 0.5, "pop": 0.6, "chime": 0.38, "shimmer": 0.3, "thump": 0.8}[kind]
        mix_in(sfx, KINDS[kind](), t_of_scene(scene) + sec - 0.02, g)

# ---------------------------------------------------------------- music bed (A minor, 100 BPM)
BPM = 100
beat = 60 / BPM
bar = 4 * beat


def midi(m):
    return 440 * 2 ** ((m - 69) / 12)


CHORDS = [  # (bass, [pad notes], [arp notes])
    (45, [57, 60, 64, 67], [69, 72, 76, 79]),   # Am7
    (41, [53, 57, 60, 64], [65, 69, 72, 76]),   # Fmaj7
    (48, [55, 60, 64, 67], [67, 72, 76, 79]),   # C
    (43, [55, 59, 62, 66], [67, 71, 74, 79]),   # G6
]
music = np.zeros(N, dtype=np.float32)
total_s = N / SR
n_bars = int(total_s / bar) + 1


def pad_note(freq, d):
    t = tt(d)
    y = np.zeros(len(t))
    for det in (-0.004, 0.0, 0.004):
        for h, a in ((1, 1.0), (2, 0.45), (3, 0.2), (4, 0.1)):
            y += a * np.sin(2 * np.pi * freq * (1 + det) * h * t + rng.uniform(0, 6.28))
    return y * env(len(t), 0.9, 1.0, 1.5) / 12


def pluck(freq, d=0.7):
    t = tt(d)
    y = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * 2 * freq * t) * np.exp(-t * 9)
    return y * np.exp(-t * 6.5) * 0.5


def kick():
    t = tt(0.4)
    f = 110 * np.exp(-t * 28) + 44
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 11)


def hat():
    t = tt(0.05)
    return rng.standard_normal(len(t)) * np.exp(-t * 90) * 0.3


for b in range(n_bars):
    t0 = b * bar
    bass, pad, arp = CHORDS[(b // 2) % 4]
    for note in pad:
        mix_in(music, pad_note(midi(note), bar * 1.25).astype(np.float32), t0, 1.0)
    # sub bass: one soft note per bar
    tb = tt(bar * 0.95)
    mix_in(music, (np.sin(2 * np.pi * midi(bass - 12) * tb) * env(len(tb), 0.02, 0.5, 1.5) * 0.22).astype(np.float32), t0)
    if t0 >= 4 * bar - 0.01:  # arp enters after the first 4 bars
        pattern = [0, 2, 1, 3, 2, 3, 1, 2]
        for k, idx in enumerate(pattern):
            mix_in(music, pluck(midi(arp[idx])).astype(np.float32), t0 + k * beat / 2, 0.16)
    if t0 >= 6 * bar:  # gentle pulse once the demo starts
        for k in range(4):
            mix_in(music, kick().astype(np.float32), t0 + k * beat, 0.34)
            mix_in(music, hat().astype(np.float32), t0 + k * beat + beat / 2, 0.20)

# echo on the whole bed (dotted-eighth), a touch of width-free "space"
echo = np.zeros_like(music)
for k, g in enumerate((0.32, 0.16, 0.08), start=1):
    d = int(0.75 * beat * k * SR)
    echo[d:] += music[:-d] * g
music = music + echo * 0.5

# overall arc: swell in, gentle lift into the CTA, fade out at the end
tl = np.arange(N) / SR
arc = np.interp(tl, [0, 3, t_of_scene(7), t_of_scene(9), total_s - 2.2, total_s - 0.3], [0.0, 0.8, 0.95, 1.0, 0.9, 0.0])
music *= arc.astype(np.float32)

# side-chain duck under the voice: smoothed voice envelope -> gain reduction
w = int(0.18 * SR)
venv = np.convolve(np.abs(voice), np.ones(w) / w, mode="same")
venv = np.convolve(venv, np.ones(w) / w, mode="same")
duck = 1 - 0.6 * np.clip(venv / 0.05, 0, 1)
music *= duck.astype(np.float32)

# ---------------------------------------------------------------- final mix
def rms_norm(x, target):
    r = np.sqrt(np.mean(x[np.abs(x) > 1e-4] ** 2) + 1e-12)
    return x * (target / r)


voice_n = rms_norm(voice, 0.13)
music_n = rms_norm(music, 0.040)
sfx_n = sfx * 0.9
mix = voice_n + music_n + sfx_n
peak = np.max(np.abs(mix))
mix = mix / max(peak, 1e-6) * 0.89
mix = mix[: int(TOTAL / FPS * SR)]
mix[-int(0.25 * SR):] *= np.linspace(1, 0, int(0.25 * SR))

pcm = (mix * 32767).astype("<i2").tobytes()
out = ROOT / "public" / "audio"
out.mkdir(parents=True, exist_ok=True)
subprocess.run(
    ["ffmpeg", "-y", "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
     "-af", "pan=stereo|c0=c0|c1=c0", "-c:a", "libmp3lame", "-b:a", "192k", str(out / "mix.mp3")],
    input=pcm, check=True,
)
print(f"total {TOTAL} frames = {TOTAL / FPS:.1f}s; scenes:", [(s['id'], s['start'], s['frames']) for s in scenes])
