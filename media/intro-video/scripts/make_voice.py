"""Sentence-level narration for the TinyCompiler video.

Reads data/narration.json, synthesises each sentence with edge-tts, joins sentences with
natural pauses, masters each scene (high-pass, gentle compression, loudness), and writes
data/s<N>.mp3 plus data/s<N>.json (word timings relative to the scene start, no lead-in).
build_audio.py then adds the lead-in and builds the mix.

Run: ../../.venv/bin/python scripts/make_voice.py   (needs the edge-tts venv in the scratchpad)
"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parent.parent
SR = 44100
spec = json.loads((ROOT / "data" / "narration.json").read_text())
only = set(sys.argv[1:])  # optional scene ids


async def speak(text: str):
    for attempt in range(6):
        try:
            c = edge_tts.Communicate(text, spec["voice"], rate=spec["rate"], pitch=spec["pitch"],
                                     boundary="WordBoundary")
            audio, words = bytearray(), []
            async for ch in c.stream():
                if ch["type"] == "audio":
                    audio += ch["data"]
                elif ch["type"] == "WordBoundary":
                    words.append({"t": ch["text"], "s": ch["offset"] / 1e7, "d": ch["duration"] / 1e7})
            if audio:
                return bytes(audio), words
        except Exception as e:  # network hiccups from the edge service
            print("retry:", e)
        await asyncio.sleep(2)
    raise SystemExit(f"could not synthesise: {text[:40]}")


def probe(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


async def build_scene(sid: str, sentences: list[str]):
    tmp = ROOT / "data" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    parts, words, cursor = [], [], 0.0
    pause = spec["sentence_pause"]
    for j, text in enumerate(sentences):
        raw, w = await speak(text)
        f = tmp / f"s{sid}_{j}.mp3"
        f.write_bytes(raw)
        d = probe(f)
        words += [{"t": x["t"], "s": cursor + x["s"], "d": x["d"]} for x in w]
        parts.append(f)
        cursor += d + (pause if j < len(sentences) - 1 else 0)
    # join: each sentence followed by silence (except the last)
    filt, ins = [], []
    for i, p in enumerate(parts):
        ins += ["-i", str(p)]
    n = len(parts)
    for i in range(n):
        filt.append(f"[{i}:a]aresample={SR},aformat=sample_fmts=fltp:channel_layouts=mono[a{i}]")
    chain = []
    for i in range(n):
        chain.append(f"[a{i}]")
        if i < n - 1:
            chain.append(f"anullsrc=r={SR}:cl=mono:d={pause}[p{i}];")
            chain.append(f"[p{i}]")
    join_inputs = []
    for i in range(n):
        join_inputs.append(f"[a{i}]")
        if i < n - 1:
            join_inputs.append(f"[p{i}]")
    # silence sources must be declared in the graph
    sil = "".join(f"anullsrc=r={SR}:cl=mono:d={pause}[p{i}];" for i in range(n - 1))
    graph = ";".join(filt) + ";" + sil + "".join(join_inputs) + f"concat=n={len(join_inputs)}:v=0:a=1,"
    graph += "highpass=f=85,acompressor=threshold=-20dB:ratio=2.2:attack=6:release=140:makeup=2,"
    graph += "equalizer=f=3200:t=q:w=1.2:g=1.5,loudnorm=I=-17:TP=-1.5:LRA=7[out]"
    out = ROOT / "data" / f"s{sid}.mp3"
    subprocess.run(["ffmpeg", "-y", "-v", "error", *ins, "-filter_complex", graph, "-map", "[out]",
                    "-c:a", "libmp3lame", "-b:a", "192k", str(out)], check=True)
    (ROOT / "data" / f"s{sid}.json").write_text(json.dumps(words, indent=1))
    print(f"scene {sid}: {probe(out):.2f}s, {len(words)} words")


async def main():
    for sid, sentences in spec["scenes"].items():
        if only and sid not in only:
            continue
        await build_scene(sid, sentences)


asyncio.run(main())
