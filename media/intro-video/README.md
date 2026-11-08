# TinyCompiler intro video (source)

The 78-second narrated intro shown in the root README. This folder holds the Remotion project
that builds it. The shipped files are in [`docs/media`](../../docs/media):

- `tinycompiler-intro.mp4`: the full video with narration and music (1920x1080, 30 fps, H.264).
- `tinycompiler-preview.gif`: a short silent loop used as the autoplay preview on GitHub.

## Layout

| Path | What it is |
|---|---|
| `src/` | Remotion scenes: `Root.tsx` (composition), `Intro.tsx`, `ExplorerAct.tsx`, `Outro.tsx`, shared theme and timing |
| `data/narration.json` | Narration text per scene and voice settings |
| `data/events.json` | Sound and animation cues, each anchored to a spoken word |
| `scripts/make_voice.py` | Generates the voiceover with edge-tts (`data/s<N>.mp3`, word timings) |
| `scripts/build_audio.py` | Builds the music bed and sound effects, mixes them, writes `src/timing.json` and `public/audio/mix.mp3` |
| `scripts/stills.mjs` | Renders single frames for checking layout |

The Explorer screens use the real compiler output (`tinyc --emit tgpu`, `--locs`) as on-screen data.

## Rebuild

```bash
cd media/intro-video
npm install
python3 scripts/make_voice.py      # only if the narration changed (needs edge-tts)
npm run audio                      # rebuilds src/timing.json and public/audio/mix.mp3
npm run render                     # writes out/tinycompiler-intro.mp4
```

Then copy the MP4 to `docs/media/` and regenerate the GIF preview from it with ffmpeg.
