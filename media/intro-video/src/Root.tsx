import React from "react";
import { AbsoluteFill, Audio, Composition, Sequence, staticFile } from "remotion";
import { Background, Progress } from "./fx";
import { ExplorerAct, ACT_FRAMES } from "./ExplorerAct";
import { Scene1, Scene2 } from "./Intro";
import { Scene7, Scene8, Scene9 } from "./Outro";
import { FPS, TOTAL, scene } from "./sync";
import { C } from "./theme";

const Video: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg }}>
    <Audio src={staticFile("audio/mix.mp3")} />
    <Background />
    <Sequence from={scene(1).start} durationInFrames={scene(1).frames}><Scene1 /></Sequence>
    <Sequence from={scene(2).start} durationInFrames={scene(2).frames}><Scene2 /></Sequence>
    <Sequence from={scene(3).start} durationInFrames={ACT_FRAMES}><ExplorerAct /></Sequence>
    <Sequence from={scene(7).start} durationInFrames={scene(7).frames}><Scene7 /></Sequence>
    <Sequence from={scene(8).start} durationInFrames={scene(8).frames}><Scene8 /></Sequence>
    <Sequence from={scene(9).start} durationInFrames={scene(9).frames}><Scene9 /></Sequence>
    <Progress />
  </AbsoluteFill>
);

export const Root: React.FC = () => (
  <Composition id="TinyCompilerIntro" component={Video} durationInFrames={TOTAL} fps={FPS} width={1920} height={1080} />
);
