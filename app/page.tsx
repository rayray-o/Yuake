"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { YuakeGestureTracker } from "../lib/gesture/tracker";
import { CursorPredictor } from "../lib/gesture/predictor";
import { GrabManager } from "../lib/gesture/grab";
import { YuakeWebSystem } from "../lib/web";
import type { GestureFrame, GestureHand } from "../lib/gesture/types";

type Status = "idle" | "starting" | "live" | "error";
type FrameMetadata = {
  mediaTime: number;
  presentedFrames: number;
  presentationTime?: number;
  expectedDisplayTime?: number;
  width?: number;
  height?: number;
  processingDuration?: number;
  captureTime?: number;
  receiveTime?: number;
};
type VideoWithFrameCallback = HTMLVideoElement & {
  requestVideoFrameCallback?: (callback: (now: number, metadata: FrameMetadata) => void) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};
type Diagnostics = {
  cameraFps: number; trackFps: number; inferenceMs: number; callbackDelayMs: number;
  decodeMs: number; droppedFrames: number; frameNumber: number; resolution: string;
  videoDelayMs: number; jitter: number; loopMs: number; isolated: boolean; processingWidth: number;
};

const EMPTY_FRAME: GestureFrame = { timestamp: 0, detected: false, hands: [], primaryHand: null, frameTime: 0, processingTime: 0 };
const EMPTY_DIAGNOSTICS: Diagnostics = {
  cameraFps: 0, trackFps: 0, inferenceMs: 0, callbackDelayMs: 0, decodeMs: 0, droppedFrames: 0,
  frameNumber: 0, resolution: "--", videoDelayMs: 0, jitter: 0, loopMs: 0, isolated: false, processingWidth: 0
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);
  const webMountRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const trackerRef = useRef<YuakeGestureTracker | null>(null);
  const webSystemRef = useRef<YuakeWebSystem | null>(null);
  const runningRef = useRef(false);
  const rafRef = useRef<number | null>(null);
  const videoCallbackRef = useRef<number | null>(null);
  const lastPresentedFrame = useRef(-1);
  const lastUiUpdate = useRef(0);
  const predictorRef = useRef(new CursorPredictor());
  const renderLoopRef = useRef<number | null>(null);
  const grabManagerRef = useRef(new GrabManager());
  const objectRef = useRef<HTMLDivElement>(null);
  const latestHandRef = useRef<GestureHand | null>(null);
  const latestTrackingTimeRef = useRef(0);

  const diagnosticRef = useRef<Diagnostics>(EMPTY_DIAGNOSTICS);
  const diagnosticWindowRef = useRef({
    start: 0, cameraFrames: 0, trackedFrames: 0, lastTrackTime: 0, lastPresentedFrame: -1,
    droppedFrames: 0, previousPoint: null as { x: number; y: number } | null,
    jitterSamples: [] as number[], lastCallbackTime: 0
  });

  const [diagnostics, setDiagnostics] = useState<Diagnostics>(EMPTY_DIAGNOSTICS);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");
  const [frame, setFrame] = useState<GestureFrame>(EMPTY_FRAME);
  const [showDebug, setShowDebug] = useState(true);

  const updateCursor = useCallback((hand: GestureHand | null) => {
    const cursor = cursorRef.current;
    latestHandRef.current = hand;
    latestTrackingTimeRef.current = performance.now();
    if (!cursor) return;
    if (!hand) { cursor.style.opacity = "0"; return; }
    predictorRef.current.update(hand.cursor, hand.velocity, performance.now());
    cursor.style.opacity = "1";
    cursor.classList.toggle("pinch", hand.pinch);
    const label = cursor.querySelector(".cursorLabel");
    if (label) label.textContent = hand.pinch ? "GRAB" : hand.pose.toUpperCase();
    const point = { x: hand.cursor.x, y: hand.cursor.y };
    const d = diagnosticWindowRef.current;
    if (d.previousPoint) {
      const dx = point.x - d.previousPoint.x, dy = point.y - d.previousPoint.y;
      d.jitterSamples.push(Math.sqrt(dx * dx + dy * dy));
      if (d.jitterSamples.length > 120) d.jitterSamples.shift();
    }
    d.previousPoint = point;
  }, []);

  const renderCursorLoop = useCallback((rafTimestamp?: number) => {
    if (!runningRef.current) { renderLoopRef.current = null; return; }
    const now = rafTimestamp ?? performance.now();
    const predicted = predictorRef.current.predict(now);
    const cursor = cursorRef.current;
    if (predicted && cursor) {
      cursor.style.transform = `translate3d(${predicted.x * 100}vw,${predicted.y * 100}vh,0)`;
    }

    const box = objectRef.current;
    const grab = grabManagerRef.current;
    const objects = grab.getObjects();
    const grabbedId = grab.getGrabbedId();
    const predictedObject = grabbedId ? grab.predict(now) : null;
    if (box && objects.length) {
      const object = objects[0];
      const drawX = predictedObject?.x ?? object.x;
      const drawY = predictedObject?.y ?? object.y;
      box.style.transform = `translate3d(${drawX}px,${drawY}px,0) translate(-50%, -50%)`;
      box.classList.toggle("grabbed", grabbedId === object.id);
    }

    // The web system runs on the display clock, not React or MediaPipe cadence.
    const web = webSystemRef.current;
    const video = videoRef.current;
    const hand = latestHandRef.current;
    if (web && video) web.update({ hand, video, now, viewportWidth: window.innerWidth, viewportHeight: window.innerHeight });

    renderLoopRef.current = requestAnimationFrame(renderCursorLoop);
  }, []);

  const publishDiagnostics = useCallback(() => {
    const d = diagnosticWindowRef.current;
    const now = performance.now();
    if (!d.start) return;
    const elapsed = now - d.start;
    if (elapsed < 500) return;
    const cameraFps = d.cameraFrames / (elapsed / 1000);
    const trackFps = d.trackedFrames / (elapsed / 1000);
    const jitter = d.jitterSamples.length
      ? (d.jitterSamples.reduce((s, v) => s + v, 0) / d.jitterSamples.length) * 1000
      : 0;
    const next: Diagnostics = {
      ...diagnosticRef.current,
      cameraFps,
      trackFps,
      jitter,
      isolated: typeof window !== "undefined" && window.crossOriginIsolated === true,
      processingWidth: trackerRef.current?.getProcessingWidth() ?? 0
    };
    diagnosticRef.current = next;
    setDiagnostics({ ...next });
  }, []);

  const stopCamera = useCallback(() => {
    runningRef.current = false;
    const video = videoRef.current as VideoWithFrameCallback | null;
    if (video && videoCallbackRef.current !== null && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(videoCallbackRef.current);
    videoCallbackRef.current = null;
    if (rafRef.current !== null) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    if (renderLoopRef.current !== null) { cancelAnimationFrame(renderLoopRef.current); renderLoopRef.current = null; }
    predictorRef.current.clear();
    trackerRef.current?.close();
    trackerRef.current = null;
    webSystemRef.current?.reset();
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    if (videoRef.current) { videoRef.current.pause(); videoRef.current.srcObject = null; }
    lastPresentedFrame.current = -1;
    latestHandRef.current = null;
    latestTrackingTimeRef.current = 0;
    diagnosticWindowRef.current = {
      start: 0, cameraFrames: 0, trackedFrames: 0, lastTrackTime: 0, lastPresentedFrame: -1,
      droppedFrames: 0, previousPoint: null, jitterSamples: [], lastCallbackTime: 0
    };
    diagnosticRef.current = EMPTY_DIAGNOSTICS;
    setDiagnostics(EMPTY_DIAGNOSTICS);
    updateCursor(null);
    grabManagerRef.current.update(null, performance.now());
    setFrame(EMPTY_FRAME);
    setStatus("idle");
  }, [updateCursor]);

  const publishFrame = useCallback((result: GestureFrame, timestamp: number) => {
    if (timestamp - lastUiUpdate.current < 80) return;
    lastUiUpdate.current = timestamp;
    setFrame(result);
    publishDiagnostics();
  }, [publishDiagnostics]);

  const scheduleFallback = useCallback(() => {
    if (!runningRef.current) return;
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(timestamp => processFrame(timestamp));
  }, []);

  const processFrame = useCallback((timestamp: number, metadata?: FrameMetadata) => {
    if (!runningRef.current) return;
    const video = videoRef.current;
    const tracker = trackerRef.current;
    if (!video || !tracker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) { scheduleFallback(); return; }
    const d = diagnosticWindowRef.current;
    d.cameraFrames++;
    if (metadata && d.lastPresentedFrame >= 0) {
      const difference = metadata.presentedFrames - d.lastPresentedFrame;
      if (difference > 1) d.droppedFrames += difference - 1;
    }
    if (metadata) { d.lastPresentedFrame = metadata.presentedFrames; lastPresentedFrame.current = metadata.presentedFrames; }
    const width = metadata?.width ?? video.videoWidth;
    const height = metadata?.height ?? video.videoHeight;
    diagnosticRef.current.resolution = width && height ? `${width}×${height}` : "--";
    diagnosticRef.current.callbackDelayMs = metadata?.presentationTime !== undefined ? Math.max(0, performance.now() - metadata.presentationTime) : 0;
    diagnosticRef.current.decodeMs = metadata?.processingDuration ?? 0;
    if (d.lastTrackTime) diagnosticRef.current.loopMs = timestamp - d.lastTrackTime;
    d.lastTrackTime = timestamp;

    const before = performance.now();
    try {
      const result = tracker.process(video, timestamp);
      const after = performance.now();
      d.trackedFrames++;
      diagnosticRef.current.inferenceMs = after - before;
      updateCursor(result.primaryHand);
      grabManagerRef.current.update(result.primaryHand, after);
      diagnosticRef.current.videoDelayMs = metadata?.presentationTime !== undefined ? Math.max(0, after - metadata.presentationTime) : 0;
      diagnosticRef.current.frameNumber = metadata?.presentedFrames ?? diagnosticRef.current.frameNumber + 1;
      publishFrame(result, timestamp);
    } catch (err) {
      console.error("YUAKE tracking error:", err);
      setError(err instanceof Error ? err.message : String(err));
      setStatus("error");
      stopCamera();
    }
  }, [publishFrame, scheduleFallback, stopCamera, updateCursor]);

  const startFrameLoop = useCallback(() => {
    if (!runningRef.current) return;
    const video = videoRef.current as VideoWithFrameCallback | null;
    if (!video) return;
    if (video.requestVideoFrameCallback) {
      const callback = (now: number, metadata: FrameMetadata) => {
        if (!runningRef.current) return;
        processFrame(now, metadata);
        videoCallbackRef.current = video.requestVideoFrameCallback!(callback);
      };
      videoCallbackRef.current = video.requestVideoFrameCallback(callback);
    } else scheduleFallback();
  }, [processFrame, scheduleFallback]);

  const startCamera = useCallback(async () => {
    if (runningRef.current) return;
    setError(""); setStatus("starting");
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access is not supported by this browser.");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60, min: 30 } },
        audio: false
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("Camera element unavailable.");
      video.srcObject = stream;
      video.style.transform = "none";
      await video.play();

      const tracker = new YuakeGestureTracker();
      trackerRef.current = tracker;
      await tracker.initialize();

      diagnosticWindowRef.current = {
        start: performance.now(), cameraFrames: 0, trackedFrames: 0, lastTrackTime: 0, lastPresentedFrame: -1,
        droppedFrames: 0, previousPoint: null, jitterSamples: [], lastCallbackTime: 0
      };
      diagnosticRef.current = { ...EMPTY_DIAGNOSTICS, resolution: video.videoWidth && video.videoHeight ? `${video.videoWidth}×${video.videoHeight}` : "--" };
      setDiagnostics(diagnosticRef.current);
      runningRef.current = true;
      lastPresentedFrame.current = -1;
      lastUiUpdate.current = performance.now();
      setStatus("live");
      startFrameLoop();
      renderLoopRef.current = requestAnimationFrame(renderCursorLoop);
    } catch (err) {
      console.error(err);
      runningRef.current = false;
      streamRef.current?.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      setError(err instanceof Error ? err.message : String(err));
      setStatus("error");
    }
  }, [renderCursorLoop, startFrameLoop]);

  useEffect(() => {
    if (!webMountRef.current) return;
    const system = new YuakeWebSystem();
    system.mount(webMountRef.current);
    webSystemRef.current = system;
    return () => {
      if (webSystemRef.current === system) webSystemRef.current = null;
      system.destroy();
    };
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  useEffect(() => {
    grabManagerRef.current.setObjects([{ id: "box1", x: window.innerWidth / 2, y: window.innerHeight / 2, radius: 70 }]);
  }, []);

  useEffect(() => {
    if (status !== "live") return;
    const interval = window.setInterval(publishDiagnostics, 1000);
    return () => window.clearInterval(interval);
  }, [status, publishDiagnostics]);

  const primary = frame.primaryHand;
  const pinchPercent = Math.round((primary?.pinchStrength ?? 0) * 100);
  const webDiagnostics = webSystemRef.current?.getDiagnostics();

  return (
    <main className="yuake">
      <video ref={videoRef} className="camera" autoPlay muted playsInline />
      <div className="cameraShade" />
      <div ref={webMountRef} aria-hidden="true" />

      <header className="topBar">
        <div className="brand">YUAKE</div>
        <div className="status"><span className={status === "live" ? "statusDot live" : "statusDot"} />{status === "live" ? "TRACKING" : status === "starting" ? "INITIALIZING" : status === "error" ? "ERROR" : "OFFLINE"}</div>
      </header>

      {status === "idle" && <section className="intro">
        <div className="introEyebrow">CAMERA INPUT SYSTEM</div>
        <h1>ENTER<br />REALITY.</h1>
        <p>Your hand becomes the interface.</p>
        <button className="enterButton" onClick={startCamera}><span>ENTER REALITY</span><span>→</span></button>
        <div className="introNote">Processing happens locally in your browser.</div>
      </section>}

      {status === "starting" && <section className="loading"><div className="loadingRing" /><div><strong>INITIALIZING</strong><span>Loading vision system</span></div></section>}

      {status === "live" && <>
        <section className="centerMessage">
          {!frame.detected ? <><span>SEARCHING</span><h2>SHOW YOUR HAND</h2><p>Move a hand into the camera view.</p></> : <><span>{(primary?.pose ?? "none").toUpperCase()}</span><h2>{primary?.pinch ? "PINCH" : primary?.pose === "point" ? "POINT" : primary?.pose === "open" ? "OPEN" : primary?.pose === "fist" ? "FIST" : "TRACKING"}</h2><p>{primary?.pinch ? `${pinchPercent}% GRIP` : "Hand locked"}</p></>}
        </section>

        <div ref={cursorRef} className="handCursor" style={{ opacity: 0, left: 0, top: 0 }}><div className="cursorCore" /><div className="cursorRing" /><div className="cursorLabel">POINT</div></div>
        <div ref={objectRef} className="grabObject" style={{ left: 0, top: 0 }} />

        <footer className="bottomBar">
          <div>HANDS {frame.hands.length} / 2</div><div>{Math.round(frame.processingTime)}ms</div>
          <button onClick={() => setShowDebug(v => !v)}>{showDebug ? "HIDE DEBUG" : "DEBUG"}</button>
          <button onClick={stopCamera}>EXIT</button>
        </footer>

        {showDebug && <DebugPanel frame={frame} diagnostics={diagnostics} webDiagnostics={webDiagnostics} />}
      </>}

      {status === "error" && <section className="errorScreen"><div className="errorCode">TRACKING ERROR</div><h2>THE SYSTEM<br />COULDN'T START.</h2><pre>{error}</pre><button onClick={() => { setError(""); setStatus("idle"); }}>TRY AGAIN</button></section>}
    </main>
  );
}

function DebugPanel({ frame, diagnostics, webDiagnostics }: { frame: GestureFrame; diagnostics: Diagnostics; webDiagnostics?: { poseConfidence: number; targetConfidence: number; speed: number; distance: number; tension: number; state: string } }) {
  const hand = frame.primaryHand;
  const stats = [
    ["DETECTED", frame.detected ? "YES" : "NO"], ["HANDS", String(frame.hands.length)], ["POSE", hand?.pose ?? "none"],
    ["PINCH", hand?.pinch ? "ACTIVE" : "OFF"], ["STRENGTH", `${Math.round((hand?.pinchStrength ?? 0) * 100)}%`],
    ["SPEED", (hand?.speed ?? 0).toFixed(2)], ["CONF", `${Math.round((hand?.confidence ?? 0) * 100)}%`]
  ];
  const pipe = [
    ["CAM FPS", diagnostics.cameraFps.toFixed(1)], ["TRACK FPS", diagnostics.trackFps.toFixed(1)], ["INFER", `${diagnostics.inferenceMs.toFixed(1)}ms`],
    ["LOOP", `${diagnostics.loopMs.toFixed(1)}ms`], ["CALLBACK", `${diagnostics.callbackDelayMs.toFixed(1)}ms`], ["DECODE", `${diagnostics.decodeMs.toFixed(1)}ms`],
    ["VIDEO DLY", `${diagnostics.videoDelayMs.toFixed(1)}ms`], ["JITTER", diagnostics.jitter.toFixed(2)], ["DROPPED", String(diagnostics.droppedFrames)],
    ["ISOLATED", diagnostics.isolated ? "YES" : "NO"], ["PROC W", `${diagnostics.processingWidth}px`], ["FRAME", String(diagnostics.frameNumber)], ["RES", diagnostics.resolution]
  ];
  const web = webDiagnostics ? [
    ["WEB POSE", `${Math.round(webDiagnostics.poseConfidence * 100)}%`], ["WEB STATE", webDiagnostics.state.toUpperCase()],
    ["TARGET", `${Math.round(webDiagnostics.targetConfidence * 100)}%`], ["WEB SPEED", `${Math.round(webDiagnostics.speed)}`],
    ["WEB DIST", `${Math.round(webDiagnostics.distance)}px`], ["TENSION", `${Math.round(webDiagnostics.tension * 100)}%`]
  ] : [];
  const group = (title: string, items: string[][]) => <><div className="debugTitle">{title}</div><div className="debugGrid">{items.map(([a, b]) => <div className="debugStat" key={a}><span>{a}</span><strong>{b}</strong></div>)}</div></>;
  return <aside className="debug">{group("YUAKE / VISION", stats)}<div className="debugDivider" />{group("PIPELINE", pipe)}{web.length > 0 && <><div className="debugDivider" />{group("WEB SYSTEM", web)}</>}</aside>;
}
