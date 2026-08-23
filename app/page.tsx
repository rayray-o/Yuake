"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState
} from "react";

import {
  YuakeGestureTracker
} from "../lib/gesture/tracker";

import {
  CursorPredictor
} from "../lib/gesture/predictor";

import {
  GrabManager
} from "../lib/gesture/grab";

import type {
  GestureFrame,
  GestureHand
} from "../lib/gesture/types";

type Status =
  | "idle"
  | "starting"
  | "live"
  | "error";

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

type VideoWithFrameCallback =
  HTMLVideoElement & {
    requestVideoFrameCallback?: (
      callback: (
        now: number,
        metadata: FrameMetadata
      ) => void
    ) => number;

    cancelVideoFrameCallback?: (
      handle: number
    ) => void;
  };

type Diagnostics = {
  cameraFps: number;
  trackFps: number;
  inferenceMs: number;
  callbackDelayMs: number;
  decodeMs: number;
  droppedFrames: number;
  frameNumber: number;
  resolution: string;
  videoDelayMs: number;
  jitter: number;
  loopMs: number;
};

const EMPTY_FRAME:
  GestureFrame = {
    timestamp: 0,
    detected: false,
    hands: [],
    primaryHand: null,
    frameTime: 0,
    processingTime: 0
  };

const EMPTY_DIAGNOSTICS:
  Diagnostics = {
    cameraFps: 0,
    trackFps: 0,
    inferenceMs: 0,
    callbackDelayMs: 0,
    decodeMs: 0,
    droppedFrames: 0,
    frameNumber: 0,
    resolution: "--",
    videoDelayMs: 0,
    jitter: 0,
    loopMs: 0
  };

export default function Home() {
  const videoRef =
    useRef<HTMLVideoElement>(null);

  const cursorRef =
    useRef<HTMLDivElement>(null);

  const streamRef =
    useRef<MediaStream | null>(
      null
    );

  const trackerRef =
    useRef<YuakeGestureTracker | null>(
      null
    );

  const runningRef =
    useRef(false);

  const rafRef =
    useRef<number | null>(null);

  const videoCallbackRef =
    useRef<number | null>(
      null
    );

  const lastPresentedFrame =
    useRef(-1);

  const lastUiUpdate =
    useRef(0);

  const predictorRef =
    useRef(
      new CursorPredictor()
    );

  const renderLoopRef =
    useRef<number | null>(
      null
    );

  const grabManagerRef =
    useRef(
      new GrabManager()
    );

  const objectRef =
    useRef<HTMLDivElement>(
      null
    );

  /*
   * --------------------------------------------------
   * DIAGNOSTIC STATE
   * --------------------------------------------------
   */

  const diagnosticRef =
    useRef<Diagnostics>(
      EMPTY_DIAGNOSTICS
    );

  const diagnosticWindowRef =
    useRef({
      start: 0,

      cameraFrames: 0,

      trackedFrames: 0,

      lastTrackTime: 0,

      lastPresentedFrame: -1,

      droppedFrames: 0,

      previousPoint: null as {
        x: number;
        y: number;
      } | null,

      jitterSamples: [] as number[],

      lastCallbackTime: 0
    });

  const [diagnostics, setDiagnostics] =
    useState<Diagnostics>(
      EMPTY_DIAGNOSTICS
    );

  const [status, setStatus] =
    useState<Status>("idle");

  const [error, setError] =
    useState("");

  const [frame, setFrame] =
    useState<GestureFrame>(
      EMPTY_FRAME
    );

  const [showDebug, setShowDebug] =
    useState(true);

  /*
   * --------------------------------------------------
   * DIRECT CURSOR RENDERER
   * --------------------------------------------------
   */

  const updateCursor =
    useCallback(
      (
        hand: GestureHand | null
      ) => {
        const cursor =
          cursorRef.current;

        if (!cursor) {
          return;
        }

        if (!hand) {
          cursor.style.opacity =
            "0";

          return;
        }

        /*
         * Position is NOT set here anymore.
         *
         * The predictor/render loop owns position
         * so the dot can glide at full screen
         * refresh rate between real detections
         * instead of teleporting whenever a new
         * MediaPipe result arrives.
         */
        predictorRef.current.update(
          hand.cursor,
          hand.velocity,
          performance.now()
        );

        cursor.style.opacity =
          "1";

        cursor.classList.toggle(
          "pinch",
          hand.pinch
        );

        const label =
          cursor.querySelector(
            ".cursorLabel"
          );

        if (label) {
          label.textContent =
            hand.pinch
              ? "GRAB"
              : hand.pose.toUpperCase();
        }

        /*
         * ------------------------------------------------
         * FINGERTIP JITTER MEASUREMENT
         * ------------------------------------------------
         *
         * This measures movement between consecutive
         * tracker outputs.
         *
         * It does NOT alter the cursor.
         */
        const point = {
          x: hand.cursor.x,
          y: hand.cursor.y
        };

        const diagnostic =
          diagnosticWindowRef.current;

        if (
          diagnostic.previousPoint
        ) {
          const dx =
            point.x -
            diagnostic.previousPoint.x;

          const dy =
            point.y -
            diagnostic.previousPoint.y;

          const distance =
            Math.sqrt(
              dx * dx +
              dy * dy
            );

          /*
           * Only retain the recent window.
           */
          diagnostic.jitterSamples.push(
            distance
          );

          if (
            diagnostic.jitterSamples
              .length > 120
          ) {
            diagnostic.jitterSamples.shift();
          }
        }

        diagnostic.previousPoint =
          point;
      },
      []
    );

  /*
   * --------------------------------------------------
   * CURSOR RENDER LOOP
   * --------------------------------------------------
   *
   * Runs every animation frame (screen refresh rate),
   * completely decoupled from how often MediaPipe
   * finishes a detection. Draws the PREDICTED
   * position, so the dot glides smoothly even when
   * inference is slower than the display.
   */

  const renderCursorLoop =
    useCallback(() => {
      if (
        !runningRef.current
      ) {
        renderLoopRef.current =
          null;

        return;
      }

      const predicted =
        predictorRef.current.predict(
          performance.now()
        );

      const cursor =
        cursorRef.current;

      if (
        predicted &&
        cursor
      ) {
        cursor.style.transform =
          `translate3d(` +
          `${predicted.x * 100}vw,` +
          `${predicted.y * 100}vh,` +
          `0)`;
      }

      const box =
        objectRef.current;

      const objects =
        grabManagerRef.current.getObjects();

      const grabbedId =
        grabManagerRef.current.getGrabbedId();

      const predictedObject =
        grabManagerRef.current.predict(
          performance.now()
        );

      if (
        box &&
        objects[0]
      ) {
        /*
         * While grabbed, use the predictor's
         * glide-smoothed position so the object
         * doesn't stair-step between MediaPipe
         * detections. At rest, just draw its
         * last known (static) position.
         */
        const drawX =
          predictedObject?.x ??
          objects[0].x;

        const drawY =
          predictedObject?.y ??
          objects[0].y;

        /*
         * Setting style.transform here fully
         * REPLACES the CSS class's transform,
         * so the -50%/-50% centering must be
         * included here too, or the box renders
         * offset by half its own size.
         */
        box.style.transform =
          `translate3d(` +
          `${drawX}px,` +
          `${drawY}px,` +
          `0) translate(-50%, -50%)`;

        box.classList.toggle(
          "grabbed",
          grabbedId ===
            objects[0].id
        );
      }

      renderLoopRef.current =
        requestAnimationFrame(
          renderCursorLoop
        );
    }, []);

  /*
   * --------------------------------------------------
   * DIAGNOSTIC UPDATE
   * --------------------------------------------------
   */

  const publishDiagnostics =
    useCallback(() => {
      const diagnostic =
        diagnosticWindowRef.current;

      const now =
        performance.now();

      if (
        diagnostic.start === 0
      ) {
        return;
      }

      const elapsed =
        now -
        diagnostic.start;

      if (
        elapsed < 500
      ) {
        return;
      }

      const cameraFps =
        diagnostic.cameraFrames /
        (elapsed / 1000);

      const trackFps =
        diagnostic.trackedFrames /
        (elapsed / 1000);

      let jitter = 0;

      if (
        diagnostic.jitterSamples
          .length
      ) {
        const total =
          diagnostic.jitterSamples.reduce(
            (sum, value) =>
              sum + value,
            0
          );

        jitter =
          (
            total /
            diagnostic.jitterSamples.length
          ) * 1000;
      }

      const previous =
        diagnosticRef.current;

      const next: Diagnostics = {
        ...previous,

        cameraFps,

        trackFps,

        jitter
      };

      diagnosticRef.current =
        next;

      setDiagnostics({
        ...next
      });
    }, []);

  /*
   * --------------------------------------------------
   * STOP CAMERA
   * --------------------------------------------------
   */

  const stopCamera =
    useCallback(() => {
      runningRef.current =
        false;

      const video =
        videoRef.current as
          | VideoWithFrameCallback
          | null;

      if (
        video &&
        videoCallbackRef.current !==
          null &&
        video.cancelVideoFrameCallback
      ) {
        video.cancelVideoFrameCallback(
          videoCallbackRef.current
        );
      }

      videoCallbackRef.current =
        null;

      if (
        rafRef.current !==
        null
      ) {
        cancelAnimationFrame(
          rafRef.current
        );

        rafRef.current =
          null;
      }

      if (
        renderLoopRef.current !==
        null
      ) {
        cancelAnimationFrame(
          renderLoopRef.current
        );

        renderLoopRef.current =
          null;
      }

      predictorRef.current.clear();

      trackerRef.current?.close();

      trackerRef.current =
        null;

      streamRef.current
        ?.getTracks()
        .forEach(
          track =>
            track.stop()
        );

      streamRef.current =
        null;

      if (videoRef.current) {
        videoRef.current.pause();

        videoRef.current.srcObject =
          null;
      }

      lastPresentedFrame.current =
        -1;

      diagnosticWindowRef.current = {
        start: 0,

        cameraFrames: 0,

        trackedFrames: 0,

        lastTrackTime: 0,

        lastPresentedFrame: -1,

        droppedFrames: 0,

        previousPoint: null,

        jitterSamples: [],

        lastCallbackTime: 0
      };

      diagnosticRef.current =
        EMPTY_DIAGNOSTICS;

      setDiagnostics(
        EMPTY_DIAGNOSTICS
      );

      updateCursor(null);

      grabManagerRef.current.update(
        null,
        performance.now()
      );

      setFrame(
        EMPTY_FRAME
      );

      setStatus("idle");
    }, [updateCursor]);

  /*
   * --------------------------------------------------
   * UI FRAME UPDATE
   * --------------------------------------------------
   */

  const publishFrame =
    useCallback(
      (
        result: GestureFrame,
        timestamp: number
      ) => {
        if (
          timestamp -
            lastUiUpdate.current <
          80
        ) {
          return;
        }

        lastUiUpdate.current =
          timestamp;

        setFrame(result);

        /*
         * Diagnostic display is also deliberately
         * throttled. It must NEVER be part of the
         * cursor hot path.
         */
        publishDiagnostics();
      },
      [publishDiagnostics]
    );

  /*
   * --------------------------------------------------
   * PROCESS CAMERA FRAME
   * --------------------------------------------------
   */

  const processFrame =
    useCallback(
      (
        timestamp: number,
        metadata?: FrameMetadata
      ) => {
        if (
          !runningRef.current
        ) {
          return;
        }

        const video =
          videoRef.current;

        const tracker =
          trackerRef.current;

        if (
          !video ||
          !tracker ||
          video.readyState <
            HTMLMediaElement.HAVE_CURRENT_DATA
        ) {
          scheduleFallback();

          return;
        }

        const diagnostic =
          diagnosticWindowRef.current;

        /*
         * ------------------------------------------------
         * CAMERA FRAME COUNT
         * ------------------------------------------------
         */

        diagnostic.cameraFrames++;

        /*
         * ------------------------------------------------
         * DROPPED FRAME COUNT
         * ------------------------------------------------
         */

        if (
          metadata &&
          diagnostic.lastPresentedFrame >=
            0
        ) {
          const difference =
            metadata.presentedFrames -
            diagnostic.lastPresentedFrame;

          if (
            difference > 1
          ) {
            diagnostic.droppedFrames +=
              difference - 1;
          }
        }

        if (metadata) {
          diagnostic.lastPresentedFrame =
            metadata.presentedFrames;

          lastPresentedFrame.current =
            metadata.presentedFrames;
        }

        /*
         * ------------------------------------------------
         * CAMERA RESOLUTION
         * ------------------------------------------------
         */

        const width =
          metadata?.width ??
          video.videoWidth;

        const height =
          metadata?.height ??
          video.videoHeight;

        diagnosticRef.current.resolution =
          width && height
            ? `${width}×${height}`
            : "--";

        /*
         * ------------------------------------------------
         * CALLBACK TIMING
         * ------------------------------------------------
         */

        if (
          metadata?.presentationTime !==
            undefined
        ) {
          diagnosticRef.current.callbackDelayMs =
            Math.max(
              0,
              performance.now() -
                metadata.presentationTime
            );
        } else {
          diagnosticRef.current.callbackDelayMs =
            0;
        }

        /*
         * Chrome exposes processingDuration
         * on requestVideoFrameCallback metadata
         * when available.
         */
        diagnosticRef.current.decodeMs =
          metadata?.processingDuration ??
          0;

        /*
         * ------------------------------------------------
         * TRACKING INTERVAL
         * ------------------------------------------------
         */

        if (
          diagnostic.lastTrackTime
        ) {
          diagnosticRef.current.loopMs =
            timestamp -
            diagnostic.lastTrackTime;
        }

        diagnostic.lastTrackTime =
          timestamp;

        /*
         * ------------------------------------------------
         * ACTUAL MEDIAPIPE CALL
         * ------------------------------------------------
         */

        const before =
          performance.now();

        try {
          const result =
            tracker.process(
              video,
              timestamp
            );

          const after =
            performance.now();

          diagnostic.trackedFrames++;

          diagnosticRef.current.inferenceMs =
            after -
            before;

          /*
           * Cursor is updated immediately.
           *
           * No React state.
           */
          updateCursor(
            result.primaryHand
          );

          grabManagerRef.current.update(
            result.primaryHand,
            after
          );

          /*
           * ------------------------------------------------
           * FRAME AGE
           * ------------------------------------------------
           *
           * This is NOT pretending we can directly
           * recover sensor capture time.
           *
           * It measures the delay between the
           * presentation callback timestamp and
           * processing completion.
           */
          if (
            metadata?.presentationTime !==
              undefined
          ) {
            diagnosticRef.current.videoDelayMs =
              Math.max(
                0,
                after -
                  metadata.presentationTime
              );
          } else {
            diagnosticRef.current.videoDelayMs =
              0;
          }

          diagnosticRef.current.frameNumber =
            metadata?.presentedFrames ??
            diagnosticRef.current.frameNumber +
              1;

          publishFrame(
            result,
            timestamp
          );
        } catch (err) {
          console.error(
            "YUAKE tracking error:",
            err
          );

          setError(
            err instanceof Error
              ? err.message
              : String(err)
          );

          setStatus("error");

          stopCamera();

          return;
        }
      },
      [
        publishFrame,
        stopCamera,
        updateCursor
      ]
    );

  /*
   * --------------------------------------------------
   * RAF FALLBACK
   * --------------------------------------------------
   */

  const scheduleFallback =
    useCallback(() => {
      if (
        !runningRef.current
      ) {
        return;
      }

      if (
        rafRef.current !==
        null
      ) {
        cancelAnimationFrame(
          rafRef.current
        );
      }

      rafRef.current =
        requestAnimationFrame(
          timestamp => {
            processFrame(
              timestamp
            );
          }
        );
    }, [processFrame]);

  /*
   * --------------------------------------------------
   * VIDEO FRAME LOOP
   * --------------------------------------------------
   */

  const startFrameLoop =
    useCallback(() => {
      if (
        !runningRef.current
      ) {
        return;
      }

      const video =
        videoRef.current as
          | VideoWithFrameCallback
          | null;

      if (!video) {
        return;
      }

      if (
        video.requestVideoFrameCallback
      ) {
        const callback =
          (
            now: number,
            metadata: FrameMetadata
          ) => {
            if (
              !runningRef.current
            ) {
              return;
            }

            processFrame(
              now,
              metadata
            );

            videoCallbackRef.current =
              video.requestVideoFrameCallback!(
                callback
              );
          };

        videoCallbackRef.current =
          video.requestVideoFrameCallback(
            callback
          );

        return;
      }

      scheduleFallback();
    }, [
      processFrame,
      scheduleFallback
    ]);

  /*
   * --------------------------------------------------
   * CAMERA START
   * --------------------------------------------------
   */

  const startCamera =
    useCallback(
      async () => {
        if (
          runningRef.current
        ) {
          return;
        }

        setError("");
        setStatus("starting");

        try {
          if (
            !navigator.mediaDevices
              ?.getUserMedia
          ) {
            throw new Error(
              "Camera access is not supported by this browser."
            );
          }

          const stream =
            await navigator.mediaDevices.getUserMedia(
              {
                video: {
                  facingMode: {
                    ideal:
                      "environment"
                  },

                  /*
                   * Was requesting up to 1920x1080,
                   * but everything downstream gets
                   * downscaled to MAX_PROCESSING_WIDTH
                   * (480px) before MediaPipe ever sees
                   * it. Capturing beyond ~720p buys
                   * zero detection benefit and costs
                   * real per-frame decode/copy time,
                   * especially on mobile.
                   */
                  width: {
                    ideal: 1280
                  },

                  height: {
                    ideal: 720
                  },

                  frameRate: {
                    ideal: 60,

                    min: 30
                  }
                },

                audio: false
              }
            );

          streamRef.current =
            stream;

          const video =
            videoRef.current;

          if (!video) {
            throw new Error(
              "Camera element unavailable."
            );
          }

          video.srcObject =
            stream;

          /*
           * Rear camera.
           * Never mirror it.
           */
          video.style.transform =
            "none";

          await video.play();

          const tracker =
            new YuakeGestureTracker();

          trackerRef.current =
            tracker;

          await tracker.initialize();

          /*
           * Reset diagnostics.
           */
          diagnosticWindowRef.current = {
            start:
              performance.now(),

            cameraFrames: 0,

            trackedFrames: 0,

            lastTrackTime: 0,

            lastPresentedFrame: -1,

            droppedFrames: 0,

            previousPoint: null,

            jitterSamples: [],

            lastCallbackTime: 0
          };

          diagnosticRef.current =
            {
              ...EMPTY_DIAGNOSTICS,

              resolution:
                video.videoWidth &&
                video.videoHeight
                  ? `${video.videoWidth}×${video.videoHeight}`
                  : "--"
            };

          setDiagnostics(
            diagnosticRef.current
          );

          runningRef.current =
            true;

          lastPresentedFrame.current =
            -1;

          lastUiUpdate.current =
            performance.now();

          setStatus("live");

          startFrameLoop();

          renderLoopRef.current =
            requestAnimationFrame(
              renderCursorLoop
            );
        } catch (err) {
          console.error(err);

          runningRef.current =
            false;

          streamRef.current
            ?.getTracks()
            .forEach(
              track =>
                track.stop()
            );

          streamRef.current =
            null;

          setError(
            err instanceof Error
              ? err.message
              : String(err)
          );

          setStatus("error");
        }
      },
      [
        startFrameLoop,
        renderCursorLoop
      ]
    );

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  /*
   * Register one placeholder grabbable box, centered
   * on screen, so the grab/release mechanic can be
   * tested before any real 3D objects exist.
   */
  useEffect(() => {
    grabManagerRef.current.setObjects([
      {
        id: "box1",
        x: window.innerWidth / 2,
        y: window.innerHeight / 2,
        radius: 70
      }
    ]);
  }, []);

  /*
   * Refresh diagnostic numbers once per second.
   *
   * This is completely separate from the
   * cursor renderer.
   */
  useEffect(() => {
    if (
      status !== "live"
    ) {
      return;
    }

    const interval =
      window.setInterval(
        () => {
          publishDiagnostics();
        },
        1000
      );

    return () =>
      window.clearInterval(
        interval
      );
  }, [
    status,
    publishDiagnostics
  ]);

  const primary =
    frame.primaryHand;

  const pinchPercent =
    Math.round(
      (
        primary?.pinchStrength ??
        0
      ) * 100
    );

  return (
    <main className="yuake">
      <video
        ref={videoRef}
        className="camera"
        autoPlay
        muted
        playsInline
      />

      <div className="cameraShade" />

      <header className="topBar">
        <div className="brand">
          YUAKE
        </div>

        <div className="status">
          <span
            className={
              status === "live"
                ? "statusDot live"
                : "statusDot"
            }
          />

          {status === "live"
            ? "TRACKING"
            : status === "starting"
              ? "INITIALIZING"
              : status === "error"
                ? "ERROR"
                : "OFFLINE"}
        </div>
      </header>

      {status === "idle" && (
        <section className="intro">
          <div className="introEyebrow">
            CAMERA INPUT SYSTEM
          </div>

          <h1>
            ENTER
            <br />
            REALITY.
          </h1>

          <p>
            Your hand becomes
            the interface.
          </p>

          <button
            className="enterButton"
            onClick={
              startCamera
            }
          >
            <span>
              ENTER REALITY
            </span>

            <span>
              →
            </span>
          </button>

          <div className="introNote">
            Processing happens
            locally in your browser.
          </div>
        </section>
      )}

      {status === "starting" && (
        <section className="loading">
          <div className="loadingRing" />

          <div>
            <strong>
              INITIALIZING
            </strong>

            <span>
              Loading vision system
            </span>
          </div>
        </section>
      )}

      {status === "live" && (
        <>
          <section className="centerMessage">
            {!frame.detected ? (
              <>
                <span>
                  SEARCHING
                </span>

                <h2>
                  SHOW YOUR HAND
                </h2>

                <p>
                  Move a hand into
                  the camera view.
                </p>
              </>
            ) : (
              <>
                <span>
                  {(
                    primary?.pose ??
                    "none"
                  ).toUpperCase()}
                </span>

                <h2>
                  {primary?.pinch
                    ? "PINCH"
                    : primary?.pose ===
                        "point"
                      ? "POINT"
                      : primary?.pose ===
                          "open"
                        ? "OPEN"
                        : primary?.pose ===
                            "fist"
                          ? "FIST"
                          : "TRACKING"}
                </h2>

                <p>
                  {primary?.pinch
                    ? `${pinchPercent}% GRIP`
                    : "Hand locked"}
                </p>
              </>
            )}
          </section>

          <div
            ref={cursorRef}
            className="handCursor"
            style={{
              opacity: 0,
              left: 0,
              top: 0
            }}
          >
            <div className="cursorCore" />

            <div className="cursorRing" />

            <div className="cursorLabel">
              POINT
            </div>
          </div>

          <div
            ref={objectRef}
            className="grabObject"
            style={{
              left: 0,
              top: 0
            }}
          />

          <footer className="bottomBar">
            <div>
              HANDS{" "}
              {frame.hands.length}
              {" / "}
              2
            </div>

            <div>
              {Math.round(
                frame.processingTime
              )}
              {"ms"}
            </div>

            <button
              onClick={() =>
                setShowDebug(
                  value =>
                    !value
                )
              }
            >
              {showDebug
                ? "HIDE DEBUG"
                : "DEBUG"}
            </button>

            <button
              onClick={
                stopCamera
              }
            >
              EXIT
            </button>
          </footer>

          {showDebug && (
            <DebugPanel
              frame={frame}
              diagnostics={
                diagnostics
              }
            />
          )}
        </>
      )}

      {status === "error" && (
        <section className="errorScreen">
          <div className="errorCode">
            TRACKING ERROR
          </div>

          <h2>
            THE SYSTEM
            <br />
            COULDN'T START.
          </h2>

          <pre>
            {error}
          </pre>

          <button
            onClick={() => {
              setError("");
              setStatus("idle");
            }}
          >
            TRY AGAIN
          </button>
        </section>
      )}
    </main>
  );
}

function DebugPanel({
  frame,
  diagnostics
}: {
  frame: GestureFrame;
  diagnostics: Diagnostics;
}) {
  const hand =
    frame.primaryHand;

  return (
    <aside className="debug">
      <div className="debugTitle">
        YUAKE / VISION
      </div>

      <div className="debugGrid">
        <div className="debugStat">
          <span>DETECTED</span>
          <strong>
            {frame.detected
              ? "YES"
              : "NO"}
          </strong>
        </div>

        <div className="debugStat">
          <span>HANDS</span>
          <strong>
            {frame.hands.length}
          </strong>
        </div>

        <div className="debugStat">
          <span>POSE</span>
          <strong>
            {hand?.pose ??
              "none"}
          </strong>
        </div>

        <div className="debugStat">
          <span>PINCH</span>
          <strong>
            {hand?.pinch
              ? "ACTIVE"
              : "OFF"}
          </strong>
        </div>

        <div className="debugStat">
          <span>STRENGTH</span>
          <strong>
            {Math.round(
              (
                hand?.pinchStrength ??
                0
              ) * 100
            )}
            %
          </strong>
        </div>

        <div className="debugStat">
          <span>SPEED</span>
          <strong>
            {(
              hand?.speed ??
              0
            ).toFixed(2)}
          </strong>
        </div>

        <div className="debugStat">
          <span>CONF</span>
          <strong>
            {Math.round(
              (
                hand?.confidence ??
                0
              ) * 100
            )}
            %
          </strong>
        </div>
      </div>

      <div className="debugDivider" />

      <div className="debugTitle">
        PIPELINE
      </div>

      <div className="debugGrid">
        <div className="debugStat">
          <span>CAM FPS</span>
          <strong>
            {diagnostics.cameraFps.toFixed(
              1
            )}
          </strong>
        </div>

        <div className="debugStat">
          <span>TRACK FPS</span>
          <strong>
            {diagnostics.trackFps.toFixed(
              1
            )}
          </strong>
        </div>

        <div className="debugStat">
          <span>INFER</span>
          <strong>
            {diagnostics.inferenceMs.toFixed(
              1
            )}
            ms
          </strong>
        </div>

        <div className="debugStat">
          <span>LOOP</span>
          <strong>
            {diagnostics.loopMs.toFixed(
              1
            )}
            ms
          </strong>
        </div>

        <div className="debugStat">
          <span>CALLBACK</span>
          <strong>
            {diagnostics.callbackDelayMs.toFixed(
              1
            )}
            ms
          </strong>
        </div>

        <div className="debugStat">
          <span>DECODE</span>
          <strong>
            {diagnostics.decodeMs.toFixed(
              1
            )}
            ms
          </strong>
        </div>

        <div className="debugStat">
          <span>VIDEO DLY</span>
          <strong>
            {diagnostics.videoDelayMs.toFixed(
              1
            )}
            ms
          </strong>
        </div>

        <div className="debugStat">
          <span>JITTER</span>
          <strong>
            {diagnostics.jitter.toFixed(
              2
            )}
          </strong>
        </div>

        <div className="debugStat">
          <span>DROPPED</span>
          <strong>
            {diagnostics.droppedFrames}
          </strong>
        </div>

        <div className="debugStat">
          <span>FRAME</span>
          <strong>
            {diagnostics.frameNumber}
          </strong>
        </div>

        <div className="debugStat debugStatWide">
          <span>RES</span>
          <strong>
            {diagnostics.resolution}
          </strong>
        </div>
      </div>
    </aside>
  );
}
