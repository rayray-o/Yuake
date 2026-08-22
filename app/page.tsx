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

import type {
  GestureFrame,
  GestureHand
} from "../lib/gesture/types";

type Status =
  | "idle"
  | "starting"
  | "live"
  | "error";

const EMPTY_FRAME: GestureFrame = {
  timestamp: 0,
  detected: false,
  hands: [],
  primaryHand: null,
  frameTime: 0,
  processingTime: 0
};

export default function Home() {
  const videoRef =
    useRef<HTMLVideoElement>(
      null
    );

  const streamRef =
    useRef<MediaStream | null>(
      null
    );

  const trackerRef =
    useRef<YuakeGestureTracker | null>(
      null
    );

  const animationRef =
    useRef<number | null>(
      null
    );

  const runningRef =
    useRef(false);

  const lastProcessRef =
    useRef(0);

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

  const stopCamera =
    useCallback(() => {
      runningRef.current =
        false;

      if (
        animationRef.current !==
        null
      ) {
        cancelAnimationFrame(
          animationRef.current
        );

        animationRef.current =
          null;
      }

      trackerRef.current?.close();

      trackerRef.current =
        null;

      streamRef.current
        ?.getTracks()
        .forEach(track =>
          track.stop()
        );

      streamRef.current =
        null;

      if (videoRef.current) {
        videoRef.current.srcObject =
          null;
      }

      setFrame(
        EMPTY_FRAME
      );

      setStatus("idle");
    }, []);

  const processLoop =
    useCallback(
      (timestamp: number) => {
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
          animationRef.current =
            requestAnimationFrame(
              processLoop
            );

          return;
        }

        /*
         * Keep processing around 30 FPS.
         *
         * Rendering can still run at
         * the device's native refresh rate.
         */
        if (
          timestamp -
            lastProcessRef.current >=
          33
        ) {
          lastProcessRef.current =
            timestamp;

          try {
            const result =
              tracker.process(
                video,
                timestamp
              );

            setFrame(
              result
            );
          } catch (err) {
            console.error(
              err
            );

            setError(
              err instanceof Error
                ? err.message
                : String(err)
            );

            setStatus(
              "error"
            );

            stopCamera();

            return;
          }
        }

        animationRef.current =
          requestAnimationFrame(
            processLoop
          );
      },
      [stopCamera]
    );

  const startCamera =
    useCallback(
      async () => {
        if (
          runningRef.current
        ) {
          return;
        }

        setError("");
        setStatus(
          "starting"
        );

        try {
          const stream =
            await navigator.mediaDevices.getUserMedia(
              {
                video: {
                  facingMode: {
                    ideal:
                      "user"
                  },

                  width: {
                    ideal:
                      1280
                  },

                  height: {
                    ideal:
                      720
                  },

                  frameRate: {
                    ideal:
                      30,

                    max:
                      30
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

          await video.play();

          const tracker =
            new YuakeGestureTracker();

          trackerRef.current =
            tracker;

          await tracker.initialize();

          runningRef.current =
            true;

          setStatus("live");

          lastProcessRef.current =
            performance.now();

          animationRef.current =
            requestAnimationFrame(
              processLoop
            );
        } catch (err) {
          console.error(
            err
          );

          streamRef.current
            ?.getTracks()
            .forEach(track =>
              track.stop()
            );

          streamRef.current =
            null;

          setError(
            err instanceof Error
              ? err.message
              : String(err)
          );

          setStatus(
            "error"
          );
        }
      },
      [processLoop]
    );

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  const primary =
    frame.primaryHand;

  const gestureLabel =
    primary?.pose ??
    "none";

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
          YUAKÉ
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
            : status ===
                "starting"
              ? "INITIALIZING"
              : status ===
                  "error"
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
            Your hand becomes the
            interface.
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
                  {gestureLabel.toUpperCase()}
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

          {primary && (
            <HandCursor
              hand={
                primary
              }
            />
          )}

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
              setStatus(
                "idle"
              );
            }}
          >
            TRY AGAIN
          </button>
        </section>
      )}
    </main>
  );
}

function HandCursor({
  hand
}: {
  hand: GestureHand;
}) {
  const style = {
    left: `${hand.cursor.x * 100}%`,
    top: `${hand.cursor.y * 100}%`
  };

  return (
    <div
      className={
        hand.pinch
          ? "handCursor pinch"
          : "handCursor"
      }
      style={style}
    >
      <div className="cursorCore" />

      <div className="cursorRing" />

      <div className="cursorLabel">
        {hand.pinch
          ? "GRAB"
          : hand.pose.toUpperCase()}
      </div>
    </div>
  );
}

function DebugPanel({
  frame
}: {
  frame: GestureFrame;
}) {
  const hand =
    frame.primaryHand;

  return (
    <aside className="debug">
      <div className="debugTitle">
        YUAKÉ / VISION
      </div>

      <div className="debugRow">
        <span>
          DETECTED
        </span>

        <strong>
          {frame.detected
            ? "YES"
            : "NO"}
        </strong>
      </div>

      <div className="debugRow">
        <span>
          HANDS
        </span>

        <strong>
          {frame.hands.length}
        </strong>
      </div>

      <div className="debugRow">
        <span>
          POSE
        </span>

        <strong>
          {hand?.pose ??
            "none"}
        </strong>
      </div>

      <div className="debugRow">
        <span>
          PINCH
        </span>

        <strong>
          {hand?.pinch
            ? "ACTIVE"
            : "OFF"}
        </strong>
      </div>

      <div className="debugRow">
        <span>
          STRENGTH
        </span>

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

      <div className="debugRow">
        <span>
          SPEED
        </span>

        <strong>
          {(
            hand?.speed ??
            0
          ).toFixed(2)}
        </strong>
      </div>

      <div className="debugRow">
        <span>
          CONFIDENCE
        </span>

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
    </aside>
  );
}
