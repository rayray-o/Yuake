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
    useRef<HTMLVideoElement>(null);

  const cursorRef =
    useRef<HTMLDivElement>(null);

  const streamRef =
    useRef<MediaStream | null>(null);

  const trackerRef =
    useRef<YuakeGestureTracker | null>(null);

  const animationRef =
    useRef<number | null>(null);

  const runningRef =
    useRef(false);

  const lastUiUpdateRef =
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

  const updateCursor =
    useCallback(
      (hand: GestureHand | null) => {
        const cursor =
          cursorRef.current;

        if (!cursor) {
          return;
        }

        if (!hand) {
          cursor.style.opacity = "0";
          return;
        }

        /*
         * The tracker now gives us the
         * actual fingertip position.
         *
         * Move the DOM element directly.
         * React does NOT need to render
         * another component for every frame.
         */
        cursor.style.left =
          `${hand.cursor.x * 100}%`;

        cursor.style.top =
          `${hand.cursor.y * 100}%`;

        cursor.style.opacity = "1";

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
      },
      []
    );

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
        .forEach(track => {
          track.stop();
        });

      streamRef.current =
        null;

      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.srcObject =
          null;
      }

      updateCursor(null);

      setFrame(
        EMPTY_FRAME
      );

      setStatus("idle");
    }, [updateCursor]);

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

        try {
          /*
           * Process every animation frame.
           *
           * The old code intentionally
           * discarded frames to stay around
           * 30 FPS. That creates visible
           * tracking latency.
           */
          const result =
            tracker.process(
              video,
              timestamp
            );

          /*
           * Critical:
           *
           * Cursor movement happens
           * immediately and independently
           * from React state.
           */
          updateCursor(
            result.primaryHand
          );

          /*
           * UI/debug information does NOT
           * need 60+ React renders per second.
           *
           * Keep it around 12 FPS.
           */
          if (
            timestamp -
              lastUiUpdateRef.current >=
            80
          ) {
            lastUiUpdateRef.current =
              timestamp;

            setFrame(result);
          }
        } catch (err) {
          console.error(err);

          setError(
            err instanceof Error
              ? err.message
              : String(err)
          );

          setStatus("error");

          stopCamera();

          return;
        }

        animationRef.current =
          requestAnimationFrame(
            processLoop
          );
      },
      [
        stopCamera,
        updateCursor
      ]
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
        setStatus("starting");

        try {
          if (
            !navigator.mediaDevices?.getUserMedia
          ) {
            throw new Error(
              "Camera access is not supported by this browser."
            );
          }

          const stream =
            await navigator.mediaDevices.getUserMedia(
              {
                video: {
                  /*
                   * REAR CAMERA
                   */
                  facingMode: {
                    ideal:
                      "environment"
                  },

                  width: {
                    ideal: 1920
                  },

                  height: {
                    ideal: 1080
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
           * Rear camera should not
           * be horizontally mirrored.
           */
          video.style.transform =
            "none";

          await video.play();

          const tracker =
            new YuakeGestureTracker();

          trackerRef.current =
            tracker;

          await tracker.initialize();

          runningRef.current =
            true;

          lastUiUpdateRef.current =
            performance.now();

          setStatus("live");

          animationRef.current =
            requestAnimationFrame(
              processLoop
            );
        } catch (err) {
          console.error(err);

          runningRef.current =
            false;

          streamRef.current
            ?.getTracks()
            .forEach(track => {
              track.stop();
            });

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
      [processLoop]
    );

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

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
              opacity: 0
            }}
          >
            <div className="cursorCore" />

            <div className="cursorRing" />

            <div className="cursorLabel">
              POINT
            </div>
          </div>

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
  frame
}: {
  frame: GestureFrame;
}) {
  const hand =
    frame.primaryHand;

  return (
    <aside className="debug">
      <div className="debugTitle">
        YUAKE / VISION
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
