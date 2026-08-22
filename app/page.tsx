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

  const streamRef =
    useRef<MediaStream | null>(null);

  const trackerRef =
    useRef<YuakeGestureTracker | null>(null);

  const animationRef =
    useRef<number | null>(null);

  const videoFrameCallbackRef =
    useRef<number | null>(null);

  const runningRef =
    useRef(false);

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
      runningRef.current = false;

      if (
        animationRef.current !== null
      ) {
        cancelAnimationFrame(
          animationRef.current
        );

        animationRef.current = null;
      }

      const video = videoRef.current;

      if (
        video &&
        videoFrameCallbackRef.current !== null &&
        "cancelVideoFrameCallback" in video
      ) {
        try {
          (
            video as HTMLVideoElement & {
              cancelVideoFrameCallback: (
                handle: number
              ) => void;
            }
          ).cancelVideoFrameCallback(
            videoFrameCallbackRef.current
          );
        } catch {
          // Some browsers may not expose cancellation.
        }
      }

      videoFrameCallbackRef.current =
        null;

      trackerRef.current?.close();

      trackerRef.current = null;

      streamRef.current
        ?.getTracks()
        .forEach(track => {
          track.stop();
        });

      streamRef.current = null;

      if (video) {
        video.pause();
        video.srcObject = null;
      }

      setFrame(EMPTY_FRAME);
      setStatus("idle");
    }, []);

  const processFrame =
    useCallback(
      (timestamp: number) => {
        if (!runningRef.current) {
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
          return;
        }

        try {
          const result =
            tracker.process(
              video,
              timestamp
            );

          setFrame(result);
        } catch (err) {
          console.error(err);

          setError(
            err instanceof Error
              ? err.message
              : String(err)
          );

          runningRef.current = false;

          setStatus("error");
        }
      },
      []
    );

  const scheduleVideoFrame =
    useCallback(() => {
      if (!runningRef.current) {
        return;
      }

      const video =
        videoRef.current;

      if (!video) {
        return;
      }

      const extendedVideo =
        video as HTMLVideoElement & {
          requestVideoFrameCallback?: (
            callback: (
              now: number,
              metadata: VideoFrameCallbackMetadata
            ) => void
          ) => number;
        };

      if (
        typeof extendedVideo.requestVideoFrameCallback ===
        "function"
      ) {
        videoFrameCallbackRef.current =
          extendedVideo.requestVideoFrameCallback(
            (now) => {
              if (!runningRef.current) {
                return;
              }

              processFrame(now);

              scheduleVideoFrame();
            }
          );

        return;
      }

      animationRef.current =
        requestAnimationFrame(
          now => {
            if (!runningRef.current) {
              return;
            }

            processFrame(now);
            scheduleVideoFrame();
          }
        );
    }, [processFrame]);

  const startCamera =
    useCallback(
      async () => {
        if (runningRef.current) {
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

          /*
           * REAR CAMERA
           *
           * `environment` asks the browser
           * for the outward-facing camera.
           *
           * No mirror is applied to the
           * actual camera feed.
           */
          const stream =
            await navigator.mediaDevices.getUserMedia(
              {
                video: {
                  facingMode: {
                    ideal: "environment"
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

          video.srcObject = stream;

          /*
           * Rear camera footage should NOT
           * be horizontally mirrored.
           *
           * This inline style intentionally
           * overrides the old CSS mirror.
           */
          video.style.transform =
            "none";

          await video.play();

          const tracker =
            new YuakeGestureTracker();

          trackerRef.current =
            tracker;

          await tracker.initialize();

          runningRef.current = true;

          setStatus("live");

          scheduleVideoFrame();
        } catch (err) {
          console.error(err);

          runningRef.current = false;

          streamRef.current
            ?.getTracks()
            .forEach(track => {
              track.stop();
            });

          streamRef.current = null;

          if (videoRef.current) {
            videoRef.current.srcObject =
              null;

            videoRef.current.style.transform =
              "none";
          }

          trackerRef.current?.close();
          trackerRef.current = null;

          setError(
            err instanceof Error
              ? err.message
              : String(err)
          );

          setStatus("error");
        }
      },
      [scheduleVideoFrame]
    );

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  const primary =
    frame.primaryHand;

  const gestureLabel =
    primary?.pose ?? "none";

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
            onClick={startCamera}
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
              hand={primary}
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
                  value => !value
                )
              }
            >
              {showDebug
                ? "HIDE DEBUG"
                : "DEBUG"}
            </button>

            <button
              onClick={stopCamera}
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
