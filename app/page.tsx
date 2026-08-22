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

type VideoWithFrameCallback =
  HTMLVideoElement & {
    requestVideoFrameCallback?: (
      callback: (
        now: number,
        metadata: {
          mediaTime: number;
          presentedFrames: number;
        }
      ) => void
    ) => number;

    cancelVideoFrameCallback?: (
      handle: number
    ) => void;
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
    useRef<number | null>(
      null
    );

  const videoCallbackRef =
    useRef<number | null>(
      null
    );

  const lastPresentedFrame =
    useRef(-1);

  const lastUiUpdate =
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

  /*
   * --------------------------------------------------
   * DIRECT CURSOR RENDERER
   * --------------------------------------------------
   *
   * This function never uses React state.
   *
   * It writes directly to the cursor.
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
         * Use transform rather than
         * left/top so the browser can
         * move the cursor on the compositor.
         */
        cursor.style.transform =
          `translate3d(` +
          `${hand.cursor.x * 100}vw,` +
          `${hand.cursor.y * 100}vh,` +
          `0)`;

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
      },
      []
    );

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

      updateCursor(null);

      setFrame(
        EMPTY_FRAME
      );

      setStatus("idle");
    }, [updateCursor]);

  /*
   * UI update is deliberately throttled.
   *
   * The cursor itself NEVER goes through
   * this state update.
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
      },
      []
    );

  /*
   * --------------------------------------------------
   * CAMERA FRAME LOOP
   * --------------------------------------------------
   */
  const processFrame =
    useCallback(
      (
        timestamp: number,
        presentedFrames?: number
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

        /*
         * Prevent processing the exact
         * same camera frame twice.
         */
        if (
          presentedFrames !==
            undefined &&
          presentedFrames ===
            lastPresentedFrame.current
        ) {
          return;
        }

        if (
          presentedFrames !==
          undefined
        ) {
          lastPresentedFrame.current =
            presentedFrames;
        }

        try {
          /*
           * Detect using the timestamp
           * belonging to this actual frame.
           */
          const result =
            tracker.process(
              video,
              timestamp
            );

          /*
           * THIS is the hot path.
           *
           * No React render.
           * No setState.
           * No waiting.
           */
          updateCursor(
            result.primaryHand
          );

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
   * The callback must be installed
   * after the video has begun playing.
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

      if (
        !video
      ) {
        return;
      }

      /*
       * Preferred path:
       *
       * browser tells us when a new
       * camera frame has actually been
       * presented.
       */
      if (
        video.requestVideoFrameCallback
      ) {
        const callback =
          (
            now: number,
            metadata: {
              mediaTime: number;
              presentedFrames: number;
            }
          ) => {
            if (
              !runningRef.current
            ) {
              return;
            }

            processFrame(
              now,
              metadata.presentedFrames
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

      /*
       * Older browser fallback.
       */
      scheduleFallback();
    }, [
      processFrame,
      scheduleFallback
    ]);

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
           * Rear camera:
           * do NOT mirror it.
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

          lastPresentedFrame.current =
            -1;

          lastUiUpdate.current =
            performance.now();

          setStatus("live");

          startFrameLoop();
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
      [startFrameLoop]
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
