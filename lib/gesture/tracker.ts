import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult
} from "@mediapipe/tasks-vision";

import {
  DETECTION_CONFIDENCE,
  HAND_LOST_GRACE_MS,
  HAND_MODEL_URL,
  MAX_HANDS,
  MEDIAPIPE_WASM_URL,
  PRESENCE_CONFIDENCE,
  TRACKING_CONFIDENCE
} from "./constants";

import {
  calculatePalm,
  calculatePalmNormal
} from "./geometry";

import {
  OneEuroPointFilter
} from "./oneEuro";

import {
  GestureClassifier
} from "./classify";

import type {
  GestureFrame,
  GestureHand,
  Handedness,
  Point3D
} from "./types";

type Point2D = {
  x: number;
  y: number;
};

type RuntimeState = {
  filter: OneEuroPointFilter;
  classifier: GestureClassifier;

  previousCursor: Point2D;
  previousTime: number;

  lastSeen: number;
};

const INDEX_TIP = 8;

/*
 * IMPORTANT:
 *
 * These constants assume the filter operates on
 * PIXEL coordinates, not normalized 0..1 values.
 *
 * beta is a speed-adaptive term: it lowers the
 * filter's smoothing (cutoff goes up) in proportion
 * to how fast the tracked point is moving, so the
 * cursor "catches up" during fast motion instead of
 * lagging behind at a constant smoothing strength.
 *
 * That only works if beta is scaled to match the
 * units of the derivative. On a 0..1 normalized
 * scale, finger velocity is a tiny number
 * (e.g. ~2 "screens per second" during a fast swipe),
 * so a pixel-tuned beta like 0.01 barely moves the
 * cutoff at all -> the filter behaves like a constant
 * low-pass filter regardless of speed -> the cursor
 * always trails behind instead of locking on.
 *
 * Fixing this by filtering in PIXEL space (see
 * convertResult below) lets us use standard,
 * well-tested 1-euro constants.
 */
const FILTER_MIN_CUTOFF = 0.7;
const FILTER_BETA = 0.04;
const FILTER_D_CUTOFF = 1.0;

/*
 * Keep the fast mobile processing pipeline.
 *
 * The visible camera remains full resolution.
 * MediaPipe receives a smaller same-aspect-ratio
 * processing frame.
 */
/*
 * --------------------------------------------------
 * ADAPTIVE PROCESSING RESOLUTION
 * --------------------------------------------------
 *
 * A fixed processing width means a fast device
 * (e.g. iPhone-class GPU) never gets to use its
 * spare headroom for better precision, and a slow
 * device (e.g. budget Android GPU) is stuck paying
 * a cost it can't afford - it just runs at a low,
 * laggy effective frame rate forever.
 *
 * Instead, the tracker watches its OWN measured
 * inference time and automatically steps the
 * processing resolution up or down to converge on
 * a target inference budget. Same code, same
 * targets, for every device - it just calibrates
 * itself to whatever hardware it's actually running
 * on, live, with no per-device special casing.
 */
const INITIAL_PROCESSING_WIDTH = 320;
const MIN_PROCESSING_WIDTH = 176;
const MAX_PROCESSING_WIDTH_CAP = 480;

/*
 * If average inference time drifts above this,
 * the device is struggling - shrink the input.
 */
const TARGET_INFER_MS_HIGH = 45;

/*
 * If average inference time is comfortably below
 * this, the device has spare headroom - grow the
 * input for better precision.
 */
const TARGET_INFER_MS_LOW = 18;

const PROCESSING_WIDTH_STEP = 32;

/*
 * Reassess every N frames rather than every single
 * frame, so a couple of noisy samples don't cause
 * the resolution to flicker up and down.
 */
const ADAPT_SAMPLE_COUNT = 20;

let cachedViewport:
  | {
      width: number;
      height: number;
      left: number;
      top: number;
      time: number;
    }
  | null = null;

function clamp(
  value: number,
  min: number,
  max: number
) {
  return Math.max(
    min,
    Math.min(max, value)
  );
}

function clamp01(
  value: number
) {
  return clamp(
    value,
    0,
    1
  );
}

function toPoint3D(
  point: {
    x: number;
    y: number;
    z: number;
  }
): Point3D {
  return {
    x: Number.isFinite(point.x)
      ? point.x
      : 0,

    y: Number.isFinite(point.y)
      ? point.y
      : 0,

    z: Number.isFinite(point.z)
      ? point.z
      : 0
  };
}

function getHandedness(
  result: HandLandmarkerResult,
  index: number
): Handedness {
  const label =
    result.handednesses?.[
      index
    ]?.[0]?.categoryName;

  if (
    label === "Left" ||
    label === "Right"
  ) {
    return label;
  }

  return "Unknown";
}

/*
 * Get the actual DOM rectangle occupied by
 * the video element.
 *
 * IMPORTANT:
 * This is NOT the visible image rectangle.
 *
 * With object-fit: cover, the source image
 * is scaled until it completely covers this
 * rectangle and some source pixels are cropped.
 */
function getViewportRect(
  video: HTMLVideoElement
) {
  const now =
    performance.now();

  const element =
    video.getBoundingClientRect();

  if (
    element.width <= 0 ||
    element.height <= 0
  ) {
    return null;
  }

  if (
    cachedViewport &&
    now -
      cachedViewport.time <
      100 &&
    cachedViewport.width ===
      element.width &&
    cachedViewport.height ===
      element.height &&
    cachedViewport.left ===
      element.left &&
    cachedViewport.top ===
      element.top
  ) {
    return cachedViewport;
  }

  cachedViewport = {
    left: element.left,
    top: element.top,
    width: element.width,
    height: element.height,
    time: now
  };

  return cachedViewport;
}

/*
 * --------------------------------------------------
 * CORRECT OBJECT-FIT: COVER MAPPING
 * --------------------------------------------------
 *
 * MediaPipe gives normalized coordinates:
 *
 *   x = 0..1 across the COMPLETE camera frame
 *   y = 0..1 across the COMPLETE camera frame
 *
 * But the CSS camera uses:
 *
 *   object-fit: cover
 *
 * Therefore the complete camera frame is NOT
 * necessarily visible.
 *
 * We must reproduce the exact cover transform:
 *
 *     scale = max(
 *       viewportWidth / sourceWidth,
 *       viewportHeight / sourceHeight
 *     )
 *
 * Then determine how many source pixels are
 * cropped from each side.
 *
 * This is the important correction that keeps
 * the cursor physically attached to landmark 8.
 */
function cameraToViewport(
  video: HTMLVideoElement,
  point: Point3D
): Point2D {
  const viewport =
    getViewportRect(
      video
    );

  if (
    !viewport ||
    !video.videoWidth ||
    !video.videoHeight
  ) {
    return {
      x: clamp01(
        point.x
      ),

      y: clamp01(
        point.y
      )
    };
  }

  const sourceWidth =
    video.videoWidth;

  const sourceHeight =
    video.videoHeight;

  const viewportWidth =
    viewport.width;

  const viewportHeight =
    viewport.height;

  /*
   * Exact CSS object-fit: cover scale.
   */
  const scale =
    Math.max(
      viewportWidth /
        sourceWidth,

      viewportHeight /
        sourceHeight
    );

  /*
   * Scaled dimensions of the COMPLETE
   * camera image before cropping.
   */
  const scaledWidth =
    sourceWidth *
    scale;

  const scaledHeight =
    sourceHeight *
    scale;

  /*
   * object-position defaults to 50% 50%.
   *
   * Therefore the overflow is split equally
   * between the two sides.
   */
  const cropX =
    (
      scaledWidth -
      viewportWidth
    ) /
    2;

  const cropY =
    (
      scaledHeight -
      viewportHeight
    ) /
    2;

  /*
   * Convert normalized MediaPipe coordinates
   * back into source pixels.
   */
  const sourceX =
    clamp01(
      point.x
    ) *
    sourceWidth;

  const sourceY =
    clamp01(
      point.y
    ) *
    sourceHeight;

  /*
   * Scale source coordinates exactly like CSS.
   */
  const displayedX =
    sourceX *
      scale -
    cropX;

  const displayedY =
    sourceY *
      scale -
    cropY;

  /*
   * Add the DOM element's position.
   */
  const screenX =
    viewport.left +
    displayedX;

  const screenY =
    viewport.top +
    displayedY;

  return {
    x: clamp01(
      screenX /
        window.innerWidth
    ),

    y: clamp01(
      screenY /
        window.innerHeight
    )
  };
}

export class YuakeGestureTracker {
  private detector:
    HandLandmarker | null =
      null;

  private runtime =
    new Map<
      string,
      RuntimeState
    >();

  private initialized =
    false;

  private processingTime =
    0;

  private lastTimestamp =
    0;

  /*
   * Low-resolution processing canvas.
   *
   * This is never displayed.
   */
  private processingCanvas:
    HTMLCanvasElement | null =
      null;

  private processingContext:
    CanvasRenderingContext2D | null =
      null;

  private processingWidth =
    0;

  private processingHeight =
    0;

  /*
   * The CURRENT target width the adaptive
   * system has converged on for this device.
   * Starts at a neutral default and self-
   * adjusts from there.
   */
  private processingWidthTarget =
    INITIAL_PROCESSING_WIDTH;

  private inferenceSamples:
    number[] = [];

  private framesSinceAdapt =
    0;

  async initialize() {
    if (
      this.initialized
    ) {
      return;
    }

    const vision =
      await FilesetResolver.forVisionTasks(
        MEDIAPIPE_WASM_URL
      );

    this.detector =
      await HandLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              HAND_MODEL_URL,

            /*
             * TESTING: was "GPU".
             *
             * 200ms inference with cross-origin
             * isolation confirmed active suggests
             * GPU delegate overhead (frame
             * upload/readback) may be the actual
             * bottleneck on this device, not
             * threading. Now that multi-threaded
             * WASM is available, CPU delegate can
             * use multiple cores + SIMD - worth a
             * direct comparison.
             */
            delegate:
              "CPU"
          },

          runningMode:
            "VIDEO",

          numHands:
            MAX_HANDS,

          /*
           * These constants existed in
           * constants.ts but were never
           * actually passed here - the
           * detector was silently running
           * on generic defaults instead of
           * the app's tuned values.
           */
          minHandDetectionConfidence:
            DETECTION_CONFIDENCE,

          minHandPresenceConfidence:
            PRESENCE_CONFIDENCE,

          minTrackingConfidence:
            TRACKING_CONFIDENCE
        }
      );

    /*
     * Create the processing canvas once.
     */
    this.processingCanvas =
      document.createElement(
        "canvas"
      );

    this.processingContext =
      this.processingCanvas.getContext(
        "2d",
        {
          alpha: false,
          desynchronized: true
        }
      );

    if (
      !this.processingContext
    ) {
      throw new Error(
        "Unable to create hand-tracking processing canvas."
      );
    }

    this.processingContext.imageSmoothingEnabled =
      false;

    this.initialized =
      true;
  }

  /*
   * Build the small processing frame.
   *
   * Aspect ratio is preserved exactly.
   */
  private prepareProcessingFrame(
    video: HTMLVideoElement
  ) {
    if (
      !this.processingCanvas ||
      !this.processingContext
    ) {
      throw new Error(
        "Processing canvas is not initialized."
      );
    }

    const sourceWidth =
      video.videoWidth;

    const sourceHeight =
      video.videoHeight;

    if (
      !sourceWidth ||
      !sourceHeight
    ) {
      throw new Error(
        "Camera dimensions are unavailable."
      );
    }

    const scale =
      Math.min(
        1,

        this.processingWidthTarget /
          sourceWidth
      );

    const width =
      Math.max(
        1,
        Math.round(
          sourceWidth *
            scale
        )
      );

    const height =
      Math.max(
        1,
        Math.round(
          sourceHeight *
            scale
        )
      );

    if (
      width !==
        this.processingWidth ||
      height !==
        this.processingHeight
    ) {
      this.processingWidth =
        width;

      this.processingHeight =
        height;

      this.processingCanvas.width =
        width;

      this.processingCanvas.height =
        height;

      this.processingContext =
        this.processingCanvas.getContext(
          "2d",
          {
            alpha: false,
            desynchronized: true
          }
        );

      if (
        !this.processingContext
      ) {
        throw new Error(
          "Unable to recreate processing canvas."
        );
      }

      this.processingContext.imageSmoothingEnabled =
        false;
    }

    this.processingContext.drawImage(
      video,
      0,
      0,
      width,
      height
    );

    return this.processingCanvas;
  }

  process(
    video: HTMLVideoElement,
    timestamp: number
  ): GestureFrame {
    if (
      !this.detector
    ) {
      throw new Error(
        "Gesture tracker has not been initialized."
      );
    }

    const start =
      performance.now();

    const safeTimestamp =
      Math.max(
        timestamp,
        this.lastTimestamp +
          0.001
      );

    /*
     * IMPORTANT:
     *
     * MediaPipe receives the small processing
     * frame, NOT the full-resolution camera.
     *
     * This is what gave YUAKE the large
     * smoothness improvement.
     */
    const processingFrame =
      this.prepareProcessingFrame(
        video
      );

    const result =
      this.detector.detectForVideo(
        processingFrame,
        safeTimestamp
      );

    const hands =
      this.convertResult(
        result,
        video,
        safeTimestamp
      );

    this.lastTimestamp =
      safeTimestamp;

    this.processingTime =
      performance.now() -
      start;

    this.adaptProcessingWidth(
      this.processingTime
    );

    return {
      timestamp:
        safeTimestamp,

      detected:
        hands.length >
        0,

      hands,

      primaryHand:
        this.selectPrimaryHand(
          hands
        ),

      frameTime:
        safeTimestamp,

      processingTime:
        this.processingTime
    };
  }

  /*
   * Step the processing resolution up or down
   * to converge on a target inference budget,
   * based on this device's own measured
   * performance. Called once per processed
   * frame; only actually adjusts every
   * ADAPT_SAMPLE_COUNT frames.
   */
  private adaptProcessingWidth(
    latestInferenceMs: number
  ) {
    this.inferenceSamples.push(
      latestInferenceMs
    );

    if (
      this.inferenceSamples
        .length >
      ADAPT_SAMPLE_COUNT
    ) {
      this.inferenceSamples.shift();
    }

    this.framesSinceAdapt++;

    if (
      this.framesSinceAdapt <
        ADAPT_SAMPLE_COUNT ||
      this.inferenceSamples
        .length <
        ADAPT_SAMPLE_COUNT
    ) {
      return;
    }

    this.framesSinceAdapt =
      0;

    const average =
      this.inferenceSamples.reduce(
        (sum, value) =>
          sum + value,
        0
      ) /
      this.inferenceSamples
        .length;

    if (
      average >
        TARGET_INFER_MS_HIGH &&
      this.processingWidthTarget >
        MIN_PROCESSING_WIDTH
    ) {
      this.processingWidthTarget =
        Math.max(
          MIN_PROCESSING_WIDTH,

          this.processingWidthTarget -
            PROCESSING_WIDTH_STEP
        );

      return;
    }

    if (
      average <
        TARGET_INFER_MS_LOW &&
      this.processingWidthTarget <
        MAX_PROCESSING_WIDTH_CAP
    ) {
      this.processingWidthTarget =
        Math.min(
          MAX_PROCESSING_WIDTH_CAP,

          this.processingWidthTarget +
            PROCESSING_WIDTH_STEP
        );
    }
  }

  /*
   * Exposed for the debug panel, so the
   * adaptive behavior is actually visible
   * instead of invisible under the hood.
   */
  getProcessingWidth() {
    return this.processingWidthTarget;
  }

  private convertResult(
    result: HandLandmarkerResult,
    video: HTMLVideoElement,
    timestamp: number
  ): GestureHand[] {
    const hands:
      GestureHand[] = [];

    for (
      let index = 0;
      index <
        result.landmarks.length;
      index++
    ) {
      const raw =
        result.landmarks[
          index
        ];

      if (
        !raw ||
        raw.length <
          21
      ) {
        continue;
      }

      const landmarks =
        raw.map(
          toPoint3D
        );

      const worldRaw =
        result.worldLandmarks?.[
          index
        ];

      const worldLandmarks =
        worldRaw?.length === 21
          ? worldRaw.map(
              toPoint3D
            )
          : landmarks.map(
              point => ({
                ...point
              })
            );

      const handedness =
        getHandedness(
          result,
          index
        );

      const id =
        handedness ===
        "Unknown"
          ? `hand-${index}`
          : handedness;

      let runtime =
        this.runtime.get(
          id
        );

      if (!runtime) {
        runtime = {
          filter:
            new OneEuroPointFilter(
              {
                minCutoff:
                  FILTER_MIN_CUTOFF,

                beta:
                  FILTER_BETA,

                dCutoff:
                  FILTER_D_CUTOFF
              }
            ),

          classifier:
            new GestureClassifier(),

          previousCursor: {
            x: 0,
            y: 0
          },

          previousTime:
            timestamp,

          lastSeen:
            timestamp
        };

        this.runtime.set(
          id,
          runtime
        );
      }

      runtime.lastSeen =
        timestamp;

      /*
       * MediaPipe landmark 8 =
       * INDEX FINGERTIP.
       *
       * No offset.
       * No thumb midpoint.
       * No artificial correction.
       */
      const fingertip =
        landmarks[
          INDEX_TIP
        ];

      /*
       * Convert the exact fingertip through
       * the real object-fit: cover transform.
       */
      const rawCursor =
        cameraToViewport(
          video,
          fingertip
        );

      /*
       * Convert to pixel space before smoothing.
       *
       * The 1-euro filter's beta (speed adaptation)
       * term needs a coordinate scale where fast
       * motion produces a large derivative. Pixels
       * give us that; normalized 0..1 values do not.
       */
      const pixelCursor = {
        x:
          rawCursor.x *
          window.innerWidth,

        y:
          rawCursor.y *
          window.innerHeight
      };

      const filteredPixels =
        runtime.filter.filter(
          pixelCursor,
          timestamp /
            1000
        );

      /*
       * Convert back to normalized 0..1 for
       * rendering / downstream consumers.
       */
      const filtered = {
        x:
          filteredPixels.x /
          window.innerWidth,

        y:
          filteredPixels.y /
          window.innerHeight
      };

      /*
       * IMPORTANT:
       *
       * Do NOT compute velocity as a raw frame-
       * to-frame position difference. MediaPipe
       * landmark positions have natural per-frame
       * noise, and dividing that noise by a small
       * dt amplifies it into a very jittery
       * "velocity" signal.
       *
       * The 1-euro filter above already computes
       * a properly SMOOTHED derivative internally
       * (that's what beta uses to adapt the
       * cutoff). Reuse that clean signal instead
       * of computing our own from raw positions.
       * This matters a lot because this velocity
       * feeds the cursor predictor's between-frame
       * extrapolation - a noisy velocity there
       * shows up as visible jitter/vibration.
       */
      const pixelVelocity =
        runtime.filter.getVelocity();

      const velocity = {
        x:
          pixelVelocity.x /
          window.innerWidth,

        y:
          pixelVelocity.y /
          window.innerHeight
      };

      const speed =
        Math.hypot(
          velocity.x,
          velocity.y
        );

      /*
       * The cursor is the filtered fingertip.
       *
       * No additional interpolation is applied.
       */
      const cursor = {
        x: clamp01(
          filtered.x
        ),

        y: clamp01(
          filtered.y
        )
      };

      /*
       * IMPORTANT: pass WORLD landmarks, not
       * image-space landmarks.
       *
       * Image-space landmarks are 2D screen
       * position plus a rough, perspective-
       * distorted depth value. Measuring
       * thumb-to-index distance in that space
       * is only reliable when the hand faces
       * the camera fairly directly - rotate
       * the hand sideways and the same
       * physical gap projects to a very
       * different apparent distance.
       *
       * World landmarks are a real 3D
       * reconstruction of the hand in actual
       * physical units, independent of camera
       * angle - the same physical pinch
       * measures the same regardless of how
       * the hand is rotated toward the camera.
       */
      const classification =
        runtime.classifier.classify(
          worldLandmarks
        );

      const palm =
        calculatePalm(
          worldLandmarks
        );

      const palmNormal =
        calculatePalmNormal(
          worldLandmarks
        );

      const confidence =
        this.getConfidence(
          result,
          index
        );

      hands.push({
        id,

        handedness,

        landmarks,

        worldLandmarks,

        wrist:
          worldLandmarks[0],

        palm,

        palmNormal,

        cursor,

        velocity,

        speed,

        pinch:
          classification.pinch,

        pinchStrength:
          classification.pinchStrength,

        pose:
          classification.pose,

        fingers:
          classification.fingers,

        confidence,

        visible:
          true,

        lastSeen:
          timestamp
      });

      runtime.previousCursor =
        rawCursor;

      runtime.previousTime =
        timestamp;
    }

    this.removeLostHands(
      timestamp
    );

    return hands;
  }

  private getConfidence(
    result: HandLandmarkerResult,
    index: number
  ) {
    return clamp01(
      result.handednesses?.[
        index
      ]?.[0]?.score ??
        0.8
    );
  }

  private selectPrimaryHand(
    hands: GestureHand[]
  ) {
    if (
      hands.length ===
      0
    ) {
      return null;
    }

    const pinching =
      hands.find(
        hand =>
          hand.pinch
      );

    if (pinching) {
      return pinching;
    }

    return [...hands].sort(
      (a, b) =>
        b.confidence -
        a.confidence
    )[0];
  }

  private removeLostHands(
    timestamp: number
  ) {
    for (
      const [
        id,
        runtime
      ] of this.runtime
    ) {
      if (
        timestamp -
          runtime.lastSeen >
        HAND_LOST_GRACE_MS
      ) {
        runtime.filter.reset();

        runtime.classifier.reset();

        this.runtime.delete(
          id
        );
      }
    }
  }

  close() {
    this.detector?.close();

    this.detector =
      null;

    this.runtime.clear();

    this.initialized =
      false;

    this.processingTime =
      0;

    this.lastTimestamp =
      0;

    this.processingCanvas =
      null;

    this.processingContext =
      null;

    this.processingWidth =
      0;

    this.processingHeight =
      0;

    this.processingWidthTarget =
      INITIAL_PROCESSING_WIDTH;

    this.inferenceSamples =
      [];

    this.framesSinceAdapt =
      0;

    cachedViewport =
      null;
  }
  }
