import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,

  /*
   * IMPORTANT:
   *
   * MediaPipe's tasks-vision WASM engine can run
   * MULTI-THREADED (via SharedArrayBuffer), which
   * is dramatically faster for hand-landmark
   * inference than the single-threaded fallback.
   *
   * Browsers only allow SharedArrayBuffer on pages
   * that are "cross-origin isolated" - which
   * requires these two headers. Without them,
   * MediaPipe silently falls back to single-
   * threaded WASM with no error or warning, and
   * inference can be several times slower - this
   * is almost certainly why tracking has felt like
   * a low, laggy frame rate no matter how the
   * smoothing/prediction code is tuned.
   */
  async headers() {
    return [
      {
        source: "/(.*)",

        headers: [
          {
            key:
              "Cross-Origin-Opener-Policy",

            value: "same-origin"
          },

          {
            key:
              "Cross-Origin-Embedder-Policy",

            value: "credentialless"
          }
        ]
      }
    ];
  }
};

export default nextConfig;
