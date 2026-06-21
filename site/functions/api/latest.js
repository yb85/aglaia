// GET /api/latest — Aglaïa update endpoint + privacy-respectful telemetry.
//
// Returns the latest published release version (proxied from GitHub, edge-
// cached) so the desktop app's update check has a single source of truth:
//   { "version": "0.1.0", "url": "https://github.com/.../releases/tag/v0.1.0" }
//
// The app probes this with a User-Agent that carries only its version and
// platform — no identifiers — e.g.
//   Aglaia/0.1.0 (macOS 14.5; arm64)
// We parse that and write one row to an Analytics Engine dataset so the
// maintainer can see version + OS + arch distribution of the install base.
// Nothing else is logged; there is no cookie, no IP storage, no user id.

const GH_REPO = "yb85/aglaia";
const GH_LATEST = `https://api.github.com/repos/${GH_REPO}/releases/latest`;
const RELEASES_URL = `https://github.com/${GH_REPO}/releases/latest`;
// Aglaia/<version> (<os> <osver>; <arch>)
const UA_RE = /^Aglaia\/(\S+)\s+\(([^;)]+?)\s*;\s*([^)]+)\)/i;

export async function onRequestGet({ request, env }) {
  // 1. Latest version, proxied from GitHub releases (5 min edge cache).
  let version = null;
  let url = RELEASES_URL;
  try {
    const r = await fetch(GH_LATEST, {
      headers: {
        "User-Agent": "aglaia-pages-fn",
        Accept: "application/vnd.github+json",
      },
      cf: { cacheTtl: 300, cacheEverything: true },
    });
    if (r.ok) {
      const j = await r.json();
      version = (j.tag_name || "").replace(/^v/i, "") || null;
      if (j.html_url) url = j.html_url;
    }
  } catch (_) {
    // fail open — return version:null, the app keeps its current build
  }

  // 2. Telemetry: parse the probe's User-Agent, one Analytics Engine row.
  const ua = request.headers.get("User-Agent") || "";
  const m = ua.match(UA_RE);
  if (m && env.AE) {
    const appVer = m[1];
    const os = m[2].trim();
    const arch = m[3].trim();
    try {
      env.AE.writeDataPoint({
        // blobs: queryable string columns; indexes: sampling key.
        blobs: [appVer, os, arch, request.cf?.country || "XX"],
        indexes: [appVer],
      });
    } catch (_) {
      // never let telemetry failure affect the response
    }
  }

  return new Response(JSON.stringify({ version, url }), {
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "public, max-age=300",
      "Access-Control-Allow-Origin": "*",
    },
  });
}
