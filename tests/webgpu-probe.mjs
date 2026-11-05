// Asks the installed Edge, with default flags, whether WebGPU gives a GPU adapter on this computer.
// Usage: node tests/webgpu-probe.mjs [chrome|msedge]
import { createServer } from "node:http";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright-core";

const channel = process.argv[2] ?? "msedge";
const server = createServer((req, res) => res.end("<!doctype html><title>probe</title>"));
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const profile = mkdtempSync(join(process.env.TMP ?? tmpdir(), "fb-probe-"));
const context = await chromium.launchPersistentContext(profile, { channel, headless: true });
try {
  const page = await context.newPage();
  await page.goto(`http://localhost:${server.address().port}/`);
  const result = await page.evaluate(async () => {
    const out = { userAgent: navigator.userAgent, secure: isSecureContext, hasGpu: !!navigator.gpu };
    if (!navigator.gpu) return out;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    out.adapter = !!adapter;
    if (adapter) {
      const info = adapter.info ?? (await adapter.requestAdapterInfo?.());
      out.vendor = info?.vendor;
      out.architecture = info?.architecture;
      out.description = info?.description;
      out.fallback = adapter.isFallbackAdapter ?? info?.isFallbackAdapter;
      out.features = [...adapter.features].filter((name) => name.includes("f16") || name.includes("timestamp"));
    }
    return out;
  });
  console.log(JSON.stringify(result, null, 2));
} finally {
  await context.close();
  server.close();
  rmSync(profile, { recursive: true, force: true });
}
