// Packs the committed extension files into the zip the Chrome Web Store takes. Tests, docs and
// development files stay out. Usage: npm run pack
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const { version } = JSON.parse(readFileSync(join(root, "manifest.json"), "utf8"));
const files = ["manifest.json", "background.js", "popup.html", "popup.css", "popup.js", "src", "icons", "models", "LICENSE"];
const status = execFileSync("git", ["status", "--porcelain", "--", ...files], { cwd: root, encoding: "utf8" });
if (status.trim()) {
  console.error("These extension files have uncommitted changes, and the zip would not include them:\n" + status);
  process.exit(1);
}
const out = `frameboost-${version}.zip`;
execFileSync("git", ["archive", "--format=zip", "-o", out, "HEAD", ...files], { cwd: root, stdio: "inherit" });
console.log(`${out} written`);
