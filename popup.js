const DEFAULTS = { enabled: true, targetFps: 120, quality: "neural", hud: false };
const $ = (id) => document.getElementById(id);

let activeFrameId = null;
let settings = { ...DEFAULTS };

async function activeTabId() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab?.id;
}

function showPower() {
  const on = settings.enabled !== false;
  $("enabled").setAttribute("aria-pressed", String(on));
  $("enabled").textContent = on ? "补帧已开启" : "补帧已关闭";
}

function showSettings() {
  showPower();
  $("hud").checked = settings.hud;
  document.querySelector(`input[name=targetFps][value="${settings.targetFps}"]`).checked = true;
  document.querySelector(`input[name=quality][value="${settings.quality}"]`).checked = true;
}

const FLOW_NAMES = { high: "精细", medium: "均衡", low: "省电" };

// The quality setting says what was asked for; this line says what is actually running, which is
// what tells apart AI frames from the optical flow fallback when the GPU is busy.
function showEngine(status) {
  const line = $("engine");
  const reason = $("engine-note");
  let kind = "";
  let text = "";
  if (status?.state === "off") {
    // Nothing is interpolating, so no engine is worth naming.
  } else if (status?.engine === "neural") {
    kind = "ai";
    if (status.state === "loading") text = "AI 补帧：正在加载模型";
    else if (status.state === "audioWait") text = "AI 补帧：等你点一下视频页面";
    else text = `AI 补帧生效中${status.model ? ` · ${status.model.replace(/^AI /, "")}` : ""}`;
  } else if (status?.engine === "flow") {
    kind = settings.quality === "neural" ? "fallback" : "flow";
    text = kind === "fallback" ? "AI 未生效，正在用光流补帧" : `光流补帧（${FLOW_NAMES[settings.quality] ?? "精细"}）`;
  }
  line.hidden = !text;
  line.textContent = text;
  line.dataset.kind = kind;
  reason.hidden = kind !== "fallback" || !status?.note;
  reason.textContent = status?.note ? `原因：${status.note}` : "";
}

// Players often live in iframes, so ask every frame and report the largest active video.
async function pickStatus(tabId) {
  const results = await chrome.scripting.executeScript({
    target: { tabId, allFrames: true },
    func: () => globalThis.__frameBoost?.status() ?? null,
  });
  let best = null;
  for (const { frameId, result } of results) {
    if (result && (!best || result.area > best.status.area)) best = { frameId, status: result };
  }
  return best;
}

async function refresh() {
  let picked = null;
  let failure = "";
  try {
    const tabId = await activeTabId();
    picked = tabId ? await pickStatus(tabId) : null;
  } catch {
    failure = "这个页面不允许扩展运行（例如 edge:// 设置页或扩展商店）";
  }
  const status = picked?.status;
  activeFrameId = picked?.frameId ?? null;

  const running = status?.state === "running";
  showEngine(status);
  $("rate").hidden = !running;
  $("meta").hidden = !status?.width;
  $("reload-cors").hidden = !status?.canReloadCors;
  $("message").textContent = failure || status?.message || "刷新页面后才能在已打开的标签页生效";
  if (running) $("rate").textContent = `${status.sourceFps.toFixed(0)} → ${status.outputFps} fps`;
  if (status?.width) {
    $("meta").textContent = `${status.width}×${status.height} · 屏幕 ${status.refreshRate} Hz${status.model ? ` · ${status.model}` : ""}`;
  }
  const capped = status?.refreshRate && settings.targetFps > status.refreshRate;
  const raised = running && settings.targetFps && status.outputFps > settings.targetFps * 1.08;
  $("fps-note").hidden = !capped && !raised;
  if (capped) $("fps-note").textContent = `屏幕刷新率是 ${status.refreshRate} Hz，实际最高只能显示 ${status.refreshRate} fps。`;
  if (raised) $("fps-note").textContent = `输出取屏幕 ${status.refreshRate} Hz 的整除帧率以保持匀速，所以当前是 ${status.outputFps} fps。`;
}

function save(partial) {
  settings = { ...settings, ...partial };
  chrome.storage.sync.set(partial);
}

$("enabled").addEventListener("click", () => {
  save({ enabled: !settings.enabled });
  showPower();
  refresh();
});
// The shortcut can switch it while the popup is open.
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "sync" || !changes.enabled) return;
  settings.enabled = changes.enabled.newValue;
  showPower();
});
$("hud").addEventListener("change", (event) => save({ hud: event.target.checked }));
document.querySelectorAll("input[name=targetFps]").forEach((input) => {
  input.addEventListener("change", () => save({ targetFps: Number(input.value) }));
});
document.querySelectorAll("input[name=quality]").forEach((input) => {
  input.addEventListener("change", () => save({ quality: input.value }));
});
$("reload-cors").addEventListener("click", async () => {
  const tabId = await activeTabId();
  if (tabId === undefined || activeFrameId === null) return;
  await chrome.scripting.executeScript({
    target: { tabId, frameIds: [activeFrameId] },
    func: () => globalThis.__frameBoost?.reloadWithCors(),
  });
  refresh();
});

chrome.storage.sync.get(DEFAULTS).then((stored) => {
  settings = stored;
  showSettings();
  refresh();
  setInterval(refresh, 1000);
});
