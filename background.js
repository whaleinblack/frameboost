// Tab-scoped CORS header for media, only after the user asks to reload a blocked video.
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== "allow-media-cors" || !sender.tab) return;
  const tabId = sender.tab.id;
  chrome.declarativeNetRequest
    .updateSessionRules({
      removeRuleIds: [tabId],
      addRules: [
        {
          id: tabId,
          condition: { tabIds: [tabId], resourceTypes: ["media"] },
          action: {
            type: "modifyHeaders",
            responseHeaders: [{ header: "access-control-allow-origin", operation: "set", value: "*" }],
          },
        },
      ],
    })
    .then(() => sendResponse({ ok: true }), (error) => sendResponse({ ok: false, error: error.message }));
  return true;
});

chrome.tabs.onRemoved.addListener((tabId) => {
  chrome.declarativeNetRequest.updateSessionRules({ removeRuleIds: [tabId] });
});

chrome.commands.onCommand.addListener(async (command) => {
  if (command !== "toggle-boost") return;
  const { enabled = true } = await chrome.storage.sync.get("enabled");
  await chrome.storage.sync.set({ enabled: !enabled });
});
