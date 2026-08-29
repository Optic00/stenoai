'use strict';

const { normalizeOpenAiAsrApiUrl } = require('./openai-asr-key-store');

// Keep the sensitive endpoint argument construction outside main.js so it can
// be exercised without Electron. The URL is validated before any backend
// process is spawned: query strings, fragments, and userinfo can carry secrets
// and must never briefly appear in a local process listing.
function registerOpenAiAsrIpc({
  ipcMain,
  runPythonScript,
  migrateLegacyOpenAiAsrApiKey,
  hasOpenAiAsrKey,
}) {
  ipcMain.handle('set-openai-asr-config', async (_event, cfg) => {
    const args = ['set-openai-asr-config'];
    const updatesEndpoint = cfg && cfg.api_url !== undefined;
    if (updatesEndpoint) {
      const apiUrl = normalizeOpenAiAsrApiUrl(cfg.api_url);
      if (!apiUrl) {
        return { success: false, error: 'OpenAI ASR endpoint is invalid' };
      }
      args.push('--api-url', apiUrl);
    }
    if (cfg && cfg.model !== undefined) args.push('--model', cfg.model);

    try {
      const migrated = await migrateLegacyOpenAiAsrApiKey();
      // A surviving plaintext snapshot is bound to its old endpoint. Do not
      // commit a new endpoint while cleanup failed, or a later migration
      // could associate that credential with the replacement origin.
      if (updatesEndpoint && !migrated) {
        return { success: false, error: 'OpenAI ASR credential migration is incomplete' };
      }
      const result = await runPythonScript('simple_recorder.py', args, true);
      const jsonData = JSON.parse(result.trim());
      jsonData.api_key_set = hasOpenAiAsrKey();
      return jsonData;
    } catch (e) { return { success: false, error: e.message }; }
  });
}

module.exports = { registerOpenAiAsrIpc };
