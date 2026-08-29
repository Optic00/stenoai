'use strict';

const { test } = require('node:test');
const assert = require('node:assert');
const { registerOpenAiAsrIpc } = require('./openai-asr-ipc');

function harness() {
  const handlers = {};
  const calls = { migrate: 0, python: [] };
  registerOpenAiAsrIpc({
    ipcMain: { handle: (channel, handler) => { handlers[channel] = handler; } },
    migrateLegacyOpenAiAsrApiKey: async () => { calls.migrate += 1; },
    runPythonScript: async (script, args, silent) => {
      calls.python.push({ script, args, silent });
      return '{"success":true}';
    },
    hasOpenAiAsrKey: () => false,
  });
  return { handlers, calls };
}

test('unsafe OpenAI ASR endpoint never reaches legacy migration or backend argv', async () => {
  for (const apiUrl of [
    'https://user:secret@provider.example/v1',
    'https://provider.example/v1?sig=secret',
    'https://provider.example/v1#secret',
    'http://provider.example/v1',
  ]) {
    const { handlers, calls } = harness();
    const result = await handlers['set-openai-asr-config']({}, { api_url: apiUrl });
    assert.deepStrictEqual(result, { success: false, error: 'OpenAI ASR endpoint is invalid' });
    assert.strictEqual(calls.migrate, 0, 'unsafe URL must fail before any subprocess work');
    assert.deepStrictEqual(calls.python, [], 'unsafe URL must never be passed to the backend CLI');
  }
});

test('OpenAI ASR endpoint argv uses a safe canonical URL and retains /v1 paths', async () => {
  for (const [input, expected] of [
    [' HTTPS://API.EXAMPLE.COM:443/v1/ ', 'https://api.example.com/v1'],
    ['http://127.0.0.1:9000/v1/', 'http://127.0.0.1:9000/v1'],
  ]) {
    const { handlers, calls } = harness();
    const result = await handlers['set-openai-asr-config']({}, { api_url: input });
    assert.deepStrictEqual(result, { success: true, api_key_set: false });
    assert.strictEqual(calls.migrate, 1);
    assert.deepStrictEqual(calls.python, [{
      script: 'simple_recorder.py',
      args: ['set-openai-asr-config', '--api-url', expected],
      silent: true,
    }]);
  }
});
