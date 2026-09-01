'use strict';

const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const MAIN = fs.readFileSync(path.join(__dirname, 'main.js'), 'utf8');

function handlerBody(channel) {
  const start = MAIN.indexOf(`ipcMain.handle('${channel}'`);
  assert.notStrictEqual(start, -1, `no ipcMain.handle('${channel}') found in main.js`);
  const next = MAIN.indexOf('ipcMain.handle(', start + 1);
  return MAIN.slice(start, next === -1 ? MAIN.length : next);
}

test('first-run Ollama setup cannot silently select Apple Intelligence', () => {
  const setup = handlerBody('setup-ollama-and-model');

  assert.doesNotMatch(
    setup,
    /\['set-model',\s*'apple:system'\]|apple_intelligence/,
    'Apple Intelligence must remain an explicit Settings choice',
  );
});

test('re-running setup preserves an explicit Apple Intelligence choice', () => {
  const setup = handlerBody('setup-ollama-and-model');
  const currentModelCheck = setup.indexOf("['get-model']");
  const ollamaResolution = setup.indexOf("['resolve-setup-model']");

  assert.notStrictEqual(currentModelCheck, -1, 'setup must inspect the current model');
  assert.ok(
    currentModelCheck < ollamaResolution,
    'the explicit Apple choice must be checked before Ollama model resolution',
  );
  assert.match(setup, /current\.model === 'apple:system'/);
  assert.match(setup, /skipped:\s*true/);
});
