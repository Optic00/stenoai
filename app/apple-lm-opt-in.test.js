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

test('setup fails closed when the current model cannot be read', () => {
  const setup = handlerBody('setup-ollama-and-model');
  const failureLog = setup.indexOf('Could not read current summary model:');
  const retryError = setup.indexOf('Please retry setup.', failureLog);

  assert.notStrictEqual(failureLog, -1);
  assert.ok(retryError > failureLog);
  assert.doesNotMatch(setup, /Could not read current summary model, proceeding/);
});

test('setup fails closed when the provider cannot be read', () => {
  const setup = handlerBody('setup-ollama-and-model');

  assert.match(setup, /Could not read AI provider:/);
  assert.match(setup, /Could not read the AI provider\. Please retry setup\./);
  assert.doesNotMatch(setup, /Could not read AI provider, proceeding/);
});

test('setup enforces bundled Ollama and atomically preserves newer choices', () => {
  const setup = handlerBody('setup-ollama-and-model');
  const bundledCheck = setup.indexOf('await findOllamaExecutable()');
  const ollamaResolution = setup.indexOf("['resolve-setup-model']");

  assert.notStrictEqual(bundledCheck, -1);
  assert.ok(
    bundledCheck < ollamaResolution,
    'installed-model resolution must not bypass the bundled binary requirement',
  );
  assert.match(setup, /\['set-model-if-current',\s*setupModelAtStart,/);
  assert.doesNotMatch(setup, /\['set-model',\s*resolved\.installed\]/);
});
