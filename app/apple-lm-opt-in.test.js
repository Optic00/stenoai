'use strict';

const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const { assertOllamaSetupModel, modelSetupSaveError } = require('./model-setup-guard');

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
  assert.throws(
    () => assertOllamaSetupModel('apple:system'),
    /cannot select Apple Intelligence/,
  );
  assert.strictEqual(
    assertOllamaSetupModel('gemma4:e2b-it-qat'),
    'gemma4:e2b-it-qat',
  );
  assert.strictEqual(
    (setup.match(/setOllamaSetupModelIfCurrent\(/g) || []).length,
    2,
    'both setup persistence paths must use the guarded compare-and-set helper',
  );
  assert.doesNotMatch(setup, /\['set-model-if-current'/);
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
  const appleShortCircuit = setup.match(
    /if \(current\.model === 'apple:system'\) \{[\s\S]*?return \{ success: true, skipped: true, message: 'Apple Intelligence remains selected' \};[\s\S]*?\}/,
  );
  assert.ok(appleShortCircuit, 'Apple selection must return before Ollama setup continues');
  assert.ok(
    setup.indexOf(appleShortCircuit[0]) < ollamaResolution,
    'the Apple short-circuit must precede Ollama model resolution',
  );
});

test('setup fails closed when the current model cannot be read', () => {
  const setup = handlerBody('setup-ollama-and-model');
  assert.match(
    setup,
    /catch \(e\) \{\s*sendDebugLog\(`Could not read current summary model:[\s\S]*?return \{ success: false, error: 'Could not read current summary model\. Please retry setup\.' \};\s*\}/,
  );
  assert.doesNotMatch(setup, /Could not read current summary model, proceeding/);
});

test('setup fails closed when the provider cannot be read', () => {
  const setup = handlerBody('setup-ollama-and-model');

  assert.match(
    setup,
    /catch \(e\) \{\s*sendDebugLog\(`Could not read AI provider:[\s\S]*?return \{ success: false, error: 'Could not read the AI provider\. Please retry setup\.' \};\s*\}/,
  );
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
  assert.match(setup, /setOllamaSetupModelIfCurrent\(\s*setupModelAtStart,/);
  assert.doesNotMatch(setup, /\['set-model',\s*resolved\.installed\]/);
});

test('setup owns Ollama before probing installed models', () => {
  const setup = handlerBody('setup-ollama-and-model');
  const serviceStart = setup.indexOf("ollamaProcess = spawn(finalOllamaPath, ['serve']");
  const readinessGate = setup.indexOf('if (!ready)');
  const modelResolution = setup.indexOf("['resolve-setup-model']");

  assert.notStrictEqual(serviceStart, -1, 'setup must own a newly started Ollama service');
  assert.notStrictEqual(readinessGate, -1, 'setup must wait for Ollama readiness');
  assert.notStrictEqual(modelResolution, -1, 'setup must probe installed models');
  assert.ok(serviceStart < modelResolution, 'model probing must not start Ollama before Electron owns it');
  assert.ok(readinessGate < modelResolution, 'model probing must happen after the readiness gate');
  assert.match(
    setup,
    /if \(!ready\) \{[\s\S]*?return \{ success: false, error: 'Ollama did not become ready\. Please retry setup\.' \};[\s\S]*?\}/,
    'setup must fail before model resolution when Ollama never becomes ready',
  );
});

test('setup model-save errors expose only fixed messages', () => {
  assert.strictEqual(
    modelSetupSaveError({ stdout: '{"success":false,"error":"Could not lock config"}\n' }),
    'Could not lock config',
  );
  assert.strictEqual(
    modelSetupSaveError({
      stdout: '{"success":false,"error":"permission denied: /Users/example/config.json"}\n',
      message: 'backend failed at /Users/example/config.json',
    }),
    'Failed to save the selected model.',
  );
  assert.strictEqual(
    modelSetupSaveError({ success: false, error: 'permission denied: /Users/example/config.json' }),
    'Failed to save the selected model.',
  );
  assert.strictEqual(
    modelSetupSaveError({ success: false, error: 'Could not lock config' }),
    'Could not lock config',
  );
  const setup = handlerBody('setup-ollama-and-model');
  assert.match(setup, /error:\s*modelSetupSaveError\(e\)/);
  assert.strictEqual(
    (setup.match(/error:\s*modelSetupSaveError\(setRes\)/g) || []).length,
    2,
    'both non-success responses must normalize backend errors',
  );
  assert.doesNotMatch(setup, /Failed to save the selected model:\s*\$\{e\.message\}/);
});
