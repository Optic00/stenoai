'use strict';

const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const source = fs.readFileSync(path.join(__dirname, 'main.js'), 'utf8');

function functionSource(name, nextName) {
  const start = source.indexOf(`function ${name}(`);
  const end = source.indexOf(`function ${nextName}(`, start + 1);
  assert.notStrictEqual(start, -1, `${name} must exist`);
  assert.notStrictEqual(end, -1, `${nextName} must follow ${name}`);
  return source.slice(start, end);
}

test('OpenAI ASR plaintext-key migration runs at application startup', () => {
  const ready = source.indexOf('app.whenReady().then(async () => {');
  const migration = source.indexOf('void migrateLegacyOpenAiAsrApiKey();', ready);
  const menu = source.indexOf('// Application menu.', ready);

  assert.ok(ready >= 0, 'main process must initialize when Electron is ready');
  assert.ok(migration > ready, 'startup must schedule legacy-key migration');
  assert.ok(migration < menu, 'migration must not depend on opening Settings or selecting an engine');
});

test('OpenAI ASR key writes atomically verify readback before plaintext removal', () => {
  const save = functionSource('saveOpenAiAsrKey', 'loadOpenAiAsrKey');
  const secureLegacy = functionSource('secureLegacyOpenAiAsrApiKey', 'migrateLegacyOpenAiAsrApiKey');

  assert.match(save, /fs\.writeFileSync\(tempPath, encrypted, \{ mode: 0o600 \}\)/);
  assert.match(save, /fs\.renameSync\(tempPath, keyPath\)/);
  assert.match(save, /loadOpenAiAsrKey\(\) !== key/);
  assert.match(save, /prior credential state restored/);
  assert.match(save, /rollback also failed/);
  assert.match(secureLegacy, /stored === legacyKey/);
  assert.match(secureLegacy, /saveOpenAiAsrKey\(legacyKey\)/);
});
