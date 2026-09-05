'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { EventEmitter } = require('node:events');
const { makeLineReader } = require('./backend-stream');
const source = fs.readFileSync(require('node:path').join(__dirname, 'main.js'), 'utf8');

for (const channel of ['setup-parakeet', 'pull-parakeet-model']) {
  function start() {
    let handler;
    const proc = new EventEmitter();
    proc.stdout = new EventEmitter();
    proc.stderr = new EventEmitter();
    const events = [];
    const begin = source.indexOf(`ipcMain.handle('${channel}'`);
    const end = source.indexOf('\n});', begin) + 5;
    vm.runInNewContext(source.slice(begin, end), {
      ipcMain: { handle: (_, fn) => { handler = fn; } },
      spawn: () => proc, getBackendPath: () => '/synthetic/backend', getBackendCwd: () => '/synthetic',
      makeLineReader, sendDebugLog: () => {}, setTimeout, clearTimeout,
      mainWindow: { isDestroyed: () => false, webContents: { send: (name, data) => events.push([name, data]) } },
    });
    return { proc, events, promise: handler({}, 'synthetic-model') };
  }
  test(`${channel}: split progress and final JSON survive arbitrary stdout chunks`, async () => {
    const { proc, events, promise } = start();
    for (const chunk of ['PARAKEET_PULL_PRO', 'GRESS:{"stage":"downloading","file_bytes":42}\r', '\n{"suc', 'cess":true}\n']) {
      proc.stdout.emit('data', Buffer.from(chunk));
    }
    proc.emit('close', 0);
    assert.equal((await promise).success, true);
    assert.equal(events[0][0], 'parakeet-pull-progress');
    assert.equal(events[0][1].file_bytes, 42);
  });
  for (const final of ['', '{"success":false,"error":"Synthetic failure"}\n']) {
    test(`${channel}: missing/failed result cannot masquerade as completion (${Boolean(final)})`, async () => {
      const { proc, promise } = start();
      proc.stdout.emit('data', Buffer.from(final));
      proc.emit('close', 0);
      assert.equal((await promise).success, false);
    });
  }
}
