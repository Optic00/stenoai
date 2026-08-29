'use strict';

const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {
  isEncryptedKeyCleared,
  legacyKeyMigrationAction,
  markEncryptedKeyClearedAtomically,
  saveEncryptedKeyAtomically,
} = require('./openai-asr-key-store');

const source = fs.readFileSync(path.join(__dirname, 'main.js'), 'utf8');

test('OpenAI ASR plaintext-key migration runs at application startup', () => {
  const ready = source.indexOf('app.whenReady().then(async () => {');
  const migration = source.indexOf('void migrateLegacyOpenAiAsrApiKey();', ready);
  const menu = source.indexOf('// Application menu.', ready);

  assert.ok(ready >= 0, 'main process must initialize when Electron is ready');
  assert.ok(migration > ready, 'startup must schedule legacy-key migration');
  assert.ok(migration < menu, 'migration must not depend on opening Settings or selecting an engine');
});

function withKeyDirectory(run) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'stenoai-asr-key-'));
  const keyPath = path.join(directory, '.openai-asr-api-key');
  try {
    return run(keyPath);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
}

function encrypted(value) {
  return Buffer.from(`encrypted:${value}`);
}

function safeStorage(overrides = {}) {
  return {
    encryptString: (value) => encrypted(value),
    decryptString: (value) => value.toString().replace(/^encrypted:/, ''),
    ...overrides,
  };
}

function save(keyPath, key, storage = safeStorage(), fsImpl = fs) {
  return saveEncryptedKeyAtomically({
    fs: fsImpl,
    path,
    processId: 123,
    now: 456,
    keyPath,
    key,
    safeStorage: storage,
  });
}

test('early encryption failure leaves the previous encrypted key byte-for-byte intact', () => {
  withKeyDirectory((keyPath) => {
    const previous = encrypted('old-key');
    fs.writeFileSync(keyPath, previous);

    assert.throws(
      () => save(keyPath, 'new-key', safeStorage({
        encryptString: () => { throw new Error('encryption unavailable'); },
      })),
      /prior credential state restored/,
    );
    assert.deepStrictEqual(fs.readFileSync(keyPath), previous);
    assert.deepStrictEqual(fs.readdirSync(path.dirname(keyPath)), [path.basename(keyPath)]);
  });
});

test('early read failure never deletes the previous encrypted key', () => {
  withKeyDirectory((keyPath) => {
    const previous = encrypted('old-key');
    fs.writeFileSync(keyPath, previous);
    const fsImpl = Object.create(fs);
    fsImpl.readFileSync = (target, ...args) => {
      if (target === keyPath) throw new Error('keychain file temporarily unreadable');
      return fs.readFileSync(target, ...args);
    };

    assert.throws(() => save(keyPath, 'new-key', safeStorage(), fsImpl));
    assert.deepStrictEqual(fs.readFileSync(keyPath), previous);
  });
});

test('late decrypt mismatch atomically restores the previous encrypted key', () => {
  withKeyDirectory((keyPath) => {
    const previous = encrypted('old-key');
    fs.writeFileSync(keyPath, previous);

    assert.throws(
      () => save(keyPath, 'new-key', safeStorage({ decryptString: () => 'wrong-key' })),
      /prior credential state restored/,
    );
    assert.deepStrictEqual(fs.readFileSync(keyPath), previous);
    assert.deepStrictEqual(fs.readdirSync(path.dirname(keyPath)), [path.basename(keyPath)]);
  });
});

test('late readback failure atomically restores the previous encrypted key', () => {
  withKeyDirectory((keyPath) => {
    const previous = encrypted('old-key');
    fs.writeFileSync(keyPath, previous);
    let keyReads = 0;
    const fsImpl = Object.create(fs);
    fsImpl.readFileSync = (target, ...args) => {
      if (target === keyPath && ++keyReads === 2) throw new Error('readback failed');
      return fs.readFileSync(target, ...args);
    };

    assert.throws(() => save(keyPath, 'new-key', safeStorage(), fsImpl));
    assert.deepStrictEqual(fs.readFileSync(keyPath), previous);
  });
});

test('rollback filesystem failure retains an encrypted recovery copy of the old key', () => {
  withKeyDirectory((keyPath) => {
    const previous = encrypted('old-key');
    fs.writeFileSync(keyPath, previous);
    const rollbackPath = `${keyPath}.123.456.tmp.rollback`;
    const fsImpl = Object.create(fs);
    fsImpl.renameSync = (source, target) => {
      if (source === rollbackPath && target === keyPath) {
        throw new Error('restore rename failed');
      }
      return fs.renameSync(source, target);
    };

    assert.throws(
      () => save(
        keyPath,
        'new-key',
        safeStorage({ decryptString: () => 'wrong-key' }),
        fsImpl,
      ),
      /rollback also failed/,
    );
    assert.deepStrictEqual(fs.readFileSync(rollbackPath), previous);
  });
});

test('failed first key write removes the unverified new credential', () => {
  withKeyDirectory((keyPath) => {
    assert.throws(() => save(
      keyPath,
      'new-key',
      safeStorage({ decryptString: () => 'wrong-key' }),
    ));
    assert.strictEqual(fs.existsSync(keyPath), false);
  });
});

test('successful key rotation persists a decryptable replacement', () => {
  withKeyDirectory((keyPath) => {
    fs.writeFileSync(keyPath, encrypted('old-key'));
    assert.strictEqual(save(keyPath, 'new-key'), true);
    assert.strictEqual(safeStorage().decryptString(fs.readFileSync(keyPath)), 'new-key');
  });
});

test('clear marker remains authoritative when stale encrypted-key deletion fails', () => {
  withKeyDirectory((keyPath) => {
    fs.writeFileSync(keyPath, encrypted('legacy-key'));
    const fsImpl = Object.create(fs);
    fsImpl.unlinkSync = (target) => {
      if (target === keyPath) throw new Error('simulated stale-key deletion failure');
      return fs.unlinkSync(target);
    };

    assert.strictEqual(markEncryptedKeyClearedAtomically({
      fs: fsImpl,
      path,
      processId: 123,
      now: 456,
      keyPath,
    }), true);
    assert.strictEqual(fs.existsSync(keyPath), true, 'stale bytes reproduce failed deletion');
    assert.strictEqual(isEncryptedKeyCleared({ fs, keyPath }), true);
    assert.strictEqual(
      legacyKeyMigrationAction({ cleared: true, legacyKey: 'legacy-key', storedKey: null }),
      'remove-legacy',
    );
    assert.strictEqual(
      legacyKeyMigrationAction({ cleared: true, legacyKey: 'legacy-key', storedKey: null }),
      'remove-legacy',
      'config refresh must retry deletion without restoring the key',
    );
  });
});

test('an encrypted replacement makes any surviving plaintext stale', () => {
  assert.strictEqual(
    legacyKeyMigrationAction({
      cleared: false,
      legacyKey: 'old-plaintext-key',
      storedKey: 'explicit-replacement-key',
    }),
    'remove-legacy',
  );
});

test('explicit replacement activates only after its encrypted readback succeeds', () => {
  withKeyDirectory((keyPath) => {
    markEncryptedKeyClearedAtomically({
      fs,
      path,
      processId: 123,
      now: 456,
      keyPath,
    });
    assert.strictEqual(isEncryptedKeyCleared({ fs, keyPath }), true);

    assert.strictEqual(save(keyPath, 'replacement-key'), true);
    assert.strictEqual(isEncryptedKeyCleared({ fs, keyPath }), false);
    assert.strictEqual(safeStorage().decryptString(fs.readFileSync(keyPath)), 'replacement-key');
  });
});

test('failed replacement leaves the durable clear marker active', () => {
  withKeyDirectory((keyPath) => {
    markEncryptedKeyClearedAtomically({
      fs,
      path,
      processId: 123,
      now: 456,
      keyPath,
    });

    assert.throws(() => save(
      keyPath,
      'replacement-key',
      safeStorage({ decryptString: () => 'wrong-key' }),
    ));
    assert.strictEqual(isEncryptedKeyCleared({ fs, keyPath }), true);
    assert.strictEqual(fs.existsSync(keyPath), false);
  });
});

test('clear-marker removal failure rolls back before reactivating a key', () => {
  withKeyDirectory((keyPath) => {
    const previous = encrypted('stale-cleared-key');
    fs.writeFileSync(keyPath, previous);
    markEncryptedKeyClearedAtomically({
      fs,
      path,
      processId: 123,
      now: 456,
      keyPath,
    });
    // Reproduce stale encrypted bytes that a previous physical deletion could
    // not remove while the clear marker remains authoritative.
    fs.writeFileSync(keyPath, previous);
    const markerPath = `${keyPath}.cleared`;
    const fsImpl = Object.create(fs);
    fsImpl.unlinkSync = (target) => {
      if (target === markerPath) throw new Error('marker locked');
      return fs.unlinkSync(target);
    };

    assert.throws(() => save(keyPath, 'replacement-key', safeStorage(), fsImpl));
    assert.strictEqual(isEncryptedKeyCleared({ fs, keyPath }), true);
    assert.deepStrictEqual(fs.readFileSync(keyPath), previous);
  });
});

test('main process consults durable clear state before load and legacy migration', () => {
  assert.match(source, /if \(isOpenAiAsrKeyCleared\(\)\) return null;/);
  assert.match(source, /const action = legacyKeyMigrationAction\(/);
  assert.match(source, /if \(action === 'remove-legacy'\)/);
  assert.match(source, /markOpenAiAsrKeyCleared\(\)/);
});
