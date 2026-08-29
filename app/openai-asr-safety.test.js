'use strict';

const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {
  isEncryptedKeyCleared,
  isValidOpenAiAsrApiKey,
  legacyCredentialSnapshotDigest,
  legacyKeyMigrationAction,
  loadEncryptedKeyForOrigin,
  markEncryptedKeyClearedAtomically,
  readLegacyCredentialSnapshot,
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

test('transcription launches pass the credential origin alongside the key', () => {
  const launch = source.slice(
    source.indexOf('function getTranscriptionEnv()'),
    source.indexOf('// Read the Python-side ai_provider config'),
  );
  assert.match(launch, /STENOAI_OAI_API_KEY/);
  assert.match(launch, /STENOAI_OAI_API_ORIGIN/);
  assert.match(launch, /loadOpenAiAsrCredential/);
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

const ORIGIN = 'https://api.example';

test('legacy key and origin come from one immutable config snapshot', () => {
  withKeyDirectory((keyPath) => {
    const configPath = path.join(path.dirname(keyPath), 'config.json');
    let reads = 0;
    const fsImpl = Object.create(fs);
    fsImpl.existsSync = (target) => target === configPath || fs.existsSync(target);
    fsImpl.readFileSync = (target, ...args) => {
      if (target !== configPath) return fs.readFileSync(target, ...args);
      reads += 1;
      return JSON.stringify(reads === 1 ? {
        openai_asr_api_key: 'key-for-a',
        openai_asr_api_url: 'https://a.example/v1',
      } : {
        openai_asr_api_key: 'key-for-b',
        openai_asr_api_url: 'https://b.example/v1',
      });
    };

    const snapshot = readLegacyCredentialSnapshot({ fs: fsImpl, configPath });
    assert.deepStrictEqual(snapshot, {
      key: 'key-for-a',
      origin: 'https://a.example',
      snapshotDigest: legacyCredentialSnapshotDigest('key-for-a', 'https://a.example/v1'),
    });
    assert.strictEqual(reads, 1);

    assert.strictEqual(save(keyPath, snapshot.key, safeStorage(), fs, snapshot.origin), true);
    assert.strictEqual(loadEncryptedKeyForOrigin({
      fs,
      keyPath,
      origin: 'https://b.example',
      safeStorage: safeStorage(),
    }), null, 'an endpoint change must not activate A\'s snapshot at B');
  });
});

test('an invalid legacy endpoint still yields a digest-bound cleanup snapshot', () => {
  withKeyDirectory((keyPath) => {
    const configPath = path.join(path.dirname(keyPath), 'config.json');
    fs.writeFileSync(configPath, JSON.stringify({
      openai_asr_api_key: 'legacy-key',
      openai_asr_api_url: 'http://untrusted.example/v1',
    }));

    const snapshot = readLegacyCredentialSnapshot({ fs, configPath });
    assert.deepStrictEqual(snapshot, {
      key: 'legacy-key',
      origin: null,
      snapshotDigest: legacyCredentialSnapshotDigest(
        'legacy-key', 'http://untrusted.example/v1',
      ),
    });
  });
});

test('legacy snapshot digest preserves the raw URL value across migration edge cases', () => {
  const cases = [
    { config: {}, apiUrl: 'https://api.openai.com/v1', origin: 'https://api.openai.com' },
    { config: { openai_asr_api_url: '' }, apiUrl: '', origin: null },
    { config: { openai_asr_api_url: 42 }, apiUrl: 42, origin: null },
    { config: { openai_asr_api_url: '  https://api.example/v1  ' }, apiUrl: '  https://api.example/v1  ', origin: 'https://api.example' },
  ];
  withKeyDirectory((keyPath) => {
    const configPath = path.join(path.dirname(keyPath), 'config.json');
    for (const { config, apiUrl, origin } of cases) {
      fs.writeFileSync(configPath, JSON.stringify({
        openai_asr_api_key: 'legacy-key',
        ...config,
      }));
      const snapshot = readLegacyCredentialSnapshot({ fs, configPath });
      assert.strictEqual(snapshot.origin, origin);
      assert.strictEqual(
        snapshot.snapshotDigest,
        legacyCredentialSnapshotDigest('legacy-key', apiUrl),
      );
    }
  });
});

test('legacy snapshots digest raw keys but expose only valid normalized credentials', () => {
  const cases = [
    { rawKey: '\ufefflegacy-key', key: 'legacy-key' },
    { rawKey: '\u001clegacy-key', key: null },
    { rawKey: ' \t ', key: null },
    { rawKey: '\ud800legacy-key', key: null },
  ];
  withKeyDirectory((keyPath) => {
    const configPath = path.join(path.dirname(keyPath), 'config.json');
    for (const { rawKey, key } of cases) {
      fs.writeFileSync(configPath, JSON.stringify({
        openai_asr_api_key: rawKey,
      }));
      const snapshot = readLegacyCredentialSnapshot({ fs, configPath });
      assert.strictEqual(snapshot.key, key);
      assert.strictEqual(
        snapshot.snapshotDigest,
        legacyCredentialSnapshotDigest(rawKey, 'https://api.openai.com/v1'),
      );
    }
  });
});

test('main sends only the legacy snapshot digest to the cleanup CLI', () => {
  const migration = source.slice(
    source.indexOf('async function migrateLegacyOpenAiAsrApiKey()'),
    source.indexOf('// Build the env additions a Python AI-driven subprocess needs.'),
  );
  assert.match(migration, /STENOAI_OAI_LEGACY_SNAPSHOT_DIGEST: legacy\.snapshotDigest/);
  assert.doesNotMatch(migration, /\['remove-legacy-openai-asr-api-key',\s*legacy\.key\]/);
  assert.match(migration, /if \(!legacy\.key \|\| !legacy\.origin\) return removeLegacyKey\(\)/);
});

test('API key validation rejects controls, whitespace, non-ASCII, and oversized values', () => {
  assert.strictEqual(isValidOpenAiAsrApiKey('sk-valid_123'), true);
  for (const invalid of ['', ' leading', 'trailing ', 'line\nbreak', 'unicode-\u2603', 'x'.repeat(4097)]) {
    assert.strictEqual(isValidOpenAiAsrApiKey(invalid), false);
  }
  withKeyDirectory((keyPath) => {
    const invalid = 'secret\nInjected: yes';
    let failure;
    try {
      save(keyPath, invalid);
    } catch (error) {
      failure = error;
    }
    assert.match(failure.message, /was not saved/);
    assert.doesNotMatch(failure.message, /secret|Injected/);
    assert.strictEqual(fs.existsSync(keyPath), false);
  });
});

function save(keyPath, key, storage = safeStorage(), fsImpl = fs, origin = ORIGIN) {
  return saveEncryptedKeyAtomically({
    fs: fsImpl,
    path,
    processId: 123,
    now: 456,
    keyPath,
    key,
    origin,
    safeStorage: storage,
  });
}

function decryptedEnvelope(keyPath, storage = safeStorage()) {
  return JSON.parse(storage.decryptString(fs.readFileSync(keyPath)));
}

test('encrypted API keys are versioned envelopes bound to exactly one endpoint origin', () => {
  withKeyDirectory((keyPath) => {
    assert.strictEqual(save(keyPath, 'bound-key'), true);
    assert.deepStrictEqual(decryptedEnvelope(keyPath), {
      version: 1,
      origin: ORIGIN,
      key: 'bound-key',
    });
    assert.strictEqual(loadEncryptedKeyForOrigin({
      fs,
      keyPath,
      origin: ORIGIN,
      safeStorage: safeStorage(),
    }), 'bound-key');
    assert.strictEqual(loadEncryptedKeyForOrigin({
      fs,
      keyPath,
      origin: 'https://replacement.example',
      safeStorage: safeStorage(),
    }), null, 'endpoint changes must fail closed before Authorization is sent');
  });
});

test('legacy unbound encrypted blobs never become active at any endpoint', () => {
  withKeyDirectory((keyPath) => {
    fs.writeFileSync(keyPath, encrypted('legacy-unbound-key'));
    assert.strictEqual(loadEncryptedKeyForOrigin({
      fs,
      keyPath,
      origin: ORIGIN,
      safeStorage: safeStorage(),
    }), null);
  });
});

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
    assert.deepStrictEqual(decryptedEnvelope(keyPath), {
      version: 1,
      origin: ORIGIN,
      key: 'new-key',
    });
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
    assert.deepStrictEqual(decryptedEnvelope(keyPath), {
      version: 1,
      origin: ORIGIN,
      key: 'replacement-key',
    });
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
