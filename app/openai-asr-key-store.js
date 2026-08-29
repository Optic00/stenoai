'use strict';

const OPENAI_ASR_DEFAULT_URL = 'https://api.openai.com/v1';
const OPENAI_ASR_MAX_KEY_LENGTH = 4096;

function isValidOpenAiAsrApiKey(key) {
  return typeof key === 'string'
    && key.length > 0
    && key.length <= OPENAI_ASR_MAX_KEY_LENGTH
    && /^[\x21-\x7e]+$/.test(key);
}

function normalizeOpenAiAsrOrigin(apiUrl) {
  try {
    if (typeof apiUrl !== 'string' || !apiUrl.trim()) return null;
    const endpoint = new URL(apiUrl.trim());
    const hostname = endpoint.hostname.toLowerCase();
    if (!hostname || endpoint.username || endpoint.password || endpoint.search || endpoint.hash) return null;
    const loopback = hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '[::1]';
    if (endpoint.protocol !== 'https:' && !(endpoint.protocol === 'http:' && loopback)) return null;
    return endpoint.origin === 'null' ? null : endpoint.origin;
  } catch (_) {
    return null;
  }
}

function readLegacyCredentialSnapshot({ fs, configPath }) {
  try {
    if (!fs.existsSync(configPath)) return null;
    // This single read is the authority for both fields. A later endpoint
    // change must never bind this snapshot's plaintext key to another origin.
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    const key = config && typeof config.openai_asr_api_key === 'string'
      ? config.openai_asr_api_key.trim()
      : '';
    if (!key) return null;
    const apiUrl = config && Object.prototype.hasOwnProperty.call(config, 'openai_asr_api_url')
      ? config.openai_asr_api_url
      : OPENAI_ASR_DEFAULT_URL;
    const origin = normalizeOpenAiAsrOrigin(apiUrl);
    return origin ? { key, origin } : null;
  } catch (_) {
    return null;
  }
}

function clearedMarkerPath(keyPath) {
  return `${keyPath}.cleared`;
}

function isEncryptedKeyCleared({ fs, keyPath }) {
  return fs.existsSync(clearedMarkerPath(keyPath));
}

/**
 * Persist the cleared state before best-effort removal of stale encrypted
 * bytes. The marker is the authority, so a filesystem failure cannot make an
 * old key active again on the next config refresh.
 */
function markEncryptedKeyClearedAtomically({ fs, path, processId, now, keyPath }) {
  const markerPath = clearedMarkerPath(keyPath);
  const tempPath = `${markerPath}.${processId}.${now}.tmp`;
  const keyDir = path.dirname(keyPath);

  if (!fs.existsSync(keyDir)) fs.mkdirSync(keyDir, { recursive: true });
  if (!fs.existsSync(markerPath)) {
    try {
      fs.writeFileSync(tempPath, 'cleared\n', { mode: 0o600 });
      fs.renameSync(tempPath, markerPath);
    } catch (error) {
      try {
        if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
      } catch (_) {}
      throw new Error('OpenAI ASR API key clear state was not saved', { cause: error });
    }
  }

  try {
    if (fs.existsSync(keyPath)) fs.unlinkSync(keyPath);
  } catch (_) {
    // The durable marker keeps stale encrypted bytes inactive. A later clear
    // or migration pass can retry their physical removal.
  }
  return true;
}

function legacyKeyMigrationAction({ cleared, legacyKey, storedKey }) {
  if (!legacyKey) return 'none';
  if (cleared) return 'remove-legacy';
  if (!storedKey) return 'secure';
  // An explicit encrypted key is authoritative. Any surviving plaintext is
  // stale, even if a later rotation made the values differ.
  return 'remove-legacy';
}

function decodeKeyEnvelope(value, origin) {
  try {
    const envelope = JSON.parse(value);
    if (
      !envelope
      || envelope.version !== 1
      || envelope.origin !== origin
      || typeof envelope.key !== 'string'
      || !isValidOpenAiAsrApiKey(envelope.key)
    ) {
      return null;
    }
    return envelope.key;
  } catch (_) {
    // Pre-envelope safeStorage blobs intentionally fail closed. We cannot
    // prove which endpoint received the key that they contain.
    return null;
  }
}

function loadEncryptedKeyForOrigin({ fs, keyPath, origin, safeStorage }) {
  if (typeof origin !== 'string' || !origin) return null;
  try {
    if (!fs.existsSync(keyPath)) return null;
    return decodeKeyEnvelope(safeStorage.decryptString(fs.readFileSync(keyPath)), origin);
  } catch (_) {
    return null;
  }
}

/**
 * Atomically replace an encrypted OpenAI ASR credential and verify that the
 * committed bytes decrypt to the requested plaintext. The previous blob is
 * captured before encryption starts, so an early safeStorage failure cannot
 * be mistaken for "there was no previous credential".
 */
function saveEncryptedKeyAtomically({ fs, path, processId, now, keyPath, key, origin, safeStorage }) {
  const tempPath = `${keyPath}.${processId}.${now}.tmp`;
  const rollbackPath = `${tempPath}.rollback`;
  const markerPath = clearedMarkerPath(keyPath);
  const keyDir = path.dirname(keyPath);
  let hadPrevious = false;
  let previous = null;
  let rollbackPrepared = false;
  let committed = false;
  let clearedStateRemoved = false;

  try {
    if (typeof origin !== 'string' || !origin) {
      throw new Error('OpenAI ASR API key endpoint is invalid');
    }
    if (!isValidOpenAiAsrApiKey(key)) {
      throw new Error('OpenAI ASR API key has an invalid format');
    }
    if (!fs.existsSync(keyDir)) fs.mkdirSync(keyDir, { recursive: true });

    hadPrevious = fs.existsSync(keyPath);
    if (hadPrevious) previous = fs.readFileSync(keyPath);

    const envelope = JSON.stringify({ version: 1, origin, key });
    const encrypted = safeStorage.encryptString(envelope);
    fs.writeFileSync(tempPath, encrypted, { mode: 0o600 });
    if (hadPrevious) {
      // Prepare the encrypted recovery blob before replacing keyPath. A disk
      // error can therefore abort while the old path is still authoritative.
      fs.writeFileSync(rollbackPath, previous, { mode: 0o600 });
      rollbackPrepared = true;
    }
    fs.renameSync(tempPath, keyPath);
    committed = true;

    const readback = decodeKeyEnvelope(
      safeStorage.decryptString(fs.readFileSync(keyPath)), origin,
    );
    if (readback !== key) throw new Error('safeStorage readback did not match saved key');
    if (fs.existsSync(markerPath)) {
      // Only a verified explicit replacement may reactivate the credential.
      // Keep the encrypted rollback copy until this succeeds, so a failure
      // leaves the authoritative marker and prior bytes intact.
      fs.unlinkSync(markerPath);
      clearedStateRemoved = true;
    }
    if (rollbackPrepared) {
      try {
        fs.unlinkSync(rollbackPath);
        rollbackPrepared = false;
      } catch (cleanupError) {
        // After the clear marker is removed, the verified replacement is the
        // authoritative state. A stale encrypted recovery blob is safer than
        // attempting to roll back to an already-cleared key.
        if (!clearedStateRemoved) throw cleanupError;
      }
    }
    return true;
  } catch (error) {
    const rollbackFailures = [];

    try {
      if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
    } catch (cleanupError) {
      rollbackFailures.push(cleanupError);
    }

    // Before renameSync succeeds the previous keyPath is still authoritative,
    // so touching it would turn an early failure into credential loss.
    if (committed) {
      try {
        if (hadPrevious) {
          fs.renameSync(rollbackPath, keyPath);
          rollbackPrepared = false;
        } else if (fs.existsSync(keyPath)) {
          fs.unlinkSync(keyPath);
        }
      } catch (rollbackError) {
        rollbackFailures.push(rollbackError);
      }
    }

    // If restoration itself failed, retain the encrypted recovery blob rather
    // than turning a filesystem error into permanent loss of the old key.
    if (!committed || rollbackFailures.length === 0) {
      try {
        if (rollbackPrepared && fs.existsSync(rollbackPath)) fs.unlinkSync(rollbackPath);
      } catch (cleanupError) {
        rollbackFailures.push(cleanupError);
      }
    }

    const detail = rollbackFailures.length > 0
      ? '; rollback also failed'
      : '; prior credential state restored';
    throw new Error(`OpenAI ASR API key was not saved${detail}`, { cause: error });
  }
}

module.exports = {
  isEncryptedKeyCleared,
  isValidOpenAiAsrApiKey,
  legacyKeyMigrationAction,
  loadEncryptedKeyForOrigin,
  markEncryptedKeyClearedAtomically,
  normalizeOpenAiAsrOrigin,
  readLegacyCredentialSnapshot,
  saveEncryptedKeyAtomically,
};
