'use strict';

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

/**
 * Atomically replace an encrypted OpenAI ASR credential and verify that the
 * committed bytes decrypt to the requested plaintext. The previous blob is
 * captured before encryption starts, so an early safeStorage failure cannot
 * be mistaken for "there was no previous credential".
 */
function saveEncryptedKeyAtomically({ fs, path, processId, now, keyPath, key, safeStorage }) {
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
    if (!fs.existsSync(keyDir)) fs.mkdirSync(keyDir, { recursive: true });

    hadPrevious = fs.existsSync(keyPath);
    if (hadPrevious) previous = fs.readFileSync(keyPath);

    const encrypted = safeStorage.encryptString(key);
    fs.writeFileSync(tempPath, encrypted, { mode: 0o600 });
    if (hadPrevious) {
      // Prepare the encrypted recovery blob before replacing keyPath. A disk
      // error can therefore abort while the old path is still authoritative.
      fs.writeFileSync(rollbackPath, previous, { mode: 0o600 });
      rollbackPrepared = true;
    }
    fs.renameSync(tempPath, keyPath);
    committed = true;

    const readback = safeStorage.decryptString(fs.readFileSync(keyPath));
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
  legacyKeyMigrationAction,
  markEncryptedKeyClearedAtomically,
  saveEncryptedKeyAtomically,
};
