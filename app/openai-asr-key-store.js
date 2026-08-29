'use strict';

/**
 * Atomically replace an encrypted OpenAI ASR credential and verify that the
 * committed bytes decrypt to the requested plaintext. The previous blob is
 * captured before encryption starts, so an early safeStorage failure cannot
 * be mistaken for "there was no previous credential".
 */
function saveEncryptedKeyAtomically({ fs, path, processId, now, keyPath, key, safeStorage }) {
  const tempPath = `${keyPath}.${processId}.${now}.tmp`;
  const rollbackPath = `${tempPath}.rollback`;
  const keyDir = path.dirname(keyPath);
  let hadPrevious = false;
  let previous = null;
  let rollbackPrepared = false;
  let committed = false;

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
    if (rollbackPrepared) {
      fs.unlinkSync(rollbackPath);
      rollbackPrepared = false;
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

module.exports = { saveEncryptedKeyAtomically };
