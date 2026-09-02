'use strict';

const APPLE_SYSTEM_MODEL = 'apple:system';
const SAFE_MODEL_SAVE_ERRORS = new Set([
  'Could not lock config',
  'Failed to stage model config',
  'Failed to save config',
]);

function assertOllamaSetupModel(model) {
  if (typeof model !== 'string' || !model || model === APPLE_SYSTEM_MODEL) {
    throw new Error('Ollama setup cannot select Apple Intelligence');
  }
  return model;
}

function modelSetupSaveError(error) {
  try {
    const jsonLine = String(error?.stdout || '').trim().split('\n').reverse()
      .find((line) => line.trim().startsWith('{'));
    const result = jsonLine ? JSON.parse(jsonLine) : null;
    if (result && SAFE_MODEL_SAVE_ERRORS.has(result.error)) {
      return result.error;
    }
  } catch (_) {}
  return 'Failed to save the selected model.';
}

module.exports = { assertOllamaSetupModel, modelSetupSaveError };
