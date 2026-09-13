import { test, expect } from '../fixtures/electron';

async function openAiSettings(page: import('@playwright/test').Page) {
  await page.evaluate(() => {
    window.location.hash = '#/settings?tab=ai';
  });
  await expect(page.getByTestId('transcription-model-select')).toBeVisible();
}

async function chooseCloudApi(page: import('@playwright/test').Page) {
  await page.getByTestId('transcription-model-select').click();
  await page.getByRole('option', { name: /cloud api/i }).click();
}

test('cloud ASR requires confirmation and honours cancellation', async ({ launchApp }) => {
  const { page } = await launchApp({ mockIpc: true });
  await openAiSettings(page);

  await chooseCloudApi(page);
  const dialog = page.locator('[data-confirm-dialog]');
  await expect(dialog).toContainText('Send audio to a cloud service?');
  await expect(dialog).toContainText('audio leaves your computer');
  await dialog.getByRole('button', { name: 'Cancel' }).click();

  await expect(page.getByTestId('openai-asr-config')).toHaveCount(0);

  await chooseCloudApi(page);
  await dialog.getByRole('button', { name: 'Use cloud ASR' }).click();
  await expect(page.getByTestId('openai-asr-config')).toBeVisible();
});

test('cloud ASR save failures stay visible and restore the committed endpoint', async ({ launchApp }) => {
  const { page } = await launchApp({
    mockIpc: true,
    env: { STENOAI_E2E_OAI_ASR_SAVE_FAIL: '1' },
  });
  await openAiSettings(page);
  await chooseCloudApi(page);
  await page.locator('[data-confirm-dialog]').getByRole('button', { name: 'Use cloud ASR' }).click();

  const config = page.getByTestId('openai-asr-config');
  const endpoint = config.getByLabel('API base URL');
  await endpoint.fill('https://rejected.example/v1');
  await endpoint.blur();

  await expect(config.getByRole('alert')).toContainText('previous value is still active');
  await expect(endpoint).toHaveValue('https://api.openai.com/v1');
});

test('clearing a saved key wins over the replacement queued by input blur', async ({ launchApp }) => {
  const { page } = await launchApp({
    mockIpc: true,
    env: {
      STENOAI_E2E_OAI_ASR_KEY_SET: '1',
      STENOAI_E2E_OAI_ASR_KEY_RACE: '1',
    },
  });
  await openAiSettings(page);
  await chooseCloudApi(page);
  await page.locator('[data-confirm-dialog]').getByRole('button', { name: 'Use cloud ASR' }).click();

  const config = page.getByTestId('openai-asr-config');
  await config.getByLabel('API key').fill('replacement-key');
  await config.getByRole('button', { name: 'Clear' }).click();

  await expect(config).toContainText('Stored encrypted on your device');
  await expect(config.getByRole('button', { name: 'Clear' })).toHaveCount(0);
});

test('a completed model save preserves an unsaved endpoint edit', async ({ launchApp }) => {
  const { page } = await launchApp({
    mockIpc: true,
    env: { STENOAI_E2E_OAI_ASR_CONFIG_RACE: '1' },
  });
  await openAiSettings(page);
  await chooseCloudApi(page);
  await page.locator('[data-confirm-dialog]').getByRole('button', { name: 'Use cloud ASR' }).click();
  const config = page.getByTestId('openai-asr-config');
  await config.getByLabel('Model', { exact: true }).fill('replacement-model');
  await config.getByLabel('API base URL').fill('https://unsaved.example/v1');
  await expect.poll(() => page.evaluate(async () =>
    (await window.stenoai.openaiAsr.getConfig()).model,
  )).toBe('replacement-model');
  // Let the completed mutation's query invalidation and React paint settle.
  await page.evaluate(() => new Promise<void>((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  }));
  await expect(config.getByLabel('API base URL')).toHaveValue('https://unsaved.example/v1');
});

test('a completed key save preserves a newly typed replacement', async ({ launchApp }) => {
  const { page } = await launchApp({
    mockIpc: true,
    env: { STENOAI_E2E_OAI_ASR_KEY_RACE: '1' },
  });
  await openAiSettings(page);
  await chooseCloudApi(page);
  await page.locator('[data-confirm-dialog]').getByRole('button', { name: 'Use cloud ASR' }).click();
  const config = page.getByTestId('openai-asr-config');
  const input = config.getByLabel('API key');
  await input.fill('first-synthetic-key');
  await input.blur();
  await input.fill('second-synthetic-key');
  await expect(config).toContainText('A key is saved.');
  await expect(input).toHaveValue('second-synthetic-key');
  await input.blur();
  await expect(input).toHaveValue('');
});
