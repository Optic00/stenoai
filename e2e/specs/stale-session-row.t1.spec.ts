import { test, expect } from '../fixtures/electron';

/**
 * T1 — renderer-only, mock IPC. Regression guard for the phantom "Recording"
 * row.
 *
 * When a recording start fails inside the renderer — no microphone connected,
 * the mic held by another app, permission revoked — main.js has already
 * accepted `start-recording-ui` and stored the session name. Its capture-state
 * handler then clears `hasRecording` but deliberately KEEPS that name, on the
 * stated assumption that "a stale name while hasRecording is false is inert".
 *
 * It was not inert: useMeetings built the synthetic in-progress row from the
 * name alone, so the list grew a pulsing "Recording" entry that no queue poll
 * and no reload ever cleared, and clicking it routed to /recording — which
 * correctly saw no session and bounced the user back home half a second later.
 *
 * This is reachable only with STENOAI_E2E_STALE_SESSION_NAME, because the mock
 * otherwise nulls the session name the moment recording goes inactive — a more
 * correct contract than the app implements, and precisely why the whole T1
 * suite stayed green while the bug shipped.
 */

const STALE_ENV = {
  STENOAI_E2E_MOCK_PARAKEET_INSTALLED: '1',
  STENOAI_E2E_STALE_SESSION_NAME: 'Note',
};

test('a session name left behind by a failed start shows no recording row', async ({
  launchApp,
}) => {
  const { page } = await launchApp({ mockIpc: true, env: STALE_ENV });

  // The backend reports exactly what main does after the failure: a name, but
  // nothing running. Assert that first, so a mock drift fails here with a clear
  // message rather than making the real assertion below vacuously pass.
  const queue = await page.evaluate(() => window.stenoai.recording.getQueue());
  expect(queue.sessionName).toBe('Note');
  expect(queue.hasRecording).toBe(false);

  // Give the list a poll cycle plus room for the optimistic caches to settle;
  // the bug's signature was a row that appeared and then never went away.
  await page.waitForTimeout(2000);

  await expect(page.locator('[data-testid="previous-row"][data-recording="true"]')).toHaveCount(0);
  await expect(page.getByText('Recording', { exact: true })).toHaveCount(0);

  // The toolbar button is the other half of the same state: while a session is
  // believed to be live it reads "Stop recording", so a user cannot start a new
  // note at all. (Asserted as an absence — "New note" matches both the icon
  // button and the labelled one.)
  await expect(page.getByRole('button', { name: 'Stop recording' })).toHaveCount(0);
});

test('a genuinely running session still shows its recording row', async ({ launchApp }) => {
  // The guard above must not have been bought by suppressing the real row.
  const { page } = await launchApp({
    mockIpc: true,
    env: { STENOAI_E2E_MOCK_PARAKEET_INSTALLED: '1' },
  });

  await page.evaluate(() => window.stenoai.recording.start('Test note'));

  await expect(
    page.locator('[data-testid="previous-row"][data-recording="true"]'),
  ).toHaveCount(1, { timeout: 10_000 });
});
