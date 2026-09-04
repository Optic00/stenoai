const { test } = require('node:test');
const assert = require('node:assert');

const {
  buildNoteReadyNotificationOptions,
  buildTranscriptReadyBody,
  buildCaptureErrorBody,
} = require('./notification-copy');

// Bug C regression guard: the note's real title must reach the notification body.
// The bug was a reprocess completion that carried only the 'Note' placeholder (or
// nothing), so note-ready showed the generic fallback and the title never showed.

test('note-ready body IS the note title when a title is provided', () => {
  const opts = buildNoteReadyNotificationOptions({ title: 'Q3 Planning sync' });
  assert.strictEqual(opts.title, 'Note ready');
  assert.strictEqual(opts.body, 'Q3 Planning sync');
  assert.strictEqual(opts.iconType, 'success');
  assert.strictEqual(opts.outcome, 'success');
});

test('note-ready falls back to a generic body only when there is no title', () => {
  const opts = buildNoteReadyNotificationOptions({});
  assert.strictEqual(opts.body, 'Your note has finished processing');
});

test('note-ready renders the transcription-failure state', () => {
  const opts = buildNoteReadyNotificationOptions({ title: 'Standup', failed: true });
  assert.strictEqual(opts.title, 'Transcription failed');
  assert.strictEqual(opts.iconType, 'alert');
  assert.strictEqual(opts.outcome, 'failed');
});

test('note-ready renders the hard-failure state and quotes the title when present', () => {
  const withTitle = buildNoteReadyNotificationOptions({ title: 'Retro', hardFailure: true });
  assert.strictEqual(withTitle.title, 'Processing failed');
  assert.ok(withTitle.body.includes('"Retro"'));
  assert.strictEqual(withTitle.outcome, 'hard_failure');

  const noTitle = buildNoteReadyNotificationOptions({ hardFailure: true });
  assert.ok(noTitle.body.includes('your note'));
});

test('transcript-ready prompt quotes the note title', () => {
  assert.strictEqual(buildTranscriptReadyBody('Weekly 1:1'), 'Summarise "Weekly 1:1"?');
  assert.strictEqual(buildTranscriptReadyBody(''), 'Summarise?');
  assert.strictEqual(buildTranscriptReadyBody(undefined), 'Summarise?');
});

// --- capture-error copy -----------------------------------------------------
//
// Regression guard for the phantom-recording bug's user-facing half: a start
// that failed in the renderer put the DOMException message straight into a
// desktop notification, so a user with no microphone connected read
// "Recording couldn't start: Requested device not found" — engine text in a
// surface that cannot be expanded or clicked.

test('a missing microphone reads as a missing microphone, not as a device error', () => {
  const body = buildCaptureErrorBody({
    name: 'NotFoundError',
    message: 'Requested device not found',
  });
  assert.match(body, /couldn't find a microphone/);
  assert.doesNotMatch(body, /device not found/i);
});

test('a gone pinned microphone reads the same way', () => {
  const body = buildCaptureErrorBody({ name: 'OverconstrainedError', message: 'deviceId' });
  assert.match(body, /couldn't find a microphone/);
});

test('a permission failure names the platform path, and only its own platform', () => {
  const mac = buildCaptureErrorBody({ name: 'NotAllowedError', platform: 'darwin' });
  assert.match(mac, /System Settings > Privacy & Security > Microphone/);

  const win = buildCaptureErrorBody({ name: 'NotAllowedError', platform: 'win32' });
  assert.match(win, /Settings > Privacy & security > Microphone/);
  assert.doesNotMatch(win, /System Settings/);

  // No settings path is named on Linux — there isn't one worth naming.
  const linux = buildCaptureErrorBody({ name: 'NotAllowedError', platform: 'linux' });
  assert.match(linux, /doesn't have permission to use the microphone/);
  assert.doesNotMatch(linux, /Settings/);
});

test('a busy microphone points at the other app rather than at a setting', () => {
  const body = buildCaptureErrorBody({
    name: 'NotReadableError',
    message: 'Could not start audio source',
  });
  assert.match(body, /another app may be using it/);
  assert.doesNotMatch(body, /audio source/i);
});

test('an unknown failure still says something, and says nothing developer-shaped', () => {
  const body = buildCaptureErrorBody({
    name: 'WeirdFutureError',
    message: "NS_ERROR_FAILURE at /opt/Steno/resources/app.asar/renderer/dist/index.html:0",
  });
  assert.strictEqual(body, "Steno couldn't start the recording. Try again in a moment.");
});

test('no developer text survives any branch', () => {
  const raws = [
    { name: 'NotFoundError', message: 'Requested device not found' },
    { name: 'NotAllowedError', message: 'Permission denied by system' },
    { name: 'NotReadableError', message: 'Could not start audio source' },
    { name: '', message: 'ENOENT: no such file or directory, open /tmp/x' },
    {},
  ];
  for (const raw of raws) {
    const body = buildCaptureErrorBody({ ...raw, platform: 'linux' });
    assert.doesNotMatch(body, /Error:|ENOENT|net::|\/tmp\/|asar/);
    assert.ok(body.length > 0);
  }
});

test('a plain Error is not text-matched into a wrong cause', () => {
  // The capture path also throws plain Errors carrying main's own messages. An
  // EACCES opening the recording FILE used to match /permission/ and tell the
  // user Steno lacked MICROPHONE access — a wrong cause, and on macOS a pointer
  // to a settings pane that could not have helped.
  const body = buildCaptureErrorBody({
    name: 'Error',
    message: "EACCES: permission denied, open '/Users/x/recordings/note.webm'",
    platform: 'darwin',
  });
  assert.doesNotMatch(body, /microphone/i);
  assert.doesNotMatch(body, /System Settings/);
  assert.strictEqual(body, "Steno couldn't start the recording. Try again in a moment.");
});

test('message matching still works for a caller that supplies no name', () => {
  const body = buildCaptureErrorBody({ message: 'Requested device not found' });
  assert.match(body, /couldn't find a microphone/);
});
