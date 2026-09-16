// e2e/specs/templates-crud.t2.spec.ts
import { test, expect } from "../fixtures/electron";
import {
  readUserConfig,
  writeUserConfig,
  writeMeetingSummary,
} from "../fixtures/user-config";
import { realUserDataDir, fileSig } from "../fixtures/real-user-data";
import { startMockOllama } from "../fixtures/mock-ollama";
import type { Page } from "@playwright/test";
import { readFileSync } from "fs";
import path from "path";

type Tmpl = { id: string; name: string; builtin: boolean; locked: boolean };
type FullTmpl = Tmpl & { prompt: string; language: string; format?: string };
type ListResult = {
  success: boolean;
  templates: FullTmpl[];
  default_template_id: string;
};
type SaveResult = {
  success: boolean;
  template?: { id: string };
  error?: string;
};
type Result = { success?: boolean };

type StenoWindow = Window & {
  stenoai: {
    templates: {
      list: () => Promise<ListResult>;
      save: (t: Record<string, unknown>) => Promise<SaveResult>;
      remove: (id: string) => Promise<Result>;
      setDefault: (id: string) => Promise<Result>;
      reset: (id: string) => Promise<Result>;
    };
    meetings: {
      generateReport: (
        summaryFile: string,
        templateId: string,
      ) => Promise<Result>;
    };
  };
};

// The editable built-in gallery. Standard is locked and covered separately.
const GALLERY_IDS = ["product-demo", "sales-call", "one-on-one", "standup"];

type Override = { name: string; prompt: string; language: string };

const listTemplates = (page: Page) =>
  page.evaluate(() =>
    (window as unknown as StenoWindow).stenoai.templates.list(),
  );

const resetTemplate = (page: Page, id: string) =>
  page.evaluate(
    (tid) => (window as unknown as StenoWindow).stenoai.templates.reset(tid),
    id,
  );

const byId = (listed: ListResult, id: string): FullTmpl => {
  const found = listed.templates.find((t) => t.id === id);
  expect(found, `template ${id} is listed`).toBeTruthy();
  return found!;
};

const storedOverrides = (userDataDir: string) =>
  (readUserConfig(userDataDir).template_overrides ?? {}) as Record<
    string,
    Override
  >;

test("templates: list ships Standard + seeded sample; default is Standard", async ({
  launchApp,
}) => {
  const { page } = await launchApp();
  const data = await page.evaluate(() =>
    (window as unknown as StenoWindow).stenoai.templates.list(),
  );
  expect(data.success).toBe(true);
  const ids = data.templates.map((t) => t.id);
  expect(ids).toContain("standard");
  expect(ids).toContain("shareable-summary");
  expect(data.default_template_id).toBe("standard");
  // standard is a locked built-in
  expect(data.templates.find((t) => t.id === "standard")?.locked).toBe(true);
});

test("templates: create custom, set default, delete; persisted to config.json", async ({
  launchApp,
  userDataDir,
}) => {
  const { page } = await launchApp();
  const saved = await page.evaluate(() =>
    (window as unknown as StenoWindow).stenoai.templates.save({
      name: "Leitung",
      prompt: "kurz halten",
      language: "de",
    }),
  );
  expect(saved.success).toBe(true);
  const id = saved.template!.id;

  await page.evaluate(
    (tid) =>
      (window as unknown as StenoWindow).stenoai.templates.setDefault(tid),
    id,
  );

  await expect
    .poll(() => readUserConfig(userDataDir).default_template_id)
    .toBe(id);
  expect(readUserConfig(userDataDir).custom_templates).toEqual(
    expect.arrayContaining([
      expect.objectContaining({ id, prompt: "kurz halten" }),
    ]),
  );

  await page.evaluate(
    (tid) => (window as unknown as StenoWindow).stenoai.templates.remove(tid),
    id,
  );
  await expect
    .poll(() =>
      (readUserConfig(userDataDir).custom_templates as Tmpl[]).map((t) => t.id),
    )
    .not.toContain(id);
  // deleting the default falls back to standard
  await expect
    .poll(() => readUserConfig(userDataDir).default_template_id)
    .toBe("standard");
});

test("templates: locked Standard rejects an edit", async ({ launchApp }) => {
  const { page } = await launchApp();
  const res = await page.evaluate(() =>
    (window as unknown as StenoWindow).stenoai.templates.save({
      id: "standard",
      name: "X",
      prompt: "hack",
      language: "auto",
    }),
  );
  expect(res.success).toBe(false);
  expect((res.error ?? "").toLowerCase()).toContain("locked");
});

test("templates: built-in edits from an earlier version survive listing until each is reset", async ({
  launchApp,
  userDataDir,
}) => {
  test.setTimeout(90_000);
  // An upgrading user's edits are already in config.json under the built-in ids
  // when the new version first starts, so they are seeded before launch rather
  // than saved through the running app.
  const earlier: Record<string, Override> = Object.fromEntries(
    GALLERY_IDS.map((id) => [
      id,
      {
        name: `Team ${id}`,
        prompt: `Earlier custom instructions for ${id}.`,
        language: "de",
      },
    ]),
  );
  writeUserConfig(userDataDir, {
    templates_seeded: true,
    template_overrides: earlier,
  });

  const { page } = await launchApp();

  const listed = await listTemplates(page);
  expect(listed.success).toBe(true);
  for (const id of GALLERY_IDS) {
    expect(byId(listed, id)).toMatchObject({
      ...earlier[id],
      builtin: true,
      locked: false,
    });
  }
  // Listing never rewrites or drops the stored edits.
  expect(storedOverrides(userDataDir)).toEqual(earlier);

  // Reset one template at a time: only that one returns to its shipped
  // definition, every other edit stays in effect and on disk.
  const pending = new Set(GALLERY_IDS);
  for (const id of GALLERY_IDS) {
    expect(await resetTemplate(page, id)).toEqual({ success: true });
    pending.delete(id);
    expect(Object.keys(storedOverrides(userDataDir)).sort()).toEqual(
      [...pending].sort(),
    );

    const afterReset = await listTemplates(page);
    const shipped = byId(afterReset, id);
    expect(shipped).toMatchObject({
      language: "auto",
      format: "markdown",
      builtin: true,
      locked: false,
    });
    expect(shipped.name).not.toBe(earlier[id].name);
    expect(shipped.prompt).not.toBe(earlier[id].prompt);
    expect(shipped.prompt.trim()).not.toBe("");
    for (const other of pending) {
      expect(byId(afterReset, other)).toMatchObject(earlier[other]);
    }
  }
});

test("templates: editing a built-in persists name, prompt and language until reset", async ({
  launchApp,
  userDataDir,
}) => {
  test.setTimeout(90_000);
  const { page } = await launchApp();
  const shipped = await listTemplates(page);
  expect(shipped.success).toBe(true);

  const edits: Record<string, Override> = {};
  for (const id of GALLERY_IDS) {
    const original = byId(shipped, id);
    expect(original).toMatchObject({
      language: "auto",
      builtin: true,
      locked: false,
    });
    expect(original.prompt.trim()).not.toBe("");
    edits[id] = {
      name: `${original.name} edited`,
      prompt: `Edited instructions for ${id}.`,
      language: "de",
    };
    const saved = await page.evaluate(
      (payload) =>
        (window as unknown as StenoWindow).stenoai.templates.save(payload),
      {
        id,
        ...edits[id],
      },
    );
    expect(saved.success).toBe(true);
    expect(saved.template).toMatchObject({ id, ...edits[id] });
  }

  expect(storedOverrides(userDataDir)).toEqual(edits);
  const edited = await listTemplates(page);
  for (const id of GALLERY_IDS) {
    expect(byId(edited, id)).toMatchObject({
      id,
      ...edits[id],
      builtin: true,
      locked: false,
    });
  }

  for (const id of GALLERY_IDS) {
    expect(await resetTemplate(page, id)).toEqual({ success: true });
  }
  expect(storedOverrides(userDataDir)).toEqual({});
  const restored = await listTemplates(page);
  for (const id of GALLERY_IDS) {
    const { name, prompt, language, format } = byId(shipped, id);
    expect(byId(restored, id)).toMatchObject({
      name,
      prompt,
      language,
      format,
      builtin: true,
      locked: false,
    });
  }
});

test("templates: report generation sends the shipped built-in prompt, then an edit, then the shipped prompt after reset", async ({
  launchApp,
  userDataDir,
}) => {
  test.setTimeout(120_000);
  const realDirBefore = fileSig(realUserDataDir());
  const id = "sales-call";
  const editedPrompt = "Edited sales instructions for this contract test.";

  // Local provider so the summarizer talks to the capturing mock Ollama.
  writeUserConfig(userDataDir, { ai_provider: "local" });
  const summaryFile = writeMeetingSummary(userDataDir, "template-prompt", {
    name: "Template Prompt Meeting",
    summary: "Existing summary",
    transcript:
      "Alice: the pilot starts next week. Bob: I will send the contract today.",
  });
  const sidecarFile = path.join(
    path.dirname(summaryFile),
    "template-prompt_reports.json",
  );

  // The reply is stored verbatim and never asserted on: this checks what is
  // sent to the model, not what a model would write.
  const ollama = await startMockOllama({
    chatReply: "## Report\n- mock reply",
  });
  try {
    const { page } = await launchApp();
    // The shipped prompt as the running backend lists it, so the spec does not
    // pin prompt wording.
    const shippedPrompt = byId(await listTemplates(page), id).prompt;
    expect(shippedPrompt.trim()).not.toBe("");

    const generate = async () => {
      const callsBefore = ollama.chatCalls();
      const res = await page.evaluate(
        ([f, tid]) =>
          (window as unknown as StenoWindow).stenoai.meetings.generateReport(
            f,
            tid,
          ),
        [summaryFile, id] as [string, string],
      );
      expect(res.success).toBe(true);
      // Exactly one model call per report, so the captured prompt is this run's.
      expect(ollama.chatCalls()).toBe(callsBefore + 1);
      const prompt = ollama.lastChatPrompt() ?? "";
      expect(prompt).toContain("the pilot starts next week");
      return prompt;
    };

    expect(await generate()).toContain(shippedPrompt);

    const saved = await page.evaluate(
      (payload) =>
        (window as unknown as StenoWindow).stenoai.templates.save(payload),
      {
        id,
        name: "Sales Call",
        prompt: editedPrompt,
        language: "de",
      },
    );
    expect(saved.success).toBe(true);
    const editedRun = await generate();
    expect(editedRun).toContain(editedPrompt);
    expect(editedRun).not.toContain(shippedPrompt);
    // The edit's pinned language reaches the report prompt as well.
    expect(editedRun).toContain("Write the report in German");

    expect(await resetTemplate(page, id)).toEqual({ success: true });
    const resetRun = await generate();
    expect(resetRun).toContain(shippedPrompt);
    expect(resetRun).not.toContain(editedPrompt);

    const sidecar = JSON.parse(readFileSync(sidecarFile, "utf8")) as {
      reports: Array<{ template_id: string }>;
    };
    expect(sidecar.reports.map((r) => r.template_id)).toEqual([id, id, id]);

    // Keystone: the real user-data dir is untouched.
    expect(fileSig(realUserDataDir())).toBe(realDirBefore);
  } finally {
    await ollama.close();
  }
});
