# src/templates.py
"""Built-in report templates + pure template helpers.

A template shapes how a meeting report is generated/exported. STANDARD is the
only *locked* built-in (drives today's structured note); the rest of
BUILTIN_TEMPLATES is a curated gallery of editable/resettable prompt-driven
templates for common meeting types (issue #297) — same override/reset
mechanism as any editable built-in, just shipped with a useful starting
prompt instead of an empty one. One editable SAMPLE custom template is also
pre-seeded by Config on first run; everything else is user-created. Built-in
definitions live here (source of truth, like whisper_models.py); custom
templates, built-in overrides, the seed flag, and the default id live in
config.json.

This module is pure — no I/O — so it is unit-testable without a running app.
"""

import re

STANDARD_TEMPLATE_ID = "standard"

MAX_NAME_LEN = 200
MAX_PROMPT_LEN = 8000
MAX_ICON_LEN = 64
VALID_FORMATS = {"structured", "markdown"}

# STANDARD is locked + structured: its "prompt" is empty because it routes
# through the existing JSON-schema summary path, not a free-form prompt.
# `format` picks the render/generation path.
#
# The rest are the built-in gallery (issue #297): editable + resettable
# (locked left unset/False) prompt-driven templates for common meeting
# types, so most users get a useful default without writing their own
# prompt — Custom-template CRUD covers anyone who wants more.
BUILTIN_TEMPLATES = {
    "standard": {
        "id": "standard",
        "name": "Standard",
        "icon": "doc",
        "prompt": "",
        "language": "auto",
        "format": "structured",
        "locked": True,
    },
    "product-demo": {
        "id": "product-demo",
        "name": "Product Demo",
        "icon": "presentation",
        "prompt": (
            "Write a concise product-demo report. Write everything, including "
            "headings, only in the language of the meeting. Use these sections at most "
            "once, with brief bullets, and omit any section that was not discussed: "
            "Needs and fit; Capabilities; Commercial terms; Concerns and open "
            "questions; Next steps. Give every capability whose status was discussed "
            "exactly one status: "
            "demonstrated (shown working during the meeting), available but not "
            "shown (said to exist but not demonstrated), planned (promised for "
            "later, with any stated timing), or not available (explicitly said to "
            "be unsupported). Never raise a status: describing a capability is not "
            "demonstrating it, and a plan is not availability. For each stated "
            "need, give the status of the matching capability and the "
            "participants' own assessment of fit; do not judge fit yourself. Keep "
            "a need whose matching capability was not addressed under open questions "
            "with its status marked as not established. Keep "
            "conditions, approvals, and limits attached to the prices and offers "
            "they apply to. A request is not an accepted task: give an owner or "
            "date only when that person explicitly agreed to it, and otherwise "
            "list the request as open without an owner. State each point once."
        ),
        "language": "auto",
        "format": "markdown",
    },
    "sales-call": {
        "id": "sales-call",
        "name": "Sales Call",
        "icon": "handshake",
        "prompt": (
            "Write a concise sales-call report. Write everything, including "
            "headings, only in the language of the meeting. Use these sections at most "
            "once, with brief bullets: Needs and pain points; Objections and "
            "concerns; Budget, timeline, and decision process; Alternatives and "
            "competitors; Commitment status; Next steps. Omit a section that was "
            "not discussed, except Commitment status, which is always included. "
            "Under Objections and concerns, keep every objection, hesitation, or "
            "precondition the prospect raised, even when the seller answered it. "
            "Under Commitment status, state plainly whether the prospect committed "
            "to buy, committed only under conditions (name them), declined, or "
            "made no commitment; interest alone is not a commitment. Give each "
            "budget, date, and deadline its stated status, such as estimated, "
            "approved, proposed, or accepted. A date suggested by one side counts "
            "as agreed only when the other side explicitly accepts it. Attribute "
            "positions to the prospect or the seller, and do not infer decision "
            "authority or deal likelihood. Under Next steps, list only actions "
            "someone agreed to take, with owners and dates as stated, and list "
            "unanswered requests as open. State each fact once."
        ),
        "language": "auto",
        "format": "markdown",
    },
    "one-on-one": {
        "id": "one-on-one",
        "name": "1:1",
        "icon": "user-check",
        "prompt": (
            "Write a concise 1:1 report. Write everything, including headings, only "
            "in the language of the meeting. Use these sections at most once, with "
            "brief bullets, and omit any section that was not discussed: Updates; "
            "Feedback; Concerns; Differing views; Decisions; Actions and open "
            "requests. Under Differing views, give each disagreement its own "
            "bullet: the topic, then each person's position attributed by name, "
            "then whether it was resolved. Keep a disagreement even when it was "
            "brief or left unresolved, and do not say who is right. Attribute "
            "feedback, concerns, and feelings to the person who expressed them; do "
            "not infer motives or judge performance. A suggestion or request is "
            "not a decision: record a decision or action only when it was "
            "explicitly agreed, and mark deferred or unanswered items as open. "
            "Include owners, dates, and conditions only as stated. State each point "
            "once."
        ),
        "language": "auto",
        "format": "markdown",
    },
    "standup": {
        "id": "standup",
        "name": "Standup",
        "icon": "list-checks",
        "prompt": (
            "Write concise standup notes. Write everything, including headings, only "
            "in the language of the meeting. Give each identified person a short "
            "bullet list and, if any work has no stated owner, end with one separate "
            "list for it. "
            "For every work item, give its status as stated: done, in progress, "
            "planned, conditional (with the condition), blocked (with the blocker), "
            "or unblocked. Keep each status, plan, condition, and blocker attached "
            "to the exact item it was said about; when a person mentions several "
            "items, never move a status, plan, or condition from one item to "
            "another. Record requests for help. When someone agrees to help or to "
            "take over work, list that action under the person who agreed, with "
            "any stated time. Mention each item once: work without an owner appears "
            "only in the final list. Include only what was said, with no empty "
            "categories and no closing summary."
        ),
        "language": "auto",
        "format": "markdown",
    },
}

# Pre-seeded once into the user's custom templates (editable + deletable).
SAMPLE_TEMPLATE = {
    "id": "shareable-summary",
    "name": "Shareable summary",
    "icon": "megaphone",
    "prompt": (
        "Write a clear, plain-language summary I can forward to a colleague or "
        "manager: the key points, decisions, and any next steps. Write in the "
        "language of the meeting."
    ),
    "language": "auto",
    "format": "markdown",
}


def new_template_id(name: str, existing_ids: set) -> str:
    """A stable slug id from a display name, de-duped against existing ids."""
    base = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    base = base or "template"
    if base not in existing_ids:
        return base
    n = 2
    while f"{base}-{n}" in existing_ids:
        n += 1
    return f"{base}-{n}"


def validate_template(t: dict, valid_languages: set) -> tuple:
    """Return (ok, error_message) for a template dict.

    Defensive at the Python trust boundary: `t` arrives from the renderer as
    decoded JSON and may be malformed. This never raises — it always returns
    (False, "<message>") on bad input.
    """
    if not isinstance(t, dict):
        return False, "Invalid template payload"

    name = t.get("name")
    if not isinstance(name, str):
        return False, "Template name is required"
    name = name.strip()
    if not name:
        return False, "Template name is required"
    if len(name) > MAX_NAME_LEN:
        return False, f"Template name is too long (max {MAX_NAME_LEN} characters)"

    prompt = t.get("prompt")
    if not isinstance(prompt, str):
        return False, "Template prompt is required"
    if not prompt.strip():
        return False, "Template prompt is required"
    if len(prompt) > MAX_PROMPT_LEN:
        return False, f"Template prompt is too long (max {MAX_PROMPT_LEN} characters)"

    lang = t.get("language", "auto")
    if not isinstance(lang, str) or lang not in valid_languages:
        return False, f"Unsupported language: {lang}"

    fmt = t.get("format")
    if fmt is not None and (not isinstance(fmt, str) or fmt not in VALID_FORMATS):
        return False, f"Unsupported format: {fmt}"

    icon = t.get("icon")
    if icon is not None:
        if not isinstance(icon, str):
            return False, "Invalid template icon"
        if len(icon) > MAX_ICON_LEN:
            return False, "Template icon is too long"

    return True, ""


def merge_templates(overrides: dict, custom: list) -> list:
    """Built-ins (with overrides applied) first, then custom templates.

    Each entry is tagged `builtin` (and `locked` for STANDARD) so the UI knows
    which controls (Reset vs Edit/Delete) to show.
    """
    result = []
    for tid, base in BUILTIN_TEMPLATES.items():
        merged = {**base, **(overrides.get(tid) or {})}
        merged["builtin"] = True
        merged["locked"] = bool(base.get("locked"))
        result.append(merged)
    for c in custom:
        result.append({**c, "builtin": False, "locked": False})
    return result
