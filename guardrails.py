"""
guardrails.py — Input validation before the agent runs.

Security concept (course Day 4): untrusted user input is the primary
prompt-injection surface. This module is the mitigation layer:
  1. Reject non-job-posting input (off-topic text like recipes or personal notes).
  2. Detect and flag prompt-injection patterns embedded inside the posting
     so the agent never blindly executes injected instructions.

The check is intentionally DETERMINISTIC (no LLM calls), making it:
  - Fast (sub-millisecond)
  - Fully testable offline without a Gemini API key
  - A true security boundary that cannot be bypassed by a clever prompt
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── Vocabulary that strongly signals a job posting ──────────────────────────

# These phrases appear in the "requirements / responsibilities / about us"
# sections of virtually every real job description.
JOB_SIGNALS = [
    # Responsibilities section markers
    r"\b(responsibilities|you will|you'll)\b",
    r"\b(key duties|day.to.day|day-to-day)\b",
    # Requirements / qualifications markers
    r"\b(requirements|qualifications|must.have|must have|required skills)\b",
    r"\b(nice.to.have|nice to have|preferred|bonus points)\b",
    r"\b(years of experience|year[s]? experience)\b",
    r"\b(bachelor|master|degree|bs/ms|b\.s\.|m\.s\.)\b",
    # Role / hiring markers
    r"\b(we are (hiring|looking|seeking)|we're (hiring|looking|seeking))\b",
    r"\b(job title|position|role|opening|vacancy)\b",
    r"\b(full.?time|part.?time|contract|remote|hybrid|on.?site)\b",
    r"\b(salary|compensation|benefits|equity|stock options|pto)\b",
    r"\b(about (us|the (team|role|company|position)))\b",
    r"\b(equal opportunity employer|eoe)\b",
    # Tech / skill vocabulary common in job ads
    r"\b(proficiency|proficient|experience with|familiarity with|knowledge of)\b",
    r"\b(collaborate|cross.functional|stakeholder)\b",
]

# Prompt-injection patterns — attacker embeds these inside a posting to hijack
# the agent's instructions. We detect and reject rather than silently pass.
INJECTION_PATTERNS = [
    r"ignore (all |previous |prior |your )(instructions?|prompts?|context|rules?)",
    r"disregard (all |previous |your )?(instructions?|rules?|context)",
    r"forget (everything|all|your|prior|previous)",
    r"new (instructions?|task|objective|goal|role)[\s:]+",
    r"system\s*prompt",
    r"reveal (your|the) (system prompt|instructions?|context|secret)",
    r"print (your|the) (system prompt|instructions?)",
    r"act as (a |an )?(different|new|another|evil|unfiltered)",
    r"you are now",
    r"jailbreak",
    r"pretend (you are|to be|that)",
    r"override (your|all|previous) (instructions?|rules?|context)",
]

# Minimum document length (characters) for a plausible job posting.
MIN_LENGTH = 80


@dataclass
class GuardrailResult:
    """Outcome of the input check."""
    allowed: bool
    reason: str  # human-readable explanation (shown to user on rejection)

    def __bool__(self) -> bool:
        return self.allowed


def check_input(text: str) -> GuardrailResult:
    """
    Validate that *text* is a plausible job posting, not off-topic input
    or a prompt-injection attempt.

    Returns a GuardrailResult whose .allowed is True only when the text
    looks like a genuine job description. Call this BEFORE passing anything
    to the ADK agent.

    Decision logic:
      1. Reject if too short — real postings are never a single sentence.
      2. Reject immediately on any injection pattern — security-first.
      3. Score against job-posting vocabulary; require enough signal words
         to distinguish a real posting from unrelated text.
    """
    if not text or not text.strip():
        return GuardrailResult(
            allowed=False,
            reason="Input is empty. Please paste a job description."
        )

    lowered = text.lower()

    # Guard 1 — length check
    if len(text.strip()) < MIN_LENGTH:
        return GuardrailResult(
            allowed=False,
            reason=(
                "Input is too short to be a job description. "
                "Please paste the full posting text."
            )
        )

    # Guard 2 — prompt-injection detection (security boundary)
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailResult(
                allowed=False,
                reason=(
                    "Input appears to contain instructions attempting to alter "
                    "agent behavior (prompt injection). This is not allowed. "
                    "Please provide a plain job description."
                )
            )

    # Guard 3 — job-posting vocabulary scoring
    # Count how many distinct signal categories match.
    matched = sum(
        1 for sig in JOB_SIGNALS if re.search(sig, lowered)
    )

    # A genuine job posting will hit at least 3 of the ~16 signal categories.
    REQUIRED_SIGNALS = 3
    if matched < REQUIRED_SIGNALS:
        return GuardrailResult(
            allowed=False,
            reason=(
                f"This doesn't look like a job description "
                f"(matched {matched}/{REQUIRED_SIGNALS} required job-posting signals). "
                "Please paste the text of a real job posting."
            )
        )

    return GuardrailResult(allowed=True, reason="Input accepted as a job description.")
