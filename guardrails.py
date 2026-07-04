"""
guardrails.py — Input validation before the agent runs.

Security concept (course Day 4): untrusted user input is the primary
prompt-injection surface. This module is the mitigation layer:
  1. Reject non-job-posting input (off-topic text like recipes or personal notes).
  2. Detect and flag prompt-injection patterns embedded inside the posting
     so the agent never blindly executes injected instructions.
  3. Validate user-supplied URLs before the agent fetches them via MCP
     (check_url): scheme allowlist + private/loopback-host block. Fetching a URL
     pulls untrusted web content into the agent — the classic prompt-injection and
     SSRF surface — so we constrain WHAT can be fetched before it ever runs.

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
#
# These are deliberately NARROW: they match agent-directed imperative phrasing
# ("ignore your instructions", "reveal your system prompt") but NOT benign
# posting language that happens to share vocabulary ("a new role", "act as a
# point of contact", "our new goal"). Over-broad patterns caused false
# rejections of legitimate postings, so each pattern below requires the
# instruction-hijacking structure, not just a keyword.
INJECTION_PATTERNS = [
    # "ignore/disregard [your/all/previous] instructions/rules/prompt"
    r"ignore (all |the |your |previous |prior )*(instructions?|prompts?|context|rules?)",
    r"disregard (all |the |your |previous |prior )*(instructions?|rules?|context|prompts?)",
    # "forget your/all previous instructions/rules" — requires the instruction object
    r"forget (all |your |the )*(previous |prior )*(instructions?|rules?|context|prompts?)",
    # "your new instructions are:" / "here are your new instructions:"
    r"(here are |these are |follow )?(your )?new instructions?( (are|is))?\s*[:\-]",
    # Direct references to the system prompt / secrets
    r"system\s*prompt",
    r"reveal (your|the) (system prompt|instructions?|context|secret|api key)",
    r"print (your|the) (system prompt|instructions?|context)",
    r"(show|tell) me (your|the) (system prompt|instructions?|api key|secret)",
    # "act as an unfiltered/unrestricted/different AI/assistant"
    r"act as (a |an )?(different|unfiltered|unrestricted|evil|jailbroken|dan) ",
    # "you are now a <persona>" — the persona-swap jailbreak
    r"you are now (a |an )\w+",
    r"jailbreak",
    # "override your/all/previous instructions/rules"
    r"override (your|all|the|previous) (instructions?|rules?|context|prompts?)",
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


# ── URL guardrail (for --url mode, before the agent fetches via MCP) ─────────

from urllib.parse import urlparse
import ipaddress

# Only these URL schemes may be fetched. Blocks file://, javascript:, data:, etc.
_ALLOWED_SCHEMES = {"http", "https"}

# Host substrings that indicate a private / loopback / metadata target. Fetching
# these would be a Server-Side Request Forgery (SSRF) — the agent could be tricked
# into reading internal services. We reject them outright.
_BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal"}


def check_url(url: str) -> GuardrailResult:
    """
    Validate a user-supplied URL before the agent fetches it via the MCP `fetch`
    tool. This is the security boundary for --url mode (concept 2): fetching an
    arbitrary URL pulls untrusted content into the agent and can be abused for
    SSRF, so we constrain the target first.

    Rejects when:
      - the string isn't a parseable http(s) URL,
      - the scheme isn't http/https (blocks file://, data:, javascript:, etc.),
      - the host is loopback/localhost, a private/link-local IP, or a known
        cloud metadata endpoint.
    """
    if not url or not url.strip():
        return GuardrailResult(allowed=False, reason="No URL provided.")

    parsed = urlparse(url.strip())

    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return GuardrailResult(
            allowed=False,
            reason=(
                f"URL scheme '{parsed.scheme or '(none)'}' is not allowed. "
                "Only http:// and https:// URLs can be fetched."
            ),
        )

    host = (parsed.hostname or "").lower()
    if not host:
        return GuardrailResult(allowed=False, reason="URL has no host.")

    if host in _BLOCKED_HOSTS:
        return GuardrailResult(
            allowed=False,
            reason=f"Refusing to fetch internal/loopback host '{host}' (SSRF protection).",
        )

    # Block private, loopback, and link-local IP addresses (e.g. 10.x, 192.168.x,
    # 169.254.x). Hostnames that aren't IPs fall through and are allowed.
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return GuardrailResult(
                allowed=False,
                reason=f"Refusing to fetch private/reserved IP '{host}' (SSRF protection).",
            )
    except ValueError:
        pass  # not a literal IP — a normal hostname, allowed

    return GuardrailResult(allowed=True, reason="URL accepted for fetching.")
