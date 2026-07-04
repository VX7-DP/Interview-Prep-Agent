"""
memory.py — Persistent user profile for the interview prep agent.

The profile is stored in profile.json, which is:
  - Gitignored → personal data never leaves the user's machine (privacy by design).
  - Plain JSON, no database → zero infrastructure required.
  - Loaded silently on subsequent runs so the prep plan personalizes against
    the user's actual background without re-prompting.

Profile schema:
    {
        "current_role":  "Software Engineer II",
        "years_exp":     4,
        "tech_stack":    ["Python", "FastAPI", "PostgreSQL"],
        "target_roles":  ["Senior SWE", "Staff Engineer"],
        "notes":         ""   # freeform extra context
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROFILE_PATH = Path("profile.json")


def load_profile() -> dict[str, Any] | None:
    """
    Return the saved profile dict, or None if no profile exists yet.
    Designed to be called at startup; returns None signals "first run".
    """
    if not PROFILE_PATH.exists():
        return None
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupted profile — treat as no profile rather than crashing.
        return None


def save_profile(profile: dict[str, Any]) -> None:
    """
    Persist the profile dict to profile.json.
    Overwrites any existing file so updates are always applied.
    """
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def create_profile() -> dict[str, Any]:
    """
    Interactively collect the user's background on first run.
    Returns the populated profile dict (caller is responsible for saving).

    Kept simple: a few targeted questions that give the agent enough
    context to compute meaningful skill gaps.
    """
    print("\n── First run detected. Let's build your career profile ──")
    print("(This is stored locally in profile.json — never uploaded.)\n")

    current_role = input("Your current job title (e.g. 'Software Engineer II'): ").strip()
    years_exp_raw = input("Years of professional experience (e.g. '4'): ").strip()
    try:
        years_exp = int(years_exp_raw)
    except ValueError:
        years_exp = 0

    stack_raw = input(
        "Key technologies you know (comma-separated, e.g. 'Python, React, AWS'): "
    ).strip()
    tech_stack = [t.strip() for t in stack_raw.split(",") if t.strip()]

    targets_raw = input(
        "Target roles you're aiming for (comma-separated, e.g. 'Senior SWE, EM'): "
    ).strip()
    target_roles = [r.strip() for r in targets_raw.split(",") if r.strip()]

    notes = input(
        "Anything else the agent should know about you? (press Enter to skip): "
    ).strip()

    profile: dict[str, Any] = {
        "current_role": current_role,
        "years_exp": years_exp,
        "tech_stack": tech_stack,
        "target_roles": target_roles,
        "notes": notes,
    }

    print("\nProfile saved to profile.json ✓\n")
    return profile


def profile_summary(profile: dict[str, Any]) -> str:
    """
    Render the profile as a compact string for injection into agent prompts.
    The agent sees this so it can compute skill gaps and personalize output.
    """
    stack = ", ".join(profile.get("tech_stack", [])) or "not specified"
    targets = ", ".join(profile.get("target_roles", [])) or "not specified"
    notes = profile.get("notes", "")
    lines = [
        f"Current role: {profile.get('current_role', 'unknown')}",
        f"Years of experience: {profile.get('years_exp', 'unknown')}",
        f"Tech stack: {stack}",
        f"Target roles: {targets}",
    ]
    if notes:
        lines.append(f"Additional context: {notes}")
    return "\n".join(lines)
