"""
main.py — CLI entry point for Interview Prep Agent.

Usage:
    python main.py                     # interactive mode (paste posting)
    python main.py --file posting.txt  # read posting from a file
    python main.py --days 14           # set number of prep days (default 7)
    python main.py --help              # show this help

Flow:
    1. Load (or create on first run) the user profile from profile.json.
    2. Read the job posting from stdin or a file.
    3. Run guardrails — reject off-topic or prompt-injecting input immediately.
    4. Run the ADK agent via InMemoryRunner and stream the response.
    5. The agent auto-saves the plan to output/ via MCP; also print to terminal.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

from guardrails import check_input
from memory import create_profile, load_profile, profile_summary, save_profile

# Force UTF-8 on stdout/stderr so status output (which uses box-drawing and
# em-dash characters) doesn't crash on legacy Windows consoles (cp1252).
# reconfigure() exists on Python 3.7+; guarded in case the stream lacks it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

load_dotenv()  # reads GEMINI_API_KEY from .env before importing agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interview Prep Agent — paste a job posting, get a personalized study plan."
    )
    parser.add_argument(
        "--file", "-f",
        metavar="PATH",
        help="Path to a text file containing the job posting (default: interactive paste).",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=7,
        metavar="N",
        help="Number of days available to prepare (default: 7).",
    )
    return parser.parse_args()


def read_posting(file_path: str | None) -> str:
    """
    Read the job posting text from a file or from interactive stdin paste.
    For stdin, the user terminates input with Ctrl-Z (Windows) or Ctrl-D (Unix).
    """
    if file_path:
        with open(file_path, encoding="utf-8") as f:
            return f.read()

    print("\n── Paste the job posting below ──")
    print("(When done, press Enter then Ctrl-Z on Windows / Ctrl-D on Mac/Linux)\n")
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    return "\n".join(lines)


async def run_agent(posting_text: str, profile: dict, days: int) -> None:
    """
    Async core: build the ADK agent, run it with InMemoryRunner, and stream output.

    InMemoryRunner handles session management and artifact storage in-process —
    no external database or server required, which aligns with the "personal,
    locally-run agent" Concierge track pitch.
    """
    # Import here (after load_dotenv) so GEMINI_API_KEY is already in the env
    # when the ADK initializes the Gemini client.
    from google.adk.runners import InMemoryRunner
    from google.adk.sessions import InMemorySessionService
    from agent import create_agent, log_tool_call

    user_summary = profile_summary(profile)
    root_agent = create_agent(user_summary, days_to_prep=days)

    # InMemoryRunner wires together the agent, session service, and artifact
    # service so we can execute turns without standing up a server.
    runner = InMemoryRunner(
        agent=root_agent,
        app_name="interview_prep_agent",
    )

    session_service = runner.session_service
    session = await session_service.create_session(
        app_name="interview_prep_agent",
        user_id="local_user",
    )

    log_tool_call("run_agent", f"session={session.id} days={days}")

    # Build the user turn — pass the posting as the message content.
    from google.genai import types as genai_types

    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=f"Here is the job posting:\n\n{posting_text}")],
    )

    print("\n── Generating your personalized prep plan ──\n")
    print("(This may take 30–60 seconds — the agent extracts, researches, and plans.)\n")

    # Stream the agent's response event by event.
    final_response = ""
    async for event in runner.run_async(
        user_id="local_user",
        session_id=session.id,
        new_message=user_message,
    ):
        # ADK emits various event types; we print final text responses.
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(part.text, end="", flush=True)
                    final_response += part.text

    print("\n\n── Done. Check output/ for the saved Markdown plan. ──\n")
    log_tool_call("run_agent_complete", f"response_len={len(final_response)}")


def main() -> None:
    args = parse_args()

    # ── Step 1: Profile ──────────────────────────────────────────────────────
    profile = load_profile()
    if profile is None:
        profile = create_profile()
        save_profile(profile)
    else:
        print(f"\nLoaded profile: {profile.get('current_role', '?')} "
              f"({profile.get('years_exp', '?')} yrs exp)\n")

    # ── Step 2: Read posting ─────────────────────────────────────────────────
    posting_text = read_posting(args.file)

    # ── Step 3: Guardrails ───────────────────────────────────────────────────
    result = check_input(posting_text)
    if not result:
        print(f"\n[REJECTED] Input rejected: {result.reason}\n", file=sys.stderr)
        sys.exit(1)

    print(f"\n[OK] Guardrail passed. Running agent with {args.days}-day prep schedule...\n")

    # ── Step 4: Run agent ────────────────────────────────────────────────────
    if not os.getenv("GEMINI_API_KEY"):
        print(
            "ERROR: GEMINI_API_KEY not set.\n"
            "  1. Copy .env.example → .env\n"
            "  2. Add your key from https://aistudio.google.com/apikey\n",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(run_agent(posting_text, profile, args.days))


if __name__ == "__main__":
    main()
