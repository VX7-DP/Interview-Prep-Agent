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
import re
import sys

from dotenv import load_dotenv

from guardrails import check_input, check_url
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
        "--url", "-u",
        metavar="URL",
        help="URL of a job posting. The agent fetches it via the MCP fetch server.",
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


def _save_plan(plan_text: str) -> str:
    """
    Save the finished prep plan to output/ as a timestamp-free, slugged markdown
    file. Done in plain Python (not via MCP) so the artifact is always produced;
    the MCP server in this project is the `fetch` tool used for URL input.

    Returns the path written.
    """
    from agent import OUTPUT_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Derive a filename from the company name in the plan's opening line only
    # (e.g. "...role at Acme Corp."). Restricting to the first line avoids matching
    # stray tokens deeper in the text like "Lambda@Edge".
    slug = "prep_plan"
    first_line = plan_text.strip().splitlines()[0] if plan_text.strip() else ""
    m = re.search(r"\b(?:at|@)\s+([A-Z][A-Za-z0-9 .&-]{1,40}?)(?:[.,]|\s+role|$)", first_line)
    if m:
        slug = "prep_plan_" + re.sub(r"[^A-Za-z0-9]+", "_", m.group(1).strip()).strip("_")
    path = os.path.join(OUTPUT_DIR, f"{slug}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(plan_text)
    return path


async def run_agent(agent_input: str, profile: dict, days: int) -> None:
    """
    Async core: build the ADK agent, run it with InMemoryRunner, stream output,
    and save the finished plan to output/.

    *agent_input* is the message handed to the agent — either the pasted posting
    text, or an instruction to fetch a URL (in which case the agent calls the MCP
    `fetch` tool first). InMemoryRunner handles session/artifact management
    in-process — no external server — keeping everything locally run.
    """
    # Import here (after load_dotenv) so GEMINI_API_KEY is already in the env
    # when the ADK initializes the Gemini client.
    from google.adk.runners import InMemoryRunner
    from agent import create_agent, log_tool_call

    user_summary = profile_summary(profile)
    root_agent = create_agent(user_summary, days_to_prep=days)

    runner = InMemoryRunner(
        agent=root_agent,
        app_name="interview_prep_agent",
    )

    session = await runner.session_service.create_session(
        app_name="interview_prep_agent",
        user_id="local_user",
    )

    log_tool_call("run_agent", f"session={session.id} days={days}")

    from google.genai import types as genai_types

    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=agent_input)],
    )

    print("\n── Generating your personalized prep plan ──")
    print("(This may take 30–60 seconds — the agent extracts, researches, and plans.)\n")

    # Stream the agent's response event by event.
    final_response = ""
    async for event in runner.run_async(
        user_id="local_user",
        session_id=session.id,
        new_message=user_message,
    ):
        if getattr(event, "content", None):
            for part in event.content.parts:
                if getattr(part, "text", None):
                    print(part.text, end="", flush=True)
                    final_response += part.text

    log_tool_call("run_agent_complete", f"response_len={len(final_response)}")

    if final_response.strip():
        saved = _save_plan(final_response)
        print(f"\n\n── Done. Plan saved to {saved} ──\n")
    else:
        print("\n\n── Done, but the agent returned no text. Check agent.log. ──\n")


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

    # ── Step 2: Input + guardrails ───────────────────────────────────────────
    # Two modes:
    #   URL mode  (--url):  validate the URL (scheme + SSRF), then the AGENT
    #                       fetches the posting via the MCP `fetch` tool.
    #   Text mode (default): read pasted/file text, run the job-posting guardrail.
    if args.url:
        result = check_url(args.url)
        if not result:
            print(f"\n[REJECTED] URL rejected: {result.reason}\n", file=sys.stderr)
            sys.exit(1)
        print(f"\n[OK] URL guardrail passed. The agent will fetch: {args.url}\n")
        agent_input = (
            f"Fetch this job posting URL and build my interview prep plan: {args.url}"
        )
    else:
        posting_text = read_posting(args.file)
        result = check_input(posting_text)
        if not result:
            print(f"\n[REJECTED] Input rejected: {result.reason}\n", file=sys.stderr)
            sys.exit(1)
        print(f"\n[OK] Guardrail passed. Running agent with {args.days}-day prep schedule...\n")
        agent_input = f"Here is the job posting:\n\n{posting_text}"

    # ── Step 3: Run agent ────────────────────────────────────────────────────
    if not os.getenv("GEMINI_API_KEY"):
        print(
            "ERROR: GEMINI_API_KEY not set.\n"
            "  1. Copy .env.example -> .env\n"
            "  2. Add your key from https://aistudio.google.com/apikey\n",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(run_agent(agent_input, profile, args.days))


if __name__ == "__main__":
    main()
