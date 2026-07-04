# Interview Prep Agent

**Kaggle Capstone**

> Paste a job description. Get a personalized interview prep plan — skill-gap analysis, 10 technical questions, 5 behavioral questions, and a day-by-day study schedule — tailored to YOUR background, stored only on YOUR machine.

---

## Problem

Job seekers waste hours hunting for generic interview guides that don't match their background or the specific role. A senior Python engineer preparing for a Go-heavy infrastructure role needs different prep than a new grad aiming at the same posting. No tool currently bridges that gap automatically.

**Who it's for:** Software engineers at any level who want fast, targeted interview preparation without sharing their career history with a third-party service.

---

## Solution

An **ADK-powered multi-agent system** that:
1. Extracts structured requirements from any job posting (role, skills, seniority)
2. Researches the company via Google Search to ground the prep in real context
3. Computes a skill-gap analysis against your locally-stored career profile
4. Generates a complete, personalized prep plan and saves it as a Markdown file via MCP

Your profile (`profile.json`) never leaves your machine — personal data stays local by design.

---

## Architecture

```
User provides job posting (paste or file)
         │
   [guardrails.py]  ← concept 2 · Security
   Reject off-topic or prompt-injection input
         │ (accepted)
   root_agent  (ADK LlmAgent, gemini-2.5-flash)   ← concept 1 · ADK Agent
   ├── extract_requirements()  → structured JSON
   │   {role, company, must_have_skills, nice_to_have, seniority}
   │
   ├── company_research_agent  (AgentTool sub-agent)   ← multi-agent
   │   └── google_search   → company facts, tech stack, interview style
   │
   ├── generate_prep_plan()   → plan skeleton for LLM to expand
   │
   └── MCPToolset  ← concept 3 · MCP Server
       official `fetch` server (mcp-server-fetch, Python, no Node)
       When given a URL, the agent fetches the posting via MCP
         │
   main.py saves the finished plan to output/prep_plan_<company>.md
   agent.log  ← every tool call timestamped (observability)
   profile.json  ← loaded/saved across sessions (memory, gitignored)
```

---

## Course Concepts Demonstrated

| # | Concept | Location |
|---|---------|----------|
| 1 | **ADK Agent / multi-agent system** | `agent.py` — `create_agent()`, `search_sub_agent`, `search_agent_tool` |
| 2 | **Security features** (input guardrail + URL/SSRF guardrail + security tests) | `guardrails.py` — `check_input()`, `check_url()` · `tests/test_agent.py` |
| 3 | **MCP server** (official `fetch` server) | `agent.py` — `build_mcp_toolset()` |

---

## Setup

### Prerequisites
- Python 3.13
- A [Gemini API key](https://aistudio.google.com/apikey) (free tier works)

(No Node.js required — the MCP `fetch` server runs in the same Python venv.)

### Install

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/interview-prep-agent.git
cd interview-prep-agent

# 2. Create and activate a virtual environment
py -3.13 -m venv .venv          # Windows
# python3.13 -m venv .venv      # Mac/Linux

.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Gemini API key
copy .env.example .env           # Windows
# cp .env.example .env           # Mac/Linux

# Edit .env and paste your key:
# GEMINI_API_KEY=AIza...
```

### Run

```bash
python main.py                          # interactive paste mode
python main.py --file posting.txt       # read posting from a file
python main.py --url https://…/job/123  # agent fetches the posting via MCP
python main.py --days 14                # 14-day prep schedule
```

In `--url` mode the URL is first checked by `check_url()` (http/https only, no
loopback/private hosts — SSRF protection), then the agent calls the MCP `fetch`
tool to retrieve the posting. In text/file mode the posting runs through
`check_input()` (job-posting validation + prompt-injection detection).

On **first run**, you'll be prompted for your background (role, stack, years of experience). This is saved to `profile.json` (gitignored). All subsequent runs load it silently.

### Run Tests

```bash
pytest tests/ -v
# 20 offline tests pass immediately (no API key needed)
# 1 live end-to-end test runs if GEMINI_API_KEY is set (skips otherwise,
#   and skips gracefully if the free-tier daily quota is exhausted)
```

---

## Example Run

```
Loaded profile: Software Engineer II (4 yrs exp)

── Paste the job posting below ──
[paste posting, then Ctrl-Z / Ctrl-D]

✓ Guardrail passed. Running agent with 7-day prep schedule…

── Generating your personalized prep plan ──

## Interview Prep Plan: Senior Backend Engineer @ Acme Corp

### Skill Gap Analysis
You already have: Python (strong), FastAPI (strong)
Gaps to close: Go, Kubernetes, Kafka — here's the fastest path for each…

### 10 Technical Questions
1. Design a rate limiter for a public API…
[… 9 more tailored questions …]

### 5 Behavioral Questions
1. Tell me about a time you led a technical design decision under time pressure…
[… 4 more STAR prompts …]

### 7-Day Study Schedule
Day 1 (Go fundamentals): …
…

── Done. Check output/ for the saved Markdown plan. ──
```

---

## Limitations

- Google Search results vary; company research may be outdated for small companies.
- `--url` mode depends on the target page being fetchable server-side; JavaScript-rendered or bot-blocked postings may not extract cleanly. Pasting the text (`--file`/paste) always works.
- Live agent runs consume Gemini API quota — the free tier allows **20 requests/day** for `gemini-2.5-flash`, and a full run uses several. Heavy testing can exhaust the daily quota (the live test skips cleanly when it does).

---

## How It Was Built

Vibe-coded with an AI coding assistant. Every prompt is logged in [`prompts/vibe_log.md`](prompts/vibe_log.md) for full transparency.

ADK imports and MCP wiring were verified against [adk.dev](https://adk.dev) live docs during each session — the docs moved from `google.github.io/adk-docs` to `adk.dev` mid-build, which the vibe log captures.
