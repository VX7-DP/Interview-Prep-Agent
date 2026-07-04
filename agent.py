"""
agent.py — ADK agent definition for Interview Prep Agent.

Architecture (three course concepts wired here):

  Concept 1 — ADK multi-agent system:
    root_agent  ← main orchestrator on gemini-2.5-flash
      ├── extract_requirements (function tool)   → structured JSON from posting
      ├── generate_prep_plan   (function tool)   → personalized plan text
      └── search_agent_tool    (AgentTool)        → wraps a dedicated search sub-agent
              └── search_sub_agent on gemini-2.5-flash with google_search only
                  (google_search CANNOT share an agent with other tools — ADK limitation)

  Concept 3 — MCP toolset:
    filesystem_mcp  → MCPToolset backed by @modelcontextprotocol/server-filesystem
    The agent uses this to write the finished prep plan to output/ as a .md file,
    making the MCP action visible in the demo.

  Observability (Day-4 framing):
    Every tool call is appended to agent.log with a UTC timestamp so we can
    reconstruct exactly what the agent did and when.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv

# ── ADK imports (verified against google-adk==2.3.0) ───────────────────────
from google.adk.agents.llm_agent import Agent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

load_dotenv()  # reads GEMINI_API_KEY from .env

# Output directory where the MCP filesystem server is allowed to write.
OUTPUT_DIR = os.path.abspath("output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = "agent.log"
MODEL = "gemini-2.5-flash"


# ── Observability helper ─────────────────────────────────────────────────────

def log_tool_call(tool_name: str, args_preview: str = "") -> None:
    """
    Append a timestamped line to agent.log for every tool invocation.
    This gives us a full audit trail of what the agent did — useful for
    debugging and for demonstrating Day-4 observability in the writeup.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = f"[{ts}] tool={tool_name} args={args_preview[:120]}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


def _before_tool_logger(tool, args, tool_context):
    """
    ADK before_tool_callback — fires before EVERY tool the agent invokes,
    including the MCP filesystem tools and the company-research AgentTool that
    the plain function-tool logging can't reach on its own.

    Signature required by ADK: (tool, args, context) -> Optional[dict].
    Returning None lets the tool run normally; we only observe, never mutate.
    This is what makes the plan's "the MCP tool call appears in agent.log" true.
    """
    tool_name = getattr(tool, "name", type(tool).__name__)
    try:
        args_preview = json.dumps(args, default=str)
    except (TypeError, ValueError):
        args_preview = str(args)
    log_tool_call(tool_name, args_preview)
    return None  # do not override the tool's normal execution


# ── Function tools (concept 1) ───────────────────────────────────────────────

# Section headers that introduce required vs. preferred skills. Used to bucket
# bullet points into must_have vs. nice_to_have during extraction.
_MUST_HAVE_HEADERS = re.compile(
    r"(requirements?|qualifications?|must[- ]have|required skills|what you.?ll need|"
    r"what we.?re looking for|minimum qualifications?)",
    re.IGNORECASE,
)
_NICE_HEADERS = re.compile(
    r"(nice[- ]to[- ]have|preferred|bonus|plus(es)?|good to have|"
    r"desired|preferred qualifications?)",
    re.IGNORECASE,
)
# Section boundaries — a new top-level section ends the current skill bucket.
_OTHER_HEADERS = re.compile(
    r"(responsibilities|about (us|the|this)|what you.?ll do|benefits|"
    r"compensation|salary|perks|why join|our team|how to apply|equal opportunity)",
    re.IGNORECASE,
)
# Technology / skill catalogue. Each entry maps one or more lowercase search
# terms (matched as whole words inside the posting) to a canonical display name.
# Whole-word matching (\b…\b) avoids noise like matching "r" inside "product".
_SKILL_CATALOGUE = {
    "Python": ["python"], "Java": ["java"], "JavaScript": ["javascript"],
    "TypeScript": ["typescript"], "Go": ["go", "golang"], "Rust": ["rust"],
    "C++": ["c\\+\\+"], "C#": ["c#"], "Ruby": ["ruby"], "PHP": ["php"],
    "Scala": ["scala"], "Kotlin": ["kotlin"], "Swift": ["swift"],
    "SQL": ["sql"], "NoSQL": ["nosql"],
    "React": ["react"], "Angular": ["angular"], "Vue": ["vue"],
    "Node.js": ["node", "node.js", "nodejs"], "Django": ["django"],
    "Flask": ["flask"], "FastAPI": ["fastapi"], "Spring": ["spring"],
    "Rails": ["rails"], ".NET": ["\\.net"], "GraphQL": ["graphql"],
    "REST": ["rest"], "gRPC": ["grpc"],
    "PostgreSQL": ["postgresql", "postgres"], "MySQL": ["mysql"],
    "MongoDB": ["mongodb"], "Redis": ["redis"], "Cassandra": ["cassandra"],
    "DynamoDB": ["dynamodb"], "Elasticsearch": ["elasticsearch"],
    "Kafka": ["kafka"], "Kinesis": ["kinesis"], "RabbitMQ": ["rabbitmq"],
    "Spark": ["spark"], "Hadoop": ["hadoop"], "Snowflake": ["snowflake"],
    "Databricks": ["databricks"], "Airflow": ["airflow"],
    "AWS": ["aws"], "Azure": ["azure"], "GCP": ["gcp", "google cloud"],
    "Kubernetes": ["kubernetes", "k8s"], "Docker": ["docker"],
    "Terraform": ["terraform"], "Helm": ["helm"], "Jenkins": ["jenkins"],
    "CI/CD": ["ci/cd", "ci / cd"], "Linux": ["linux"],
    "Machine Learning": ["machine learning"], "Deep Learning": ["deep learning"],
    "PyTorch": ["pytorch"], "TensorFlow": ["tensorflow"], "NLP": ["nlp"],
    "Distributed Systems": ["distributed systems"], "Microservices": ["microservices"],
    "System Design": ["system design"], "Data Structures": ["data structures"],
    "Algorithms": ["algorithms"], "Agile": ["agile"], "Scrum": ["scrum"],
}


def _extract_skills(text: str) -> list:
    """
    Return the ordered, de-duplicated canonical skill names that appear in *text*.

    Matching is whole-word and case-insensitive so short terms (Go, SQL, R) don't
    match substrings inside unrelated words. Display names come from the catalogue,
    giving clean, consistently-cased output for the LLM to build on.
    """
    lowered = text.lower()
    found = []
    for display_name, terms in _SKILL_CATALOGUE.items():
        for term in terms:
            # \b word boundaries; term may itself be a regex fragment (e.g. c\+\+).
            if re.search(rf"(?<![\w]){term}(?![\w])", lowered):
                found.append(display_name)
                break  # one match is enough; move to next skill
    return found


def extract_requirements(posting_text: str) -> dict:
    """
    Parse a job posting into a structured dict the rest of the pipeline can act on.

    The ADK passes tool return values back into the model as context, so returning
    a clean JSON-serialisable dict here lets the LLM see a crisp, unambiguous
    summary rather than raw posting prose — reducing hallucination in downstream steps.
    The LLM still refines these fields (e.g. cleaning up the role title), but the tool
    does the real structural work so it is a meaningful tool, not a stub.

    Fields:
        role            – job title inferred from the first substantive line
        company         – company name if mentioned (via "at X" / "X |" header), else "Unknown"
        must_have_skills – skills found under requirements/must-have sections
        nice_to_have    – skills found under nice-to-have/preferred sections
        seniority       – "junior" | "mid" | "senior" | "staff" | "unknown"
    """
    log_tool_call("extract_requirements", posting_text[:80])

    lines = [ln.strip() for ln in posting_text.splitlines() if ln.strip()]
    lowered = posting_text.lower()

    # ── Role: first non-empty line is almost always the job title ──
    role = lines[0] if lines else "Unknown"
    # Strip trailing " | Company | Remote" style decorations from the title.
    role = re.split(r"\s[|\-–—]\s", role)[0].strip() or "Unknown"

    # ── Company: "at <Company>", or the first non-location segment of a
    #    pipe-delimited header line (e.g. "Acme Corp | Remote | Full-time"). ──
    _LOCATION_OR_TYPE = re.compile(
        r"(remote|hybrid|on.?site|full.?time|part.?time|contract|freelance)",
        re.IGNORECASE)
    company = "Unknown"
    at_match = re.search(r"\bat\s+([A-Z][A-Za-z0-9&.\- ]{1,40})", posting_text)
    if at_match:
        company = at_match.group(1).strip().rstrip(".")
    else:
        # Scan the first few header lines. Only pipe-delimited lines carry a
        # company (e.g. "Acme Corp | Remote | Full-time"); the company is the
        # first segment that isn't the role, a location/type token, or a
        # section header (a segment ending in ":" like "Requirements:").
        for ln in lines[:3]:
            if "|" not in ln:
                continue
            parts = [p.strip() for p in re.split(r"\s[|]\s", ln) if p.strip()]
            for cand in parts:
                if cand == role or cand.endswith(":"):
                    continue
                if _LOCATION_OR_TYPE.search(cand):
                    continue  # skip "Remote", "Full-time", "NY", etc.
                company = cand
                break
            if company != "Unknown":
                break

    # ── Seniority heuristics — check staff/principal before senior ──
    seniority = "unknown"
    for level in ["staff", "principal", "director", "vp ", "head of"]:
        if level in lowered:
            seniority = "staff"
            break
    if seniority == "unknown":
        for level in ["senior", "sr.", "sr "]:
            if level in lowered:
                seniority = "senior"
                break
    if seniority == "unknown":
        for level in ["junior", "jr.", "jr ", "entry", "associate", "new grad"]:
            if level in lowered:
                seniority = "junior"
                break
    if seniority == "unknown":
        seniority = "mid"

    # ── Bucket skills by walking the posting section-by-section ──
    # We track which section we're in; bullets inherit the current bucket.
    must_text, nice_text = [], []
    bucket = None  # None | "must" | "nice"
    for ln in lines:
        if _NICE_HEADERS.search(ln):
            bucket = "nice"
            nice_text.append(ln)
        elif _MUST_HAVE_HEADERS.search(ln):
            bucket = "must"
            must_text.append(ln)
        elif _OTHER_HEADERS.search(ln):
            bucket = None
        elif bucket == "must":
            must_text.append(ln)
        elif bucket == "nice":
            nice_text.append(ln)

    must_have_skills = _extract_skills("\n".join(must_text))
    nice_to_have = _extract_skills("\n".join(nice_text))

    # Fallback: if section parsing found nothing (unstructured posting),
    # scan the whole document so we never return an empty skill list.
    if not must_have_skills and not nice_to_have:
        must_have_skills = _extract_skills(posting_text)

    return {
        "role": role,
        "company": company,
        "must_have_skills": must_have_skills,
        "nice_to_have": nice_to_have,
        "seniority": seniority,
        "raw_posting_length": len(posting_text),
    }


def generate_prep_plan(
    role: str,
    company: str,
    must_have_skills: list,
    nice_to_have: list,
    seniority: str,
    user_profile_summary: str,
    days_to_prep: int = 7,
) -> dict:
    """
    Produce the skeleton of the personalized interview prep plan.

    This function returns a structured dict that the LLM agent uses as the
    basis for its final answer. Separating structure-building (this function)
    from language-generation (the LLM) keeps the output format stable while
    giving the LLM full flexibility over prose quality.

    The agent's instruction says: "call generate_prep_plan with the extracted
    fields, then expand each section into full, polished prose for the user."
    """
    log_tool_call(
        "generate_prep_plan",
        f"role={role} company={company} seniority={seniority} days={days_to_prep}"
    )

    return {
        "role": role,
        "company": company,
        "seniority": seniority,
        "must_have_skills": must_have_skills,
        "nice_to_have": nice_to_have,
        "user_profile": user_profile_summary,
        "days_available": days_to_prep,
        "sections_to_generate": [
            "skill_gap_analysis",       # compare must_have vs user's known stack
            "technical_questions_10",   # 10 technical Qs tailored to the role
            "behavioral_questions_5",   # 5 STAR-format behavioral Qs
            "day_by_day_schedule",      # study schedule spread over days_to_prep
            "resources",                # 3-5 specific learning resources per gap
        ],
    }


# ── Search sub-agent (concept 1 — multi-agent) ──────────────────────────────
# google_search can ONLY be used alone in a single agent instance (ADK limitation).
# Solution: a dedicated sub-agent that holds only google_search, then exposed to
# the root agent as an AgentTool. This satisfies the "multi-agent" requirement
# while working within ADK's tool-mixing constraints.

search_sub_agent = Agent(
    name="company_research_agent",
    model=MODEL,
    description=(
        "Researches companies using Google Search. "
        "Given a company name, returns key facts: industry, size, recent news, "
        "tech stack, engineering culture, interview style."
    ),
    instruction=(
        "You are a company research assistant. When given a company name, "
        "use google_search to find: the company's industry, approximate size, "
        "recent news (last 6 months), known tech stack, engineering blog or culture docs, "
        "and any public information about their interview process. "
        "Return a concise bulleted summary — 100–200 words maximum."
    ),
    tools=[google_search],
)

# Wrap the sub-agent so the root agent can call it like a regular tool.
search_agent_tool = AgentTool(agent=search_sub_agent)


# ── MCP filesystem toolset (concept 3) ──────────────────────────────────────
# Uses @modelcontextprotocol/server-filesystem (official MCP server, npx-launched).
# Scoped to OUTPUT_DIR so the agent can only write inside ./output/ — principle
# of least privilege applied to the MCP surface.
#
# The agent uses this to save the finished prep plan as output/prep_plan_<company>.md,
# making the MCP action visible and auditable in the demo.

def build_mcp_toolset() -> MCPToolset:
    """
    Construct the MCPToolset that connects to the filesystem MCP server.

    Launched via npx so no global install is required — npx pulls the package
    on first run and caches it. The server is scoped to OUTPUT_DIR only,
    so the agent cannot read or write outside of ./output/.
    """
    log_tool_call("build_mcp_toolset", f"output_dir={OUTPUT_DIR}")
    return MCPToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=[
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    OUTPUT_DIR,
                ],
            )
        )
    )


# ── Root agent (concepts 1 + 3) ─────────────────────────────────────────────

AGENT_INSTRUCTION = """\
You are an expert interview preparation coach. Your job is to analyze a job posting
and produce a comprehensive, personalized interview prep plan for the user.

## Workflow — follow these steps in order:

1. **Extract** — Call extract_requirements(posting_text=<the full job posting>) to parse
   the role, company, skills, and seniority level from the posting.

2. **Research** — Call company_research_agent with the company name to get background on
   the company's tech stack, culture, and interview style. Use this to make your questions
   more specific and realistic.

3. **Plan** — Call generate_prep_plan with all extracted fields plus the user's profile
   summary (provided below) and the number of prep days. This returns the structure;
   you will fill in the actual content.

4. **Write** — Expand the plan into a full, polished document with these sections:
   - **Skill Gap Analysis**: Compare must-have skills vs. the user's current stack.
     For each gap, explain what to study and why it matters for this specific role.
   - **10 Technical Interview Questions**: Tailored to the role and seniority.
     Include the question + what a strong answer should cover.
   - **5 Behavioral Interview Questions**: STAR-format prompts relevant to the company's
     culture and the role's seniority level.
   - **Day-by-Day Study Schedule**: Spread across the available prep days.
     Each day has a focus area, specific resources, and a practice task.
   - **Resources**: 3–5 specific books, courses, or docs per major skill gap.

5. **Save** — Use the MCP filesystem tool to write the finished plan to a file called
   prep_plan_<company_name>.md inside the output directory. Replace spaces with underscores
   in the company name. Log the save action.

## Personalization rules:
- Skills the user already knows: say "you already have X — focus on depth, not basics."
- Skills the user is missing: explain the gap and give the fastest learning path.
- Match question difficulty to seniority (junior = fundamentals, senior = system design + leadership).

## Format:
- Use Markdown with clear headers (##, ###).
- Keep the plan actionable — every section should tell the user exactly what to DO.
- Do NOT pad with filler. If the posting has limited info, say so and focus on the role type.
"""


def create_agent(user_profile_summary: str, days_to_prep: int = 7) -> Agent:
    """
    Build and return the root ADK agent, injecting the user's profile and
    prep-day count into the system instruction.

    Why factory function instead of module-level agent? The profile and days
    parameters are only known at runtime. Creating a new Agent instance per
    run keeps the instruction fresh and avoids state leakage between sessions.
    """
    log_tool_call("create_agent", f"days={days_to_prep}")

    personalized_instruction = (
        AGENT_INSTRUCTION
        + f"\n\n## User Profile (use this for personalization):\n{user_profile_summary}"
        + f"\n\n## Prep days available: {days_to_prep}"
    )

    mcp_toolset = build_mcp_toolset()

    return Agent(
        name="interview_prep_agent",
        model=MODEL,
        description="Analyzes job postings and generates personalized interview prep plans.",
        instruction=personalized_instruction,
        tools=[
            extract_requirements,
            generate_prep_plan,
            search_agent_tool,
            mcp_toolset,
        ],
        # Observability: log every tool call (incl. MCP + AgentTool) to agent.log.
        before_tool_callback=_before_tool_logger,
    )
