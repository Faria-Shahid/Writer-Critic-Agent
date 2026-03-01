from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from typing import Optional
from dotenv import load_dotenv
import os
import requests
from datetime import datetime, timedelta
from tavily import TavilyClient
from model import load_llm
from AgentStateClass import AgentState
import json

load_dotenv() 

# ─────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────


@tool
def get_trending_now(geo: str = "US", category: int = None, hours: int = 48) -> str:
    """
    [DISCOVERY] Fetch what is CURRENTLY breaking on Google Trends.

    Returns a ranked list of trending topics with search volumes and
    related queries. Use this in Phase 1 to build your candidate pool.

    Args:
        geo:      2-letter country code — 'US', 'GB', 'AU' etc. (default 'US')
        category: Google Trends category ID to filter results.
                  0=All, 3=Business, 5=Entertainment, 7=Health,
                  8=Science, 12=Technology, 20=Sports (default None = All)
        hours:    Recency window — 4, 24, 48 or 168 hours (default 48)
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return "Error: SERPAPI_KEY not set."

    params = {
        "engine": "google_trends_trending_now",
        "geo": geo,
        "hours": hours,
        "api_key": api_key,
    }
    if category is not None:
        params["category"] = category

    resp = requests.get("https://serpapi.com/search.json", params=params)
    if resp.status_code != 200:
        return f"SerpAPI error {resp.status_code}: {resp.text}"

    trending = resp.json().get("trending_searches", [])
    if not trending:
        return "No trending searches returned. Try a different geo or category."

    lines = [f"🔥 Google Trending Now ({geo}, last {hours}h):\n"]
    for i, item in enumerate(trending[:15], 1):
        title = item.get("query", "N/A")
        volume = item.get("search_volume", item.get("formattedTraffic", "N/A"))
        related = ", ".join(item.get("related_queries", [])[:3]) or "—"
        articles = item.get("articles", [])
        top_article = articles[0].get("title", "") if articles else ""

        lines.append(
            f"{i}. {title}  |  Volume: {volume}\n"
            f"   Related: {related}\n"
            f"   {'📰 ' + top_article if top_article else ''}"
        )
    return "\n".join(lines)


@tool
def tavily_search(query: str, max_results: int = 8) -> str:
    """
    [DISCOVERY — FALLBACK] Search the web for niche-specific trends.

    Use this ONLY if get_trending_now returned no candidates with a credible
    bridge to the brand — i.e. every trending topic either failed hard filters
    or had no bridgeable connection.

    DO NOT use this as a parallel discovery source or to supplement Google Trends.
    It is a fallback for when Google's real-time signal is irrelevant to the brand.

    Good fallback queries:
      - "<brand niche> trends <current month year>"  e.g. "AI tools trends March 2026"
      - "rising topics in <industry>"                e.g. "rising topics in sustainable food"
      - "what is trending in <audience niche>"       e.g. "what is trending in no-code dev tools"

    Returns search results with titles, URLs, and snippets — scan for 2-3 candidate
    topics with genuine momentum, then proceed to validation as normal.

    Args:
        query:       Web search query — be specific about niche and timeframe
        max_results: Number of results to return (default 8)
    """
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not set."

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        time_range="w",
        search_depth="advanced",
        include_answer=True,
    )

    results = []
    if response.get("answer"):
        results.append(f"Summary: {response['answer']}\n")

    for i, r in enumerate(response.get("results", []), 1):
        results.append(
            f"{i}. {r['title']}\n"
            f"   URL: {r['url']}\n"
            f"   {r.get('content', '')[:250]}\n"
        )

    return "\n".join(results) if results else "No results found." 

@tool
def get_google_trends_trajectory(query: str, date_range: str = "today 3-m") -> str:
    """
    [VALIDATION] Check whether a specific topic is RISING, STABLE, or DECLINING.

    Use this in Phase 2 for one of your shortlisted candidates. A DECLINING
    result means the candidate should be dropped — don't waste the scoring
    call on a fading trend.

    Args:
        query:      Topic to check — keep concise, e.g. 'upcycled food'
        date_range: 'today 1-m', 'today 3-m', 'today 12-m' (default 'today 3-m')

    Returns:
        Trajectory label, % change, peak/current scores, and recent data points.
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return "Error: SERPAPI_KEY not set."

    resp = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google_trends",
            "q": query,
            "date": date_range,
            "data_type": "TIMESERIES",
            "tz": 420,
            "api_key": api_key,
        },
    )
    if resp.status_code != 200:
        return f"SerpAPI error {resp.status_code}: {resp.text}"

    timeline = resp.json().get("interest_over_time", {}).get("timeline_data", [])
    if not timeline:
        return f"No trend data for '{query}'. Treat as unverifiable — consider an alternative."

    points = [(p["date"], p["values"][0]["extracted_value"]) for p in timeline]
    values = [v for _, v in points]
    recent, earlier = values[-4:], values[:4]
    avg_recent = sum(recent) / len(recent)
    avg_earlier = sum(earlier) / len(earlier) or 1
    change_pct = ((avg_recent - avg_earlier) / avg_earlier) * 100
    peak, current = max(values), values[-1]

    if change_pct > 20:
        trajectory = "📈 RISING"
    elif change_pct < -20:
        trajectory = "📉 DECLINING — recommend dropping this candidate"
    else:
        trajectory = "➡️ STABLE"

    recent_summary = "\n".join(f"  {d}: {v}/100" for d, v in points[-6:])
    return (
        f"Trajectory for '{query}': {trajectory} ({change_pct:+.0f}% vs period start)\n"
        f"Peak: {peak}/100 | Current: {current}/100\n\n"
        f"Recent data points:\n{recent_summary}"
    )


@tool
def get_news_depth(query: str, days_back: int = 7, sort_by: str = "popularity") -> str:
    """
    [VALIDATION] Assess editorial depth and storytelling angles for a candidate trend.

    Use this in Phase 2 as the second validation call — AFTER trajectory confirms
    RISING or STABLE for your top candidate. Look for: volume of coverage, source
    quality, angle diversity, and recency (freshness = longevity signal).

    If the trajectory call returned DECLINING, call this on the alternative candidate.

    Args:
        query:     Trend keyword or phrase to investigate
        days_back: How many days back to search (default 7)
        sort_by:   'relevancy', 'popularity', or 'publishedAt' (default 'popularity')
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return "Error: NEWS_API_KEY not set."

    params = {
        "q": query,
        "from": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": sort_by,
        "pageSize": 5,
        "apiKey": api_key,
    }
    resp = requests.get("https://newsapi.org/v2/everything", params=params)
    if resp.status_code != 200:
        return f"NewsAPI error {resp.status_code}: {resp.text}"

    data = resp.json()
    articles = data.get("articles", [])
    if not articles:
        return f"No news coverage for '{query}'. Low editorial depth — consider an alternative."

    results = [f"Found {data['totalResults']} articles. Top {len(articles)}:\n"]
    for i, a in enumerate(articles, 1):
        results.append(
            f"{i}. {a['title']}\n"
            f"   Source: {a['source']['name']} | Published: {a.get('publishedAt', 'N/A')[:10]}\n"
            f"   {a.get('description', 'No description')[:200]}\n"
        )
    return "\n".join(results)


@tool
def score_and_recommend(
    brand_description: str,
    candidate_a: str,
    candidate_a_evidence: str,
    candidate_b: str,
    candidate_b_evidence: str,
) -> str:
    """
    [SCORING] Compare two validated candidates and select the winner.

    Use this as the FINAL tool call — only after both candidates have validation
    data from Phase 2. Pass structured evidence summaries for each candidate.
    This tool forces explicit comparison before committing to a recommendation.

    Args:
        brand_description:    The full brand context passed by the user
        candidate_a:          Topic/name of first candidate
        candidate_a_evidence: One-paragraph summary of all signals for A
                              (trajectory, news depth, reddit signal, brand fit)
        candidate_b:          Topic/name of second candidate
        candidate_b_evidence: One-paragraph summary of all signals for B
    """
    scores = {}
    for name, evidence in [(candidate_a, candidate_a_evidence), (candidate_b, candidate_b_evidence)]:
        # ── Trajectory magnitude (0-5) ──────────────────────────────
        # Extract % change if present in evidence, e.g. "+933%" or "-45%"
        import re
        pct_match = re.search(r'([+-]?\d+)%', evidence)
        pct_change = int(pct_match.group(1)) if pct_match else 0

        declining = "DECLINING" in evidence.upper()
        if declining or pct_change < -20:
            momentum = 0
        elif pct_change >= 500:
            momentum = 5   # explosive breakout
        elif pct_change >= 100:
            momentum = 4   # strong rise
        elif pct_change >= 20:
            momentum = 3   # solid rise
        else:
            momentum = 2   # stable

        # ── News depth (0-3) ────────────────────────────────────────
        # Reward volume of coverage, not just presence
        article_match = re.search(r'(\d+)\s+article', evidence.lower())
        article_count = int(article_match.group(1)) if article_match else 0
        if article_count >= 30:
            news_depth = 3
        elif article_count >= 10:
            news_depth = 2
        elif article_count > 0 or any(w in evidence.lower() for w in ["source", "news"]):
            news_depth = 1
        else:
            news_depth = 0

        # ── Bridge quality (0-2) ────────────────────────────────────
        # Penalise weak/forced bridges; reward direct brand connections
        forced_bridge = any(p in evidence.lower() for p in [
            "requires creative", "creative framing", "stretch", "forced",
            "moderate", "indirect", "weak connection"
        ])
        strong_bridge = any(p in evidence.lower() for p in [
            "direct", "natural", "perfect", "authentic", "core",
            "aligns perfectly", "directly relevant"
        ])
        bridge = 0 if forced_bridge else (2 if strong_bridge else 1)

        total = momentum + news_depth + bridge

        scores[name] = {
            "momentum (0-5)":   momentum,
            "news_depth (0-3)": news_depth,
            "bridge (0-2)":     bridge,
            "total":            total,
            "pct_change":       pct_change,
        }

    import json

    winner = max(scores, key=lambda k: scores[k]["total"])
    loser  = candidate_a if winner == candidate_b else candidate_b

    result = {
        "winner": {
            "topic":      winner,
            "score":      scores[winner]["total"],
            "breakdown": {
                "momentum":   scores[winner]["momentum (0-5)"],
                "news_depth": scores[winner]["news_depth (0-3)"],
                "bridge":     scores[winner]["bridge (0-2)"],
            },
            "pct_change": scores[winner]["pct_change"],
        },
        "runner_up": {
            "topic":      loser,
            "score":      scores[loser]["total"],
            "breakdown": {
                "momentum":   scores[loser]["momentum (0-5)"],
                "news_depth": scores[loser]["news_depth (0-3)"],
                "bridge":     scores[loser]["bridge (0-2)"],
            },
            "pct_change": scores[loser]["pct_change"],
        },
        "max_score": 10,
        "instruction": (
            f"Write the final recommendation for '{winner}' using the standard output format. "
            f"In the 'Why NOT' section, explain in one sentence why '{loser}' was rejected."
        ),
    }

    return json.dumps(result, indent=2)

# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Trend Intelligence Agent that recommends the single best content topic for a brand.

You operate in three strict phases: Discovery → Validation → Scoring.

Think step-by-step internally between phases.
Do NOT reveal intermediate reasoning.
Only return the final JSON output.

You must follow the required tool order exactly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚫 HARD FILTERS — DISCARD IMMEDIATELY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
These topics are never brand-safe under any framing. Drop without consideration:
- Politics, elections, government policy, geopolitics, political figures
- Sexual or adult content
- Religion or religious controversy
- Tragedy, mass casualty events, natural disasters, deaths
- Active legal controversies or criminal proceedings

Filtered topics are invisible — never mention them in output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ SOFT FILTER — REQUIRES A CREDIBLE BRIDGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
These topics may only be used if a natural, authentic connection to the brand exists.
If the connection feels forced, discard the topic.

Bridge test:
"Can this brand speak about this trend from its own authentic expertise without sounding opportunistic?"

If NO → discard.
If YES → proceed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — DISCOVERY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Call get_trending_now(geo="US", category=0, hours=24). (Always first.)
2. Apply hard filters.
3. Apply bridge test.
4. If fewer than 2 strong candidates remain, call:
   tavily_search(query="<brand niche> trends <current month year>")

Select the top 2 candidates with the strongest combination of:
- Momentum signal
- Editorial visibility
- Brand fit

Do not output this reasoning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Call get_google_trends_trajectory on Candidate A.
   - If DECLINING → discard A and validate B instead.
2. Call get_news_depth on the surviving candidate.

Use trajectory change %, news volume, and freshness to assess strength.

Do not output this reasoning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3 — SCORING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Call score_and_recommend(
    brand_description,
    candidate_a,
    candidate_a_evidence,
    candidate_b,
    candidate_b_evidence
)

Use the tool's result to determine the winner.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON:

{
  "trend": "A compelling, publish-ready headline",
  "trend_description": "A concise explanation of why this trend is strategically strong for the brand and what unique angle the brand should take"
}

No extra text. No explanations. No reasoning. JSON only.
"""

# ─────────────────────────────────────────────────────────────
# AGENT SETUP
# ─────────────────────────────────────────────────────────────

def create_research_agent():
    llm = load_llm()
    tools = [
        get_trending_now,               # Phase 1 — Discovery       (call 1, always)
        tavily_search,                  # Phase 1 — Discovery       (call 2, fallback only)
        get_google_trends_trajectory,   # Phase 2 — Validation      (trajectory check)
        get_news_depth,                 # Phase 2 — Validation      (editorial depth)
        score_and_recommend,            # Phase 3 — Scoring         (final call)
    ]
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


def run_research(brand_description: str) -> AgentState:
    print("[TOOL] Researching Appropriate Trend for Brand")

    agent = create_research_agent()

    state: AgentState = {
        "brand_description": brand_description,
        "trend": "",
        "trend_description": "",
        "draft": "",
        "rewrites": 0,
        "critic_review": "",
        "messages": [],
    }

    user_message = f"""Research current trends and recommend the best content topic for this brand:

{brand_description}

Follow the three-phase process: Discovery → Validation → Scoring."""

    # Add message to state
    state["messages"] = [{"role": "user", "content": user_message}]

    # Single invoke (no streaming)
    result = agent.invoke(state)

    final_output = result["messages"][-1].content

    try:
        parsed = json.loads(final_output)

        # Strict validation
        if not isinstance(parsed, dict):
            raise ValueError("Output is not a JSON object.")

        if "trend" not in parsed or "trend_description" not in parsed:
            raise ValueError("Missing required JSON fields.")

        state["trend"] = parsed["trend"]
        state["trend_description"] = parsed["trend_description"]

    except Exception as e:
        raise RuntimeError(
            f"Trend agent returned invalid JSON output.\n\nRaw output:\n{final_output}"
        ) from e
    
    print(f"Trend Picked! Trend is {state["trend"]}")

    return state


