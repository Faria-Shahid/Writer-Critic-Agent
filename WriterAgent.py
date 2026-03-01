from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from AgentStateClass import AgentState
from model import load_llm
import json
from langchain_core.messages import trim_messages

def run_writer_agent(state: AgentState) -> AgentState:
    """Generates (or rewrites) a LinkedIn post draft based on trend + brand context.
    Injects its own system prompt so the main message history stays clean.
    Returns a dict for consistency with other agent nodes."""

    SYSTEM_PROMPT = """
    You are a senior content strategist building thought-leadership posts for LinkedIn.

    You will be given:
    1. A trending topic
    2. Trend's description and angle
    3. A brand description

     Write a LinkedIn post that:
    - Opens with a strong hook
    - Uses short paragraphs
    - Playful yet professional tone
    - Natural connection to the trend
    - Ends with a question
    - Max 2 emojis
    
    OUTPUT FORMAT:
    - Return ONLY JSON.
    - DO NOT use Markdown, code fences, or extra text.
    - Format exactly like this:
    {
     "draft" : "<your post here>"
    }
    """

    if state["rewrites"] > 0:
        print("[TOOL] Rewriting Post According to Critique")
        user_prompt = f"""
        Trend: {state["trend"]}
        Brand: {state["brand_description"]}

        Previous Draft:
        {state["draft"]}

        Critic Feedback:
        {state["critic_review"]}

        Rewrite the post improving based on feedback.
        """
    else:
        print("[TOOL] Writing Post")
        user_prompt = f"""
        Trend: {state["trend"]}
        Context:
        {state["trend_description"]}

        Brand:
        {state["brand_description"]}

        Write the post.
        """

    llm = load_llm()

    writer_messages = [
        SystemMessage(content = SYSTEM_PROMPT),
        *state["messages"],
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(writer_messages)

    try:
        parsed = json.loads(response.content)

        # Strict validation
        if not isinstance(parsed, dict):
            raise ValueError("Output is not a JSON object.")

        state["draft"] = parsed["draft"]

    except Exception as e:
        raise RuntimeError(f"Writer agent returned invalid JSON: {response}") from e

    state["messages"] = [
        *state["messages"],
        HumanMessage(content=user_prompt),
        AIMessage(content=response.content),
    ]

    return state




