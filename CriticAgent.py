from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from AgentStateClass import AgentState
from model import load_llm
import json

def run_critic_agent(state: AgentState) -> AgentState:
    SYSTEM_PROMPT = """You are a LinkedIn content critic. You will be provided with a post intended for a brand's official page. 
Your task is to evaluate the post and assign a score from 1 to 10 (1 = poor, 10 = excellent) for each of the following dimensions:

1. Brand Relevance – Does the post align with the brand's values, voice, and target audience?
2. Clarity & Engagement – Is the message clear, compelling, and likely to drive interaction (likes, comments, shares)?
3. Originality – Is the content unique, or does it feel like generic or overused corporate fluff?

After scoring, provide concise, actionable suggestions for improvement for each dimension.

Respond in the following JSON format:

{
  "brand_relevance": <score>,
  "clarity_engagement": <score>,
  "originality": <score>,
  "suggestions": {
    "brand_relevance": "<text>",
    "clarity_engagement": "<text>",
    "originality": "<text>"
  },
  "approval" : <bool>
}
"""
    print("[TOOL] Criticizing draft...")

    llm = load_llm()

    user_prompt = f"""
Draft written: {state.get('draft')}
Brand's Description: {state.get('brand_description')}
Trend: {state.get('trend')}
Trend's Description: {state.get('trend_description')}
Previous Criticism: {state.get('critic_review', 'None')}

Based on all this information, critic the draft based on the system instructions provided.
"""

    critic_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state.get('messages', []),
        HumanMessage(content=user_prompt)
    ]

    raw_response = llm.invoke(critic_messages)

    parsed_response = {}
    try:
        parsed_response = json.loads(raw_response.content)

        # Ensure parsed_response is a dict
        if not isinstance(parsed_response, dict):
            raise ValueError("Output is not a JSON object.")

        # Validate expected keys
        for key in ["brand_relevance", "clarity_engagement", "originality", "suggestions", "approval"]:
            if key not in parsed_response:
                raise ValueError(f"Missing key in LLM output: {key}")

        # Validate scores
        for score_key in ["brand_relevance", "clarity_engagement", "originality"]:
            val = parsed_response[score_key]
            if not isinstance(val, (int, float)) or not (1 <= val <= 10):
                raise ValueError(f"Invalid score for {score_key}: {val}")

        # Validate suggestions
        suggestions = parsed_response["suggestions"]
        if not isinstance(suggestions, dict):
            raise ValueError("Suggestions must be a dictionary.")

    except Exception as e:
        print(f"[WARNING] Critic agent returned invalid JSON. Using fallback values.\nError: {e}\nRaw Response: {raw_response.content}")
        parsed_response = {
            "brand_relevance": 5,
            "clarity_engagement": 5,
            "originality": 5,
            "suggestions": {
                "brand_relevance": "Unable to parse suggestions.",
                "clarity_engagement": "Unable to parse suggestions.",
                "originality": "Unable to parse suggestions."
            },
            "approval": False
        }

    # Update state
    state['approval'] = parsed_response.get("approval", False)
    state['critic_review'] = parsed_response

    state["messages"] = [
        *state.get("messages", []),
        HumanMessage(content=user_prompt),
        AIMessage(content=raw_response.content)
    ]

    state["rewrites"] = state.get("rewrites", 0) + 1

    # Print rejection reasons if not approved
    if not state["approval"]:
        print("Draft Rejected! Reasons:")
        for dim, text in parsed_response.get("suggestions", {}).items():
            print(f"- {dim}: {text}")

    return state
