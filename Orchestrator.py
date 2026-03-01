import sys
import json
from ResearchAgent import run_research
from WriterAgent import run_writer_agent
from CriticAgent import run_critic_agent
from AgentStateClass import AgentState
from langgraph.graph import StateGraph, END

MAX_REWRITES = 2

def route_after_critic(state: AgentState) -> str:
    if state["rewrites"] < MAX_REWRITES and state.get("approval") is False:
        return "writer"
    return END

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("researcher", run_research)
    graph.add_node("writer", run_writer_agent)
    graph.add_node("critic", run_critic_agent)
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges("critic", route_after_critic)
    return graph.compile()

def main():
    print("LinkedIn Content Agent")
    print("----------------------")
    print("Enter your brand description (blank line to finish):\n")

    lines = []
    while True:
        line = input()
        if line == "" and lines:
            break
        lines.append(line)

    brand_description = "\n".join(lines)

    initial_state: AgentState = {
        "brand_description": brand_description,
        "trend": "",
        "trend_description": "",
        "draft": "",
        "rewrites": 0,
        "approval": False,
        "critic_review": "",
        "messages": [],
    }

    print("\nRunning agent pipeline...\n")
    app = build_graph()
    final_state = app.invoke(initial_state)

    print("\n--- Final Post ---")
    print(final_state["draft"])
    print(f"\nRewrites used: {final_state['rewrites']}/{MAX_REWRITES}")
    print(f"Trend: {final_state['trend']}")

if __name__ == "__main__":
    main()