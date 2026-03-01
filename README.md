# 📝 Research & Writing Agent

A multi-agent pipeline built with LangGraph that researches a topic, drafts content, and iteratively refines it through a critic feedback loop.

## 🔄 Architecture

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__([<p>__start__</p>]):::first
        researcher(researcher)
        writer(writer)
        critic(critic)
        __end__([<p>__end__</p>]):::last
        __start__ --> researcher;
        researcher --> writer;
        writer --> critic;
        %% loop back from critic to writer
        critic --> writer;
        critic --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```

## 🤖 Agents

| Agent | Role |
|-------|------|
| **Researcher** | Gathers and synthesizes information on the given topic |
| **Writer** | Drafts content based on the researcher's output |
| **Critic** | Reviews the draft and either approves it or sends it back for revision |

## 🚀 Getting Started

```bash
pip install langgraph langchain
python main.py
```

## 🛠️ Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — LLM tooling
