"""dspy-benchmark-studio — AgentField agent node.

Multi-reasoner system that compares naive prompting against DSPy-optimized
strategies on a Q&A benchmark. See reasoners/ for the architecture.
"""

import os

from agentfield import Agent, AIConfig

from reasoners import meta_router, strategies_router

app = Agent(
    node_id=os.getenv("AGENT_NODE_ID", "dspy-benchmark-studio"),
    agentfield_server=os.getenv("AGENTFIELD_SERVER", "http://localhost:8080"),
    version="1.0.0",
    ai_config=AIConfig(
        # LiteLLM-style model string. Examples:
        #   ollama/llama3.2
        #   openai/gpt-4o-mini
        #   openrouter/google/gemini-2.5-flash
        # Q&A + DSPy compilation go through reasoners.llm (Ollama native or
        # any OpenAI-compatible /v1/chat/completions endpoint).
        model=os.getenv("AI_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("OLLAMA_MODEL")
        or "ollama/llama3.2",
    ),
    dev_mode=True,
)

# Two routers, namespaced by tags but no prefix (so reasoner IDs are flat).
app.include_router(strategies_router)
app.include_router(meta_router)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8001")), auto_port=False)
