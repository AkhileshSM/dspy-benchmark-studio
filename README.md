# DSPy Prompt Optimizer Benchmark Studio

## What this repo is (read this first)

This is **not** a chatbot and **not** a generic “write a task, get an answer” agent.

It is a **side-by-side experiment**:

1. You pick a small Q&amp;A dataset (reading comprehension, numbers, or people).
2. A **hand-crafted prompt** — a template a person wrote — answers the held-out test questions.
3. **DSPy** looks at the *train* questions, compiles a new prompt (instructions + few-shot demos), and answers the *same* test questions.
4. The dashboard shows accuracy and every test answer: expected vs hand-crafted vs DSPy.

Same model both times. The only thing that changes is the prompt.

```
examples ──► split 60/40
                │
                ├─ train ──► DSPy optimizer ──► compiled prompt ──► test answers
                │
                └─ test  ──► hand-crafted prompt ──────────────────► test answers
                              (the baseline)
```

**“Task description”** is the one-sentence instruction both prompts start from
(e.g. “Answer a factual question grounded in a short reading passage.”).
The hand-crafted run uses it as-is. DSPy may rewrite it.

**Scores are only on the last 40% of the list** (the test split). The first 60%
exist so DSPy has examples to learn from — they are not part of the accuracy bar.

Open the UI at http://localhost:5173 after `docker compose up` and walk the
three numbered sections: choose a task → read the hand-crafted prompt → run.

---

The Q&A model can be **Ollama** (local or cloud) or **any OpenAI-compatible** endpoint
(OpenAI, OpenRouter, Groq, vLLM, LM Studio, Together, Fireworks, DeepSeek, xAI, …).

## Architecture

| Layer | Reasoner | Role |
|-------|----------|------|
| 0 (entry) | `benchmark_orchestrator` | Accepts task + dataset, splits train/eval, fans out 3 strategies |
| 1 (composers) | `naive_prompting` | Hand-crafted prompt, no DSPy |
| 1 (composers) | `dspy_predict_optimized` | DSPy Predict + BootstrapFewShot |
| 1 (composers) | `dspy_cot_optimized` | DSPy ChainOfThought + MIPROv2 |
| 1 (composers) | `enhanced_cot_runner` | Retry round with elevated compilation settings |
| 2 (meta) | `gap_analyzer` | Reads scores, decides if retry needed |
| 2 (leaf) | `single_example_runner` | One example through one strategy |
| 2 (leaf) | `confidence_scorer` | Substring check + LLM-as-judge fallback |

**Dynamic routing:** `gap_analyzer` triggers `enhanced_cot_runner` if any strategy scores < 0.3 or
naive beats both DSPy strategies by > 5%.

## Provider selection

Set `LLM_PROVIDER` in `.env`, or prefix the per-request `model` field.

| `LLM_PROVIDER` | What it talks to | Key env |
|----------------|------------------|---------|
| `auto` (default) | Infer from model prefix, then API keys, else Ollama | — |
| `ollama` | Ollama `/v1/chat/completions`, fallback `/api/chat` | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` |
| `openai` | Any OpenAI-compatible `/v1/chat/completions` | `OPENAI_BASE_URL`, `OPENAI_API_KEY` |

Model prefixes always win over auto-detection:

```text
ollama/llama3.2
openai/gpt-4o-mini
openrouter/google/gemini-2.5-flash
groq/llama-3.1-8b-instant
```

## Run

```bash
cd dspy-benchmark-studio
cp .env.example .env
# Ollama (default): set OLLAMA_BASE_URL + OLLAMA_MODEL
# OpenAI-compatible: set LLM_PROVIDER=openai, OPENAI_API_KEY, OPENAI_MODEL
#                    (optional OPENAI_BASE_URL for OpenRouter / vLLM / LM Studio)

docker compose up --build
```

### Ollama example `.env`

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.2
AI_MODEL=ollama/llama3.2
```

### OpenAI example `.env`

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
AI_MODEL=openai/gpt-4o-mini
```

### OpenRouter / vLLM / LM Studio

```bash
LLM_PROVIDER=openai
OPENAI_BASE_URL=https://openrouter.ai/api/v1   # or http://host.docker.internal:8000/v1
OPENAI_API_KEY=sk-or-v1-...                    # or any bearer the server expects
OPENAI_MODEL=google/gemini-2.5-flash
AI_MODEL=openrouter/google/gemini-2.5-flash
```

## Smoke test

```bash
# Verify agent registered
curl http://localhost:8080/api/v1/discovery/capabilities | jq '.capabilities[] | select(.agent_id=="dspy-benchmark-studio")'

# Fire the canonical async curl with the 5-question reading-comprehension benchmark
EXEC_ID=$(curl -sS -X POST http://localhost:8080/api/v1/execute/async/dspy-benchmark-studio.benchmark_orchestrator \
  -H 'Content-Type: application/json' \
  -d @sample_payload.json | jq -r '.execution_id')

# Poll until succeeded
while :; do
  R=$(curl -sS http://localhost:8080/api/v1/executions/$EXEC_ID)
  S=$(echo "$R" | jq -r '.status')
  case "$S" in
    succeeded) echo "$R" | jq '.result'; break ;;
    failed)    echo "$R" | jq '.'; break ;;
    *)         sleep 2 ;;
  esac
done
```

Use `sample_payload_openai.json` instead when the default provider is OpenAI-compatible.

Inspect the resolved provider without running a full benchmark:

```bash
curl -sS -X POST http://localhost:8080/api/v1/execute/dspy-benchmark-studio.llm_status \
  -H 'Content-Type: application/json' \
  -d '{"input": {}}' | jq
```

## Frontend

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

Or open the compose-built dashboard at http://localhost:5173 after `docker compose up`.

The dashboard lets you pick **Auto / Ollama / OpenAI-compatible**, override the model
per run, and shows the 3-strategy bar chart, live execution timeline, before/after
prompt diff, and per-example results.

## Stop

```bash
docker compose down
```
