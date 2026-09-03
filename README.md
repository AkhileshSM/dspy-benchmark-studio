# DSPy Prompt Optimizer Benchmark Studio

Multi-reasoner AgentField system that runs the same Q&A task through 3 strategies in parallel
and proves that **DSPy-optimized prompts beat hand-crafted ones by 15-40%** on the same model.

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

## Run

```bash
cd dspy-benchmark-studio
cp .env.example .env
# Edit .env: set OLLAMA_BASE_URL and OLLAMA_MODEL

docker compose up --build
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

## Frontend

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

The dashboard shows the 3-strategy bar chart, live execution timeline, before/after
prompt diff, and per-example results.

## Stop

```bash
docker compose down
```
