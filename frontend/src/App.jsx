import React, { useState, useEffect, useRef } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts'
import {
  TASKS, TRAIN_RATIO, handCraftedPrompt, splitExamples,
} from './tasks.js'

const CP = 'http://localhost:8080'

const PROVIDER_PREFIXES = [
  'ollama/', 'openai/', 'openrouter/', 'groq/', 'together/',
  'fireworks/', 'deepseek/', 'xai/', 'mistral/',
]

const STRATEGY_META = {
  naive: {
    key: 'naive',
    short: 'Hand-crafted',
    badge: 'naive',
    title: 'Hand-crafted prompt',
    body: 'A prompt a person wrote. Same wording for every test question. No training, no demos, no optimizer.',
  },
  dspy_predict: {
    key: 'dspy_predict',
    short: 'DSPy Predict',
    badge: 'predict',
    title: 'DSPy Predict',
    body: 'DSPy reads the train questions, then compiles a new prompt (instructions + few-shot demos) with BootstrapFewShot.',
  },
  dspy_cot: {
    key: 'dspy_cot',
    short: 'DSPy CoT',
    badge: 'cot',
    title: 'DSPy Chain-of-Thought',
    body: 'Same idea, plus “think step by step”. MIPROv2 searches for a better instruction and demo set.',
  },
}

function encodeModel(provider, model) {
  const m = (model || '').trim()
  if (PROVIDER_PREFIXES.some(p => m.toLowerCase().startsWith(p))) return m
  if (provider === 'ollama') return m ? `ollama/${m}` : 'ollama/'
  if (provider === 'openai') return m ? `openai/${m}` : 'openai/'
  return m || null
}

function pct(n) {
  return `${(n * 100).toFixed(0)}%`
}

function strategyOf(strategies, name) {
  return strategies.find(s => s.strategy_name === name) || null
}

function evalExamplesFrom(examples) {
  return splitExamples(examples, TRAIN_RATIO).evalSet
}

export default function App() {
  const [taskId, setTaskId] = useState(TASKS[0].id)
  const [taskDescription, setTaskDescription] = useState(TASKS[0].task_description)
  const [examples, setExamples] = useState(TASKS[0].examples)
  const [provider, setProvider] = useState('auto')
  const [model, setModel] = useState('')
  const [availableModels, setAvailableModels] = useState([])
  const [resolved, setResolved] = useState(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const [report, setReport] = useState(null)
  const [timeline, setTimeline] = useState([])
  const [execStatus, setExecStatus] = useState('idle')
  const [openExample, setOpenExample] = useState(0)
  const timelineRef = useRef(null)
  const resultsRef = useRef(null)

  const selectedTask = TASKS.find(t => t.id === taskId) || TASKS[0]
  const split = splitExamples(examples, TRAIN_RATIO)
  const promptPreview = handCraftedPrompt(taskDescription)

  function loadTask(task) {
    setTaskId(task.id)
    setTaskDescription(task.task_description)
    setExamples(task.examples.map(e => ({ ...e })))
    setOpenExample(0)
    setReport(null)
    setError(null)
    setExecStatus('idle')
    setTimeline([])
  }

  function updateExample(index, field, value) {
    setExamples(prev => prev.map((ex, i) => (i === index ? { ...ex, [field]: value } : ex)))
  }

  function addExample() {
    setExamples(prev => [
      ...prev,
      { passage: '', question: '', expected_answer: '' },
    ])
    setOpenExample(examples.length)
  }

  function removeExample(index) {
    if (examples.length <= 2) return
    setExamples(prev => prev.filter((_, i) => i !== index))
    setOpenExample(0)
  }

  async function runBenchmark() {
    setRunning(true)
    setError(null)
    setReport(null)
    setTimeline([])
    setExecStatus('running')

    const payload = {
      input: {
        task_description: taskDescription,
        examples,
        model: encodeModel(provider, model),
        train_ratio: TRAIN_RATIO,
      },
    }

    try {
      const fire = await fetch(
        `${CP}/api/v1/execute/async/dspy-benchmark-studio.benchmark_orchestrator`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        }
      )
      if (!fire.ok) {
        const text = await fire.text()
        throw new Error(`Async dispatch failed: ${fire.status} ${text}`)
      }
      const { execution_id } = await fire.json()
      setTimeline(t => [...t, { ts: new Date().toISOString(), msg: `Dispatched execution ${execution_id}` }])

      let result = null
      for (let i = 0; i < 120; i++) {
        await new Promise(r => setTimeout(r, 2000))
        const poll = await fetch(`${CP}/api/v1/executions/${execution_id}`)
        if (!poll.ok) continue
        const data = await poll.json()
        if (data.status === 'succeeded') {
          result = data
          setExecStatus('succeeded')
          setTimeline(t => [...t, { ts: new Date().toISOString(), msg: 'Execution succeeded' }])
          break
        }
        if (data.status === 'failed') {
          setExecStatus('failed')
          setTimeline(t => [...t, { ts: new Date().toISOString(), msg: 'Execution FAILED', error: true }])
          throw new Error(data.error || 'Execution failed')
        }
        if (i % 5 === 0) {
          setTimeline(t => [...t, { ts: new Date().toISOString(), msg: `Polling ${i * 2}s elapsed...` }])
        }
      }

      if (!result) throw new Error('Execution timed out after 240s')

      const reportData = result.result?.result || result.result || result
      setReport(reportData)
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 50)
    } catch (e) {
      setError(e.message)
      setExecStatus('failed')
    } finally {
      setRunning(false)
    }
  }

  useEffect(() => {
    if (timelineRef.current) {
      timelineRef.current.scrollTop = timelineRef.current.scrollHeight
    }
  }, [timeline])

  useEffect(() => {
    let cancelled = false
    async function loadStatus() {
      try {
        const res = await fetch(`${CP}/api/v1/execute/dspy-benchmark-studio.llm_status`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ input: {} }),
        })
        if (!res.ok || cancelled) return
        const data = await res.json()
        const status = data.result || data
        if (cancelled || !status) return
        setResolved(status)
        if (Array.isArray(status.available_models)) {
          setAvailableModels(status.available_models)
        }
      } catch {
        // Control plane / agent may not be up yet.
      }
    }
    loadStatus()
    return () => { cancelled = true }
  }, [])

  const strategies = report?.strategies || []
  const naive = strategyOf(strategies, 'naive')
  const predict = strategyOf(strategies, 'dspy_predict')
  const cot = strategyOf(strategies, 'dspy_cot')
  const bestStrategy = strategies.length
    ? strategies.reduce((a, b) => (a.accuracy > b.accuracy ? a : b))
    : null
  const evalSet = evalExamplesFrom(examples)
  const chartData = strategies.map(s => ({
    name: STRATEGY_META[s.strategy_name]?.short || s.strategy_name,
    accuracy: +(s.accuracy * 100).toFixed(1),
    avg_latency_ms: +s.avg_latency_ms.toFixed(0),
    total_tokens: s.total_tokens,
  }))

  return (
    <div className="app">
      <header className="hero">
        <p className="eyebrow">What this repo does</p>
        <h1>
          Same questions. Same model. Two prompts.
        </h1>
        <p className="lede">
          Pick a small Q&amp;A task. We run it first with a <strong>hand-crafted prompt</strong>
          (the wording a person wrote), then with a <strong>DSPy-optimized prompt</strong>
          (compiled from the train split). The scoreboard is only on held-out test
          questions — so you can see whether optimization actually helped.
        </p>
        <ol className="story-steps">
          <li><span>1</span> Choose a task and look at the questions</li>
          <li><span>2</span> Read the hand-crafted prompt</li>
          <li><span>3</span> Run — then compare answers side by side</li>
        </ol>
      </header>

      <section className="how">
        {Object.values(STRATEGY_META).map(s => (
          <article key={s.key} className={`how-card ${s.badge}`}>
            <span className={`badge ${s.badge}`}>{s.short}</span>
            <h2>{s.title}</h2>
            <p>{s.body}</p>
          </article>
        ))}
      </section>

      <section className="panel">
        <h2>1. Choose a task</h2>
        <p className="panel-lead">
          A “task” is a labeled dataset plus one instruction sentence. The instruction
          is what both prompts start from. DSPy may rewrite it; the hand-crafted run
          uses it as-is.
        </p>
        <div className="task-grid">
          {TASKS.map(task => (
            <button
              key={task.id}
              type="button"
              className={`task-card ${task.id === taskId ? 'selected' : ''}`}
              onClick={() => loadTask(task)}
              disabled={running}
            >
              <h3>{task.title}</h3>
              <p>{task.blurb}</p>
              <span className="task-meta">{task.examples.length} questions</span>
            </button>
          ))}
        </div>

        <p className="why">{selectedTask.why}</p>

        <label className="field">
          Instruction (this is the “task description”)
          <textarea
            value={taskDescription}
            onChange={e => setTaskDescription(e.target.value)}
            disabled={running}
            rows={2}
          />
        </label>

        <div className="split-note">
          First <strong>{split.trainCount}</strong> questions train DSPy.
          Last <strong>{split.evalCount}</strong> are the hidden test — only those are scored.
        </div>

        <div className="example-list">
          {examples.map((ex, i) => {
            const role = split.cut < examples.length && i < split.cut ? 'train' : 'test'
            const open = openExample === i
            return (
              <div key={i} className={`example ${role} ${open ? 'open' : ''}`}>
                <button
                  type="button"
                  className="example-head"
                  onClick={() => setOpenExample(open ? -1 : i)}
                >
                  <span className={`badge ${role}`}>{role}</span>
                  <span className="example-q">Q{i + 1}. {ex.question || '(empty question)'}</span>
                  <span className="example-a">{ex.expected_answer}</span>
                </button>
                {open && (
                  <div className="example-body">
                    <label>
                      Passage
                      <textarea
                        value={ex.passage}
                        onChange={e => updateExample(i, 'passage', e.target.value)}
                        disabled={running}
                        rows={3}
                      />
                    </label>
                    <label>
                      Question
                      <input
                        value={ex.question}
                        onChange={e => updateExample(i, 'question', e.target.value)}
                        disabled={running}
                      />
                    </label>
                    <label>
                      Expected answer
                      <input
                        value={ex.expected_answer}
                        onChange={e => updateExample(i, 'expected_answer', e.target.value)}
                        disabled={running}
                      />
                    </label>
                    <button
                      type="button"
                      className="linkish"
                      onClick={() => removeExample(i)}
                      disabled={running || examples.length <= 2}
                    >
                      Remove question
                    </button>
                  </div>
                )}
              </div>
            )
          })}
        </div>
        <button type="button" className="ghost" onClick={addExample} disabled={running}>
          + Add a question
        </button>
      </section>

      <section className="panel">
        <h2>2. Hand-crafted prompt (used as-is)</h2>
        <p className="panel-lead">
          Every test question is stuffed into this template. DSPy never sees this
          wording on the naive run — that is the baseline.
        </p>
        <div className="prompt-text raw">{promptPreview}</div>
      </section>

      <section className="panel run-panel">
        <h2>3. Run the comparison</h2>
        <p className="panel-lead">
          One click fires three strategies in parallel on the same model: hand-crafted,
          DSPy Predict, DSPy Chain-of-Thought.
        </p>
        <div className="controls">
          <label>
            Provider
            <select
              value={provider}
              onChange={e => setProvider(e.target.value)}
              disabled={running}
            >
              <option value="auto">Auto (from env / prefix)</option>
              <option value="ollama">Ollama</option>
              <option value="openai">OpenAI-compatible</option>
            </select>
          </label>
          <label>
            Model
            <input
              value={model}
              onChange={e => setModel(e.target.value)}
              placeholder={
                provider === 'openai'
                  ? 'gpt-4o-mini or openai/gpt-4o-mini'
                  : provider === 'ollama'
                    ? 'llama3.2 or ollama/llama3.2'
                    : 'leave blank to use AI_MODEL from .env'
              }
              list="available-models"
              disabled={running}
            />
            <datalist id="available-models">
              {availableModels.map(m => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </label>
          <button onClick={runBenchmark} disabled={running || examples.length < 2}>
            {running ? 'Running…' : 'Run hand-crafted vs DSPy'}
          </button>
        </div>
        {resolved && (
          <p className="hint">
            Default model: {resolved.provider} · {resolved.display_model || resolved.model}
            {resolved.base_url ? ` · ${resolved.base_url}` : ''}
          </p>
        )}
        {error && <div className="error">⚠ {error}</div>}
        <div className="kpi-row tight">
          <div className="kpi">
            <div className="label">
              <span className={`status-dot ${execStatus}`}></span>
              Status
            </div>
            <div className="value">{execStatus}</div>
          </div>
          <div className="kpi">
            <div className="label">Test questions</div>
            <div className="value">{split.evalCount}</div>
          </div>
          <div className="kpi">
            <div className="label">Train (DSPy only)</div>
            <div className="value">{split.trainCount}</div>
          </div>
        </div>
        <div className="timeline" ref={timelineRef}>
          {timeline.length === 0 ? (
            <div className="muted-center">The run log appears here after you click Run.</div>
          ) : (
            timeline.map((t, i) => (
              <div key={i} className={`step ${t.error ? 'error' : 'meta'}`}>
                <span className="ts">{new Date(t.ts).toLocaleTimeString()}</span> {t.msg}
              </div>
            ))
          )}
        </div>
      </section>

      <div ref={resultsRef}>
        {!report ? (
          <section className="panel results-placeholder">
            <h2>Results will land here</h2>
            <p>
              After a run you will see: (1) accuracy of the hand-crafted prompt,
              (2) accuracy of each DSPy prompt, (3) every test question with both
              answers next to the expected one, (4) the actual prompt DSPy compiled.
            </p>
          </section>
        ) : (
          <>
            <section className="panel">
              <h2>Results — hand-crafted vs DSPy</h2>
              <p className="panel-lead">
                Scored on the {evalSet.length} held-out test question{evalSet.length === 1 ? '' : 's'}
                {report.model ? ` · model ${report.model}` : ''}
                {report.provider ? ` · ${report.provider}` : ''}.
                {bestStrategy ? ` Winner: ${STRATEGY_META[bestStrategy.strategy_name]?.short || bestStrategy.strategy_name}.` : ''}
                {typeof report.improvement_pct === 'number'
                  ? ` DSPy best vs hand-crafted: ${report.improvement_pct > 0 ? '+' : ''}${report.improvement_pct}%.`
                  : ''}
              </p>
              <div className="scoreboard">
                {[naive, predict, cot].map((s, idx) => {
                  const meta = Object.values(STRATEGY_META)[idx]
                  const n = s?.per_example?.length || evalSet.length
                  const correct = s ? s.per_example.filter(r => r.correct).length : null
                  const isBest = s && bestStrategy && s.strategy_name === bestStrategy.strategy_name
                  return (
                    <article
                      key={meta.key}
                      className={`score-card ${meta.badge} ${isBest ? 'winner' : ''} ${!s ? 'empty' : ''}`}
                    >
                      <span className={`badge ${meta.badge}`}>{meta.short}</span>
                      <div className="score-pct">{s ? pct(s.accuracy) : '—'}</div>
                      <div className="score-sub">
                        {s ? `${correct}/${n} test questions correct` : 'did not return'}
                      </div>
                      <div className="score-opt">
                        {s ? `optimizer: ${s.optimizer}` : meta.title}
                      </div>
                    </article>
                  )
                })}
              </div>
            </section>

            <section className="panel">
              <h2>Test questions — both answers</h2>
              <p className="panel-lead">
                Each row is one held-out question. Read left to right: expected,
                then the hand-crafted answer, then each DSPy answer.
              </p>
              <div className="table-wrap">
                <table className="headtohead">
                  <thead>
                    <tr>
                      <th>Question</th>
                      <th>Expected</th>
                      <th>Hand-crafted</th>
                      <th>DSPy Predict</th>
                      <th>DSPy CoT</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evalSet.map((ex, i) => {
                      const row = {
                        naive: naive?.per_example?.[i],
                        predict: predict?.per_example?.[i],
                        cot: cot?.per_example?.[i],
                      }
                      return (
                        <tr key={i}>
                          <td>
                            <div className="qcell">
                              <span className="qidx">Test Q{i + 1}</span>
                              {ex.question}
                            </div>
                          </td>
                          <td>{ex.expected_answer}</td>
                          <AnswerCell result={row.naive} />
                          <AnswerCell result={row.predict} />
                          <AnswerCell result={row.cot} />
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="panel">
              <h2>What DSPy changed in the prompt</h2>
              <p className="panel-lead">
                Left is the hand-crafted template from step 2. Right is the compiled
                prompt the optimizer actually used on the test questions.
              </p>
              {strategies
                .filter(s => s.strategy_name !== 'naive')
                .map(s => (
                  <div key={s.strategy_name} className="prompt-block">
                    <h3>
                      <span className={`badge ${STRATEGY_META[s.strategy_name]?.badge || 'predict'}`}>
                        {STRATEGY_META[s.strategy_name]?.short || s.strategy_name}
                      </span>
                      <span className="prompt-acc">{pct(s.accuracy)} on test</span>
                    </h3>
                    <div className="prompt-diff">
                      <div className="prompt-col">
                        <h3>Hand-crafted</h3>
                        <div className="prompt-text raw">{s.raw_prompt}</div>
                      </div>
                      <div className="prompt-col">
                        <h3>DSPy-compiled</h3>
                        <div className="prompt-text optimized">{s.compiled_prompt}</div>
                      </div>
                    </div>
                  </div>
                ))}
            </section>

            <div className="grid-2">
              <div className="panel">
                <h2>Accuracy vs latency</h2>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={chartData}>
                    <CartesianGrid stroke="#2a2f3a" strokeDasharray="3 3" />
                    <XAxis dataKey="name" stroke="#8b94a7" fontSize={11} />
                    <YAxis stroke="#8b94a7" fontSize={11} />
                    <Tooltip contentStyle={{ background: '#1a1d27', border: '1px solid #2a2f3a' }} />
                    <Bar dataKey="accuracy" fill="#7c5cff" name="Accuracy %" />
                    <Bar dataKey="avg_latency_ms" fill="#00d4aa" name="Latency (ms)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="panel">
                <h2>Token usage</h2>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={chartData}>
                    <CartesianGrid stroke="#2a2f3a" strokeDasharray="3 3" />
                    <XAxis dataKey="name" stroke="#8b94a7" fontSize={11} />
                    <YAxis stroke="#8b94a7" fontSize={11} />
                    <Tooltip contentStyle={{ background: '#1a1d27', border: '1px solid #2a2f3a' }} />
                    <Bar dataKey="total_tokens" fill="#ff922b" name="Total tokens" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {report.enhanced_round && (
              <section className="panel">
                <h2>
                  <span className="badge enhanced">retry</span>
                  {' '}Gap analyzer sent DSPy back for another compile
                </h2>
                <p className="panel-lead">{report.gap_analysis?.reasoning}</p>
                <div className="kpi-row">
                  <div className="kpi">
                    <div className="label">Retry accuracy</div>
                    <div className="value good">{pct(report.enhanced_round.accuracy)}</div>
                  </div>
                  <div className="kpi">
                    <div className="label">Optimizer</div>
                    <div className="value" style={{ fontSize: 16 }}>
                      {report.gap_analysis?.recommended_optimizer}
                    </div>
                  </div>
                </div>
              </section>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function AnswerCell({ result }) {
  if (!result) return <td className="muted">—</td>
  return (
    <td className={result.correct ? 'correct' : 'incorrect'}>
      <span className="mark">{result.correct ? '✓' : '✗'}</span>
      {' '}
      {result.predicted_answer}
    </td>
  )
}
