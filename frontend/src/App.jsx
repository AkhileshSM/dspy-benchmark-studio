import React, { useState, useEffect, useRef } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
  LineChart, Line, CartesianGrid,
} from 'recharts'

const DEFAULT_EXAMPLES = [
  {
    passage: 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.',
    question: 'Who was the Eiffel Tower named after?',
    expected_answer: 'Gustave Eiffel',
  },
  {
    passage: 'Photosynthesis is the process used by plants, algae and certain bacteria to harness energy from sunlight. During photosynthesis, light energy converts water, carbon dioxide, and minerals into oxygen and energy-rich organic compounds.',
    question: 'What gas do plants release during photosynthesis?',
    expected_answer: 'Oxygen',
  },
  {
    passage: 'The Amazon River in South America is the largest river by discharge volume of water in the world. Its drainage basin covers about 7 million square kilometers.',
    question: 'Which is the largest river by discharge volume in the world?',
    expected_answer: 'The Amazon River',
  },
  {
    passage: 'Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person to win twice, and the only person to win in two different fields — Physics (1903) and Chemistry (1911).',
    question: 'In which two scientific fields did Marie Curie win Nobel Prizes?',
    expected_answer: 'Physics and Chemistry',
  },
  {
    passage: 'The Hubble Space Telescope was launched into low Earth orbit in 1990. The telescope was named after astronomer Edwin Hubble, whose observations in the 1920s confirmed that the universe is expanding.',
    question: 'In what year was the Hubble Space Telescope launched?',
    expected_answer: '1990',
  },
]

const CP = 'http://localhost:8080'

export default function App() {
  const [model, setModel] = useState('qwen3.5:4b-mlx')
  const [taskDescription, setTaskDescription] = useState(
    'Answer a factual question grounded in a short reading passage.'
  )
  const [examples, setExamples] = useState(DEFAULT_EXAMPLES)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const [report, setReport] = useState(null)
  const [timeline, setTimeline] = useState([])
  const [execStatus, setExecStatus] = useState('idle')
  const timelineRef = useRef(null)

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
        model: model || null,
        train_ratio: 0.6,
      },
    }

    try {
      // Fire async execution
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

      // Poll for completion
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

      // Extract result (control plane wraps it in {result: ...} or returns it directly)
      const reportData = result.result?.result || result.result || result
      setReport(reportData)
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

  const strategies = report?.strategies || []
  const chartData = strategies.map(s => ({
    name: s.strategy_name,
    accuracy: +(s.accuracy * 100).toFixed(1),
    avg_latency_ms: +s.avg_latency_ms.toFixed(0),
    total_tokens: s.total_tokens,
  }))

  const bestStrategy = strategies.length ? strategies.reduce((a, b) => a.accuracy > b.accuracy ? a : b) : null
  const naiveStrategy = strategies.find(s => s.strategy_name === 'naive')

  return (
    <div className="app">
      <h1>
        DSPy <span className="accent">Benchmark</span> Studio
      </h1>
      <p className="subtitle">
        Side-by-side comparison of hand-crafted prompts vs DSPy-optimized programs
        on the same Cloud Ollama model. The AgentField DAG below shows exactly
        how the optimization pipeline executed.
      </p>

      <div className="controls">
        <label>
          Ollama model
          <input
            value={model}
            onChange={e => setModel(e.target.value)}
            placeholder="llama3.2"
            disabled={running}
          />
        </label>
        <label style={{ flex: 1, minWidth: 280 }}>
          Task description
          <input
            value={taskDescription}
            onChange={e => setTaskDescription(e.target.value)}
            disabled={running}
          />
        </label>
        <label>
          Examples: {examples.length}
        </label>
        <button onClick={runBenchmark} disabled={running}>
          {running ? 'Running...' : 'Run Benchmark'}
        </button>
      </div>

      {error && <div className="error">⚠ {error}</div>}

      <div className="kpi-row">
        <div className="kpi">
          <div className="label">
            <span className={`status-dot ${execStatus}`}></span>
            Status
          </div>
          <div className="value">{execStatus}</div>
        </div>
        <div className="kpi">
          <div className="label">Winner</div>
          <div className="value">{bestStrategy ? bestStrategy.strategy_name : '—'}</div>
        </div>
        <div className="kpi">
          <div className="label">Best accuracy</div>
          <div className="value good">
            {bestStrategy ? `${(bestStrategy.accuracy * 100).toFixed(0)}%` : '—'}
          </div>
        </div>
        <div className="kpi">
          <div className="label">Improvement vs naive</div>
          <div className={`value ${report && report.improvement_pct > 0 ? 'good' : 'warn'}`}>
            {report ? `${report.improvement_pct > 0 ? '+' : ''}${report.improvement_pct}%` : '—'}
          </div>
        </div>
      </div>

      <div className="grid-3">
        <div className="panel">
          <h2>Strategy comparison</h2>
          {chartData.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={chartData}>
                <CartesianGrid stroke="#2a2f3a" strokeDasharray="3 3" />
                <XAxis dataKey="name" stroke="#8b94a7" fontSize={11} />
                <YAxis stroke="#8b94a7" fontSize={11} />
                <Tooltip contentStyle={{ background: '#1a1d27', border: '1px solid #2a2f3a' }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="accuracy" fill="#7c5cff" name="Accuracy %" />
                <Bar dataKey="avg_latency_ms" fill="#00d4aa" name="Latency (ms)" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ color: 'var(--muted)', padding: 40, textAlign: 'center' }}>
              Run the benchmark to see results
            </div>
          )}
        </div>

        <div className="panel">
          <h2>Execution timeline</h2>
          <div className="timeline" ref={timelineRef} style={{ maxHeight: 260, overflowY: 'auto' }}>
            {timeline.length === 0 ? (
              <div style={{ color: 'var(--muted)', padding: 20, textAlign: 'center' }}>
                Awaiting dispatch
              </div>
            ) : (
              timeline.map((t, i) => (
                <div key={i} className={`step ${t.error ? 'error' : 'meta'}`}>
                  <span className="ts">{new Date(t.ts).toLocaleTimeString()}</span> {t.msg}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {report && (
        <>
          <div className="grid-2">
            <div className="panel">
              <h2>Token usage by strategy</h2>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={chartData}>
                  <CartesianGrid stroke="#2a2f3a" strokeDasharray="3 3" />
                  <XAxis dataKey="name" stroke="#8b94a7" fontSize={11} />
                  <YAxis stroke="#8b94a7" fontSize={11} />
                  <Tooltip contentStyle={{ background: '#1a1d27', border: '1px solid #2a2f3a' }} />
                  <Bar dataKey="total_tokens" fill="#ff922b" name="Total tokens" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="panel">
              <h2>Per-example results</h2>
              <div style={{ maxHeight: 260, overflowY: 'auto' }}>
                <table>
                  <thead>
                    <tr>
                      <th>Strategy</th>
                      <th>Q#</th>
                      <th>Predicted</th>
                      <th>Expected</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {strategies.flatMap(s =>
                      (s.per_example || []).map(r => (
                        <tr key={`${s.strategy_name}-${r.example_index}`}>
                          <td><span className={`badge ${s.strategy_name === 'naive' ? 'naive' : s.strategy_name === 'dspy_predict' ? 'predict' : 'cot'}`}>
                            {s.strategy_name.replace('dspy_', '')}
                          </span></td>
                          <td>{r.example_index + 1}</td>
                          <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {r.predicted_answer}
                          </td>
                          <td>{r.expected_answer}</td>
                          <td className={r.correct ? 'correct' : 'incorrect'}>
                            {r.correct ? '✓' : '✗'}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="panel" style={{ marginBottom: 20 }}>
            <h2>Before / After — what the optimizer actually changed</h2>
            {strategies
              .filter(s => s.strategy_name !== 'naive')
              .map(s => (
                <div key={s.strategy_name} style={{ marginBottom: 20 }}>
                  <h3 style={{ fontSize: 13, color: 'var(--muted)', margin: '0 0 8px 0' }}>
                    <span className={`badge ${s.strategy_name === 'dspy_predict' ? 'predict' : 'cot'}`}>
                      {s.strategy_name}
                    </span>
                    <span style={{ marginLeft: 8 }}>optimizer: {s.optimizer}</span>
                    <span style={{ marginLeft: 8 }}>accuracy: {(s.accuracy * 100).toFixed(0)}%</span>
                  </h3>
                  <div className="prompt-diff">
                    <div className="prompt-col">
                      <h3>RAW (hand-crafted)</h3>
                      <div className="prompt-text raw">{s.raw_prompt}</div>
                    </div>
                    <div className="prompt-col">
                      <h3>DSPy-COMPILED</h3>
                      <div className="prompt-text optimized">{s.compiled_prompt}</div>
                    </div>
                  </div>
                </div>
              ))}
          </div>

          {report.enhanced_round && (
            <div className="panel" style={{ marginBottom: 20 }}>
              <h2>
                <span className="badge enhanced">enhanced</span>
                Retry round — gap_analyzer detected {report.gap_analysis.weakest_strategy} underperformed
              </h2>
              <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 12 }}>
                {report.gap_analysis.reasoning}
              </div>
              <div className="kpi-row" style={{ marginBottom: 0 }}>
                <div className="kpi">
                  <div className="label">Enhanced accuracy</div>
                  <div className="value good">
                    {(report.enhanced_round.accuracy * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="kpi">
                  <div className="label">Avg latency</div>
                  <div className="value">{report.enhanced_round.avg_latency_ms.toFixed(0)}ms</div>
                </div>
                <div className="kpi">
                  <div className="label">Total tokens</div>
                  <div className="value">{report.enhanced_round.total_tokens}</div>
                </div>
                <div className="kpi">
                  <div className="label">Optimizer</div>
                  <div className="value" style={{ fontSize: 16 }}>
                    {report.gap_analysis.recommended_optimizer}
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
