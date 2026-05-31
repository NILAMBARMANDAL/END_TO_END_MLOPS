import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

// Real statistics computed from the training dataset (124k rows).
// "fail" / "heal" are the mean values when failure=1 vs failure=0 — i.e. how
// strongly each metric separates failures from healthy devices.
const METRICS = [
  { key: "metric1", min: 0, median: 122797388, max: 244140480, heal: 122384023, fail: 127175527 },
  { key: "metric2", min: 0, median: 0, max: 64968, heal: 156, fail: 4109 },
  { key: "metric3", min: 0, median: 0, max: 24929, heal: 10, fail: 4 },
  { key: "metric4", min: 0, median: 0, max: 1666, heal: 2, fail: 55 },
  { key: "metric5", min: 1, median: 10, max: 98, heal: 14, fail: 16 },
  { key: "metric6", min: 8, median: 249799, max: 689161, heal: 260174, fail: 258304 },
  { key: "metric7", min: 0, median: 0, max: 832, heal: 0.3, fail: 30.6 },
  { key: "metric8", min: 0, median: 0, max: 832, heal: 0.3, fail: 30.6 },
  { key: "metric9", min: 0, median: 0, max: 70000, heal: 13, fail: 23 },
];

// "separation strength" = how much the failure-avg differs from healthy-avg,
// as a rough signal of each metric's importance. Normalized 0..1 for the bar.
const sep = METRICS.map((m) => {
  const denom = Math.max(Math.abs(m.heal), Math.abs(m.fail), 1);
  return Math.min(Math.abs(m.fail - m.heal) / denom, 1);
});
const maxSep = Math.max(...sep);

const SAMPLE = {
  date: "2015-03-17", device: "S1F01085",
  metric1: 215630672, metric2: 55, metric3: 0, metric4: 52, metric5: 6,
  metric6: 407438, metric7: 0, metric8: 0, metric9: 7,
};

const MODEL_INFO = [
  { label: "Algorithm", value: "Random Forest" },
  { label: "F1 Score", value: "0.917" },
  { label: "ROC-AUC", value: "0.952" },
  { label: "Threshold", value: "0.42" },
  { label: "Features", value: "19" },
  { label: "Trees", value: "200" },
];

function fmt(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(0) + "k";
  return String(n);
}

export default function App() {
  const [form, setForm] = useState(SAMPLE);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const update = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  async function predict() {
    setLoading(true); setError(null);
    try {
      const payload = { ...form };
      METRICS.forEach(({ key }) => (payload[key] = Number(payload[key])));
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      const data = await res.json();
      setTimeout(() => { setResult(data); setLoading(false); }, 450);
    } catch (e) {
      setError(e.message); setLoading(false);
    }
  }

  const isFailure = result?.prediction === 1;
  const pct = result?.failure_probability != null
    ? (result.failure_probability * 100) : null;

  return (
    <div className="app">
      <div className="bg-grid" />
      <div className="bg-glow" />

      <div className="container">
        {/* ---------- Header ---------- */}
        <header className="hdr">
          <div className="brand">
            <div className="brand-mark">◇</div>
            <div>
              <div className="brand-name">Sentinel</div>
              <div className="brand-sub">Predictive Maintenance</div>
            </div>
          </div>
          <div className="hdr-status">
            <span className="live-dot" /> model online
          </div>
        </header>

        {/* ---------- Model info strip ---------- */}
        <div className="info-strip">
          {MODEL_INFO.map((m, i) => (
            <div className="info-cell" key={m.label} style={{ animationDelay: `${i * 60}ms` }}>
              <div className="info-val">{m.value}</div>
              <div className="info-lbl">{m.label}</div>
            </div>
          ))}
        </div>

        {/* ---------- Main grid ---------- */}
        <div className="main">
          {/* Left: input */}
          <section className="card input-card">
            <div className="card-head">
              <h2>Device Telemetry</h2>
              <span className="badge">9 sensors · anonymized</span>
            </div>

            <div className="row2">
              <Field label="Reading Date">
                <input type="date" value={form.date}
                  onChange={(e) => update("date", e.target.value)} />
              </Field>
              <Field label="Device ID">
                <input value={form.device}
                  onChange={(e) => update("device", e.target.value)} />
              </Field>
            </div>

            <div className="metric-list">
              {METRICS.map((m, i) => {
                const val = Number(form[m.key]) || 0;
                const range = m.max - m.min || 1;
                const fillPct = Math.min(Math.max((val - m.min) / range, 0), 1) * 100;
                const importance = (sep[i] / maxSep) * 100;
                return (
                  <div className="metric-row" key={m.key}
                    style={{ animationDelay: `${i * 40}ms` }}>
                    <div className="metric-id">
                      <span className="metric-key">{m.key}</span>
                      <span className="metric-imp" title="how strongly this sensor separates failures from healthy in training data">
                        <span className="imp-bar" style={{ width: `${importance}%` }} />
                      </span>
                    </div>
                    <input type="number" value={form[m.key]}
                      onChange={(e) => update(m.key, e.target.value)} />
                    <div className="metric-meta">
                      <div className="range-track">
                        <span className="range-fill" style={{ width: `${fillPct}%` }} />
                      </div>
                      <span className="range-text">{fmt(m.min)}–{fmt(m.max)}</span>
                    </div>
                  </div>
                );
              })}
            </div>

            <button className="run" onClick={predict} disabled={loading}>
              {loading ? "Analyzing…" : "Run Diagnostic"}
            </button>
            {error && <p className="err">⚠ {error} — is the API running on :8000?</p>}
          </section>

          {/* Right: readout */}
          <section className="card readout-card">
            <div className="card-head">
              <h2>Diagnostic</h2>
              {result && <span className={`pill ${isFailure ? "pill-bad" : "pill-good"}`}>
                {isFailure ? "action needed" : "nominal"}
              </span>}
            </div>

            {!result && !loading && (
              <div className="empty">
                <div className="empty-ring" />
                <p>Awaiting telemetry</p>
                <span>Run a diagnostic to see the failure assessment</span>
              </div>
            )}

            {loading && (
              <div className="empty">
                <div className="empty-ring spin" />
                <p>Running inference</p>
                <span>Scaling features · querying model</span>
              </div>
            )}

            {result && !loading && (
              <div className="verdict">
                <div className={`donut ${isFailure ? "bad" : "good"}`}
                  style={{ "--pct": pct }}>
                  <div className="donut-hole">
                    <div className="donut-pct">{pct?.toFixed(0)}%</div>
                    <div className="donut-lbl">failure risk</div>
                  </div>
                </div>

                <div className={`verdict-tag ${isFailure ? "bad" : "good"}`}>
                  {isFailure ? "Failure Predicted" : "Healthy"}
                </div>

                <div className="verdict-rows">
                  <div className="vrow">
                    <span>Classification</span>
                    <b>{isFailure ? "1 · positive" : "0 · negative"}</b>
                  </div>
                  <div className="vrow">
                    <span>Confidence</span>
                    <b>{pct?.toFixed(2)}%</b>
                  </div>
                  <div className="vrow">
                    <span>Decision threshold</span>
                    <b>42%</b>
                  </div>
                  <div className="vrow">
                    <span>Model</span>
                    <b>Random Forest</b>
                  </div>
                </div>

                <p className="verdict-note">
                  {isFailure
                    ? "Risk exceeds the 42% operational threshold. Schedule inspection."
                    : "Risk below the 42% operational threshold. No action required."}
                </p>
              </div>
            )}
          </section>
        </div>

        <footer className="ftr">
          <span>FastAPI · ZenML · MLflow · Random Forest</span>
          <span>{API_URL}/predict</span>
        </footer>
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="field">
      <span className="field-lbl">{label}</span>
      {children}
    </label>
  );
}
