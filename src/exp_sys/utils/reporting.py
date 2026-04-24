"""HTML report generation for exp_sys experiment logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from exp_sys.utils.pricing import MODEL_PRICING


def _build_html(tasks: List[dict]) -> str:
    tasks_json   = json.dumps(tasks, ensure_ascii=False)
    pricing_json = json.dumps(MODEL_PRICING, ensure_ascii=False)
    return (
        _TEMPLATE
        .replace("__ALL_TASKS__",     tasks_json)
        .replace("__MODEL_PRICING__", pricing_json)
    )


def generate_html_report(json_file: str, output_html: str) -> None:
    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(json_file).read_text(encoding="utf-8"))
    out_path.write_text(_build_html([data]), encoding="utf-8")


def generate_multi_task_report(json_files: List[str], output_html: str) -> None:
    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(Path(f).read_text(encoding="utf-8")) for f in json_files]
    out_path.write_text(_build_html(tasks), encoding="utf-8")


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Experiment Report</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { height: 100%; }
body {
  height: 100%; display: flex; flex-direction: column; overflow: hidden;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 13px; background: #f1f5f9; color: #0f172a; line-height: 1.5;
}

/* ─── Header ─── */
.hdr {
  flex-shrink: 0;
  background: linear-gradient(135deg, #1e40af 0%, #0369a1 100%);
  color: #fff; padding: 10px 18px;
  display: flex; align-items: center; gap: 14px; min-height: 52px;
}
.hdr-info { flex: 1; min-width: 0; }
.hdr-title { font-size: 18px; font-weight: 700; }
.hdr-sub   { font-size: 11px; opacity: 0.8; margin-top: 2px; }
.hdr-badge { flex-shrink: 0; padding: 4px 12px; border-radius: 999px; font-size: 11px; font-weight: 700; }
.hdr-badge.success { background: #dcfce7; color: #15803d; }
.hdr-badge.fail    { background: #fee2e2; color: #b91c1c; }
.hdr-badge.multi   { background: rgba(255,255,255,0.22); color: #fff; }

/* ─── Metrics bar ─── */
.metrics {
  flex-shrink: 0; display: flex; gap: 6px;
  padding: 6px 14px; background: #fff;
  border-bottom: 1px solid #e2e8f0; overflow-x: auto;
}
.metric { flex-shrink: 0; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 5px 10px; }
.metric-k { font-size: 9px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-v { font-size: 15px; font-weight: 700; }
.metric-v.green { color: #15803d; }
.metric-v.red   { color: #b91c1c; }

/* ─── Workspace ─── */
.workspace { flex: 1; display: flex; overflow: hidden; min-height: 0; }

/* ─── Task sidebar ─── */
.task-sidebar {
  width: 195px; flex-shrink: 0; background: #fff;
  border-right: 1px solid #e2e8f0; overflow-y: auto;
  display: flex; flex-direction: column; gap: 5px; padding: 8px;
}
.task-sidebar.solo { display: none; }
.task-card {
  padding: 8px 10px; border-radius: 8px;
  border: 2px solid #e2e8f0; cursor: pointer; transition: all 0.12s;
}
.task-card:hover { border-color: #94a3b8; background: #f8fafc; }
.task-card.sel    { border-color: #2563eb; background: #eff6ff; }
.tc-id   { font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.4px; }
.tc-name { font-size: 12px; color: #0f172a; margin-top: 2px; line-height: 1.3; }
.tc-meta { font-size: 10px; color: #94a3b8; margin-top: 4px; display: flex; gap: 6px; }
.tc-ok  { color: #15803d; font-weight: 700; }
.tc-err { color: #b91c1c; font-weight: 700; }

/* ─── Main ─── */
.main { flex: 1; display: flex; min-width: 0; overflow: hidden; }

/* ─── Step list ─── */
.step-list {
  width: 240px; flex-shrink: 0; background: #fff;
  border-right: 1px solid #e2e8f0; overflow-y: auto;
}
.step-item {
  padding: 8px 12px; border-bottom: 1px solid #f1f5f9;
  cursor: pointer; border-left: 3px solid transparent; transition: background 0.1s;
}
.step-item:hover { background: #f8fafc; }
.step-item.sel   { background: #eff6ff; border-left-color: #2563eb; }
.si-row1 { display: flex; align-items: center; gap: 5px; margin-bottom: 3px; }
.si-num  { font-size: 12px; font-weight: 700; }
.si-lat  { font-size: 10px; color: #94a3b8; margin-left: auto; }
.si-action        { font-size: 11px; font-family: ui-monospace,'SF Mono',Consolas,monospace; color: #1d4ed8; word-break: break-word; line-height: 1.35; }
.si-action.done   { color: #15803d; }
.si-action.cannot { color: #b91c1c; font-weight: 700; font-family: inherit; }
.si-action.muted  { color: #64748b; font-style: italic; font-family: inherit; }

/* ─── Step detail ─── */
.step-detail { flex: 1; min-width: 0; display: flex; flex-direction: column; overflow: hidden; background: #f8fafc; }

/* ─── Tab bar ─── */
.tab-bar {
  flex-shrink: 0; display: flex; background: #fff;
  border-bottom: 2px solid #e2e8f0; padding: 0 12px; overflow-x: auto;
}
.tab-btn {
  flex-shrink: 0; padding: 8px 14px; font-size: 12px; font-weight: 500;
  color: #64748b; border: none; background: none; cursor: pointer;
  border-bottom: 2px solid transparent; margin-bottom: -2px;
  transition: all 0.12s; white-space: nowrap;
}
.tab-btn:hover { color: #0f172a; }
.tab-btn.act   { color: #2563eb; border-bottom-color: #2563eb; font-weight: 700; }

/* ─── Panel ─── */
.panel-wrap { flex: 1; overflow-y: auto; padding: 14px 16px; }
.panel     { display: none; }
.panel.act { display: block; }

/* ─── Badges ─── */
.badge { display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }
.b-oracle { background: #ede9fe; color: #6d28d9; }
.b-agent  { background: #cffafe; color: #0e7490; }
.b-judge  { background: #fef3c7; color: #92400e; }
.b-ok     { background: #dcfce7; color: #15803d; }
.b-no     { background: #fee2e2; color: #b91c1c; }
.b-gray   { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }

/* ─── Call card ─── */
.call-card {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
  padding: 14px 16px; margin-bottom: 14px;
}
.call-card:last-child { margin-bottom: 0; }
.call-hdr { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
.call-title { font-size: 14px; font-weight: 600; }
.call-stats { margin-left: auto; font-size: 11px; color: #94a3b8; }

/* ─── Section header ─── */
.sec-hdr {
  font-size: 10px; font-weight: 800; text-transform: uppercase;
  letter-spacing: 0.7px; color: #94a3b8;
  margin: 14px 0 8px; padding-top: 12px; border-top: 1px solid #f1f5f9;
}
.sec-hdr:first-child { margin-top: 0; padding-top: 0; border-top: none; }

/* ─── Message blocks (prompt) ─── */
.msg-block {
  border-radius: 6px; overflow: hidden;
  margin-bottom: 8px; border: 1px solid;
}
.msg-role-bar  { font-size: 10px; font-weight: 800; letter-spacing: 0.8px; padding: 4px 10px; text-transform: uppercase; }
.msg-content   { padding: 10px 12px; font-size: 12px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; max-height: 500px; overflow-y: auto; }
.msg-system    { border-color: #a855f7; }
.msg-system    .msg-role-bar { background: #faf5ff; color: #6d28d9; }
.msg-system    .msg-content  { background: #fdfaff; }
.msg-user      { border-color: #3b82f6; }
.msg-user      .msg-role-bar { background: #eff6ff; color: #1e40af; }
.msg-user      .msg-content  { background: #f8fbff; }
.msg-assistant { border-color: #22c55e; }
.msg-assistant .msg-role-bar { background: #f0fdf4; color: #15803d; }
.msg-assistant .msg-content  { background: #fafffe; }

/* ─── Response ─── */
.resp-text {
  background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
  padding: 12px; font-size: 12px; line-height: 1.7;
  white-space: pre-wrap; word-break: break-word;
  max-height: 400px; overflow-y: auto; color: #1e293b;
}

/* ─── Thought box (raw_reasoning) ─── */
.thought-box {
  background: #fefce8; border: 1px solid #fde68a; border-left: 3px solid #ca8a04;
  border-radius: 6px; padding: 10px 12px; font-size: 12px; line-height: 1.7;
  white-space: pre-wrap; word-break: break-word;
  max-height: 260px; overflow-y: auto; color: #713f12;
}
.cannot-box {
  background: #fef2f2; border: 1px solid #fecaca; border-left: 3px solid #dc2626;
  border-radius: 6px; padding: 10px 12px; font-size: 12px; line-height: 1.6;
  color: #991b1b; font-weight: 500;
}

/* ─── Action block ─── */
.action-block {
  background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
  padding: 10px 14px; font-family: ui-monospace,'SF Mono',Consolas,monospace;
  font-size: 13px; font-weight: 600; color: #1e40af;
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 10px;
}
.action-block.done { background: #f0fdf4; border-color: #86efac; color: #15803d; }

/* ─── Obs block ─── */
.obs-block {
  background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
  padding: 10px 12px; font-size: 12px;
  font-family: ui-monospace,'SF Mono',Consolas,monospace;
  white-space: pre-wrap; word-break: break-word;
  max-height: 220px; overflow-y: auto; line-height: 1.5;
}

/* ─── Context ─── */
.ctx-field { margin-bottom: 12px; }
.ctx-key   { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; margin-bottom: 4px; }
.rel-chip  { display: inline-block; font-size: 11px; font-family: monospace; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; padding: 2px 6px; margin: 2px; color: #475569; }

/* ─── Overview row (Step Overview tab) ─── */
.ov-row    { display: grid; grid-template-columns: 110px 1fr; gap: 10px 14px; align-items: start; padding: 8px 0; border-bottom: 1px dashed #e2e8f0; }
.ov-row:last-child { border-bottom: none; }
.ov-key    { font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; padding-top: 2px; }
.ov-val    { font-size: 12px; line-height: 1.55; color: #0f172a; word-break: break-word; }
.ov-val.code     { font-family: ui-monospace,'SF Mono',Consolas,monospace; }
.ov-val.cannot   { color: #b91c1c; font-weight: 600; }
.ov-val.success  { color: #15803d; }
.ov-val.muted    { color: #94a3b8; font-style: italic; }
.si-feedback     { font-size: 10px; margin-top: 3px; line-height: 1.3; word-break: break-word; }
.si-feedback.cannot { color: #b91c1c; }
.si-feedback.warn   { color: #b45309; }

/* ─── Suite aggregate bar (multi-task combined reports only) ─── */
.suite-bar {
  flex-shrink: 0; display: none;           /* shown only when multi-task */
  background: #1e293b; color: #e2e8f0;
  padding: 6px 16px; gap: 0;
  overflow-x: auto; align-items: stretch;
}
.suite-bar.multi { display: flex; }
.suite-group {
  display: flex; flex-direction: column; justify-content: center;
  padding: 4px 16px; flex-shrink: 0;
  border-right: 1px solid #334155;
}
.suite-group:last-child { border-right: none; }
.suite-label { font-size: 9px; font-weight: 700; text-transform: uppercase;
               letter-spacing: 0.6px; color: #94a3b8; margin-bottom: 2px; }
.suite-value { font-size: 18px; font-weight: 800; line-height: 1; }
.suite-value.ok  { color: #4ade80; }
.suite-value.bad { color: #f87171; }
.suite-value.neu { color: #e2e8f0; }
.suite-sub   { font-size: 10px; color: #64748b; margin-top: 2px; }
/* Success rate big badge (leftmost) */
.suite-rate {
  display: flex; flex-direction: column; justify-content: center;
  padding: 4px 20px 4px 16px; flex-shrink: 0;
  border-right: 1px solid #334155;
  min-width: 110px;
}
.suite-rate-pct  { font-size: 28px; font-weight: 900; line-height: 1; }
.suite-rate-frac { font-size: 11px; color: #94a3b8; margin-top: 3px; }

@media (max-width: 900px) {
  .task-sidebar { width: 160px; }
  .step-list    { width: 200px; }
}
</style>
</head>
<body>

<header class="hdr">
  <div class="hdr-info">
    <div class="hdr-title" id="hdrTitle">Loading…</div>
    <div class="hdr-sub"   id="hdrSub"></div>
  </div>
  <div id="hdrBadge" class="hdr-badge"></div>
</header>

<!-- Suite aggregate bar — populated by initSuitebar(), hidden for single-task -->
<div class="suite-bar" id="suiteBar"></div>

<div class="metrics" id="metrics"></div>

<div class="workspace">
  <aside class="task-sidebar" id="taskSidebar"></aside>
  <div class="main">
    <div class="step-list"   id="stepList"></div>
    <div class="step-detail">
      <div class="tab-bar"    id="tabBar"></div>
      <div class="panel-wrap" id="panelWrap"></div>
    </div>
  </div>
</div>

<script>
const ALL_TASKS     = __ALL_TASKS__;
/* Pricing in USD per 1M tokens: { modelId: [inputPrice, outputPrice] } */
const MODEL_PRICING = __MODEL_PRICING__;

/* Compute cost from tokens + pricing table.
   Returns null if model is unknown (caller should fall back to stored cost). */
function computeCost(model, promptTokens, completionTokens) {
  if (!model) return null;
  const p = MODEL_PRICING[model];
  if (!p) return null;
  return (promptTokens * p[0] + completionTokens * p[1]) / 1e6;
}

/* ── Display name mapping ── */
const METHOD_DISPLAY = {
  react:           'ReAct',
  crms:            'CRMS',
  pefa:            'PEFA',
  pefa_wo_history: 'PEFA w/o History',
  drms:            'DRMS',
};

const CALL_LABELS = {
  agent_selection:    'Oracle → Agent',
  action_selection:   'Action Selection',
  oracle_reasoning:   'Oracle Reasoning',
  subgoal_extraction: 'Subgoal Extraction',
  action_extraction:  'Action Extraction',
  judge_verify:       'Judge Verification',
  agent_reasoning:    'Agent Reasoning',
};

/* ── Helpers ── */
const esc = s => {
  const d = document.createElement('div');
  d.textContent = (s == null) ? '' : String(s);
  return d.innerHTML;
};

function agentBadge(agent) {
  if (!agent) return 'b-gray';
  const l = agent.toLowerCase();
  if (l === 'oracle') return 'b-oracle';
  if (l.includes('judge')) return 'b-judge';
  return 'b-agent';
}

function tokenLine(call) {
  const t = call.tokens || {};
  const parts = [];
  if (t.prompt     != null) parts.push('in\\u00a0' + t.prompt);
  if (t.completion != null) parts.push('out\\u00a0' + t.completion);
  if (t.reasoning)          parts.push('reason\\u00a0' + t.reasoning);
  if (call.latency_ms != null) parts.push((call.latency_ms / 1000).toFixed(2) + 's');
  /* Prefer cost computed from the embedded pricing table (lets historical
     logs that were written before gpt-5-mini pricing was correct render
     accurate numbers). Fall back to whatever the log wrote at run time. */
  const computed = computeCost(curTaskModel, t.prompt, t.completion);
  const effective = (computed != null) ? computed : call.cost;
  if (effective)               parts.push('$' + Number(effective).toFixed(5));
  return parts.join(' · ');
}

/* ── Prompt renderer ── */
function renderPrompt(raw) {
  let msgs = null;

  if (typeof raw === 'string') {
    const t = raw.trimStart();
    if (t.startsWith('[') || t.startsWith('{')) {
      try {
        const p = JSON.parse(raw);
        if (Array.isArray(p) && p.length > 0 && p[0] && p[0].role) msgs = p;
      } catch (_) {}
    }
  } else if (Array.isArray(raw) && raw.length > 0 && raw[0] && raw[0].role) {
    msgs = raw;
  }

  /* Always prepend system message if not already present */
  if (!msgs) {
    msgs = [{ role: 'user', content: String(raw == null ? '' : raw) }];
  }
  if (msgs[0] && msgs[0].role !== 'system') {
    msgs = [{ role: 'system', content: 'You are a helpful assistant.' }, ...msgs];
  }

  return msgs.map(msg => {
    const cls  = { system: 'msg-system', user: 'msg-user', assistant: 'msg-assistant' }[msg.role] || 'msg-user';
    const body = (msg.content == null) ? '' : String(msg.content);
    return '<div class="msg-block ' + cls + '">' +
      '<div class="msg-role-bar">' + esc(msg.role) + '</div>' +
      '<div class="msg-content">' + esc(body) + '</div>' +
      '</div>';
  }).join('');
}

/* ── Call panel renderer ── */
function renderCallPanel(call) {
  const label = CALL_LABELS[call.call_type] || call.call_type;
  const bc    = agentBadge(call.agent);
  const tLine = tokenLine(call);

  /* Execution-result badge — highlights CANNOT / unexpected_format quickly */
  const er = call.execution_result || '';
  let erBadge = '';
  if (er === 'cannot')            erBadge = '<span class="badge b-no">CANNOT</span>';
  else if (er === 'unexpected_format') erBadge = '<span class="badge b-no">unparsed</span>';
  else if (er === 'success' && call.parsed_action) erBadge = '<span class="badge b-ok">action</span>';

  /* Thought / Reasoning Summary box (raw_reasoning):
     - For reasoning models (gpt-5 / o1 / o3 families) this is OpenAI's
       reasoning.summary from the Responses API — a post-hoc summary of
       hidden reasoning tokens.
     - For non-reasoning models this is the 'Thought:' line parsed from
       the visible response content.
     We label by which source we used; a non-zero tokens.reasoning means summary. */
  const isSummary = !!(call.tokens && call.tokens.reasoning);
  const thoughtLabel = isSummary ? 'Reasoning Summary (hidden)' : 'Thought';
  const thoughtHtml = call.raw_reasoning
    ? '<div class="sec-hdr">' + thoughtLabel + '</div>' +
      '<div class="thought-box">' + esc(call.raw_reasoning) + '</div>'
    : '';

  const parsed = call.parsed_action
    ? '<div style="margin-top:8px"><span class="badge b-agent">&#8627; action</span>&nbsp;<code style="font-size:12px">' + esc(call.parsed_action) + '</code></div>'
    : '';

  return '<div class="call-card">' +
    '<div class="call-hdr">' +
      '<span class="badge ' + bc + '">' + esc(call.agent || '?') + '</span>' +
      '<span class="call-title">' + esc(label) + '</span>' +
      (erBadge ? '&nbsp;' + erBadge : '') +
      (tLine ? '<span class="call-stats">' + tLine + '</span>' : '') +
    '</div>' +
    '<div class="sec-hdr">Prompt</div>' +
    renderPrompt(call.prompt) +
    thoughtHtml +
    '<div class="sec-hdr">Response (raw)</div>' +
    '<div class="resp-text">' + esc(call.response || '(no response)') + '</div>' +
    parsed +
    '</div>';
}

/* ── Environment panel renderer ── */
function renderEnvPanel(ec) {
  if (!ec) {
    return '<div class="call-card"><p style="color:#94a3b8;font-style:italic">No environment change recorded.</p></div>';
  }
  const isDone = !!ec.success;
  const rels   = ec.changed_relations || [];

  let html = '<div class="call-card">' +
    '<div class="call-hdr">' +
      '<span class="call-title">Environment Change</span>' +
      '<span class="badge ' + (isDone ? 'b-ok' : 'b-no') + '">' + (isDone ? '&#10003; Task Complete' : '&#8635; Step') + '</span>' +
    '</div>' +
    '<div class="sec-hdr">Action Executed</div>' +
    '<div class="action-block ' + (isDone ? 'done' : '') + '">&#9658; ' + esc(ec.action || '—') + '</div>';

  if (rels.length > 0) {
    html += '<div class="sec-hdr">Changed Relations</div>' +
      '<div style="margin-bottom:10px">' +
      rels.map(r => '<span class="rel-chip">' + esc(typeof r === 'string' ? r : JSON.stringify(r)) + '</span>').join('') +
      '</div>';
  }
  if (ec.after  && ec.after.obs_text)  html += '<div class="sec-hdr">Observation After</div>'  + '<div class="obs-block">' + esc(ec.after.obs_text)  + '</div>';
  if (ec.before && ec.before.obs_text) html += '<div class="sec-hdr">Observation Before</div>' + '<div class="obs-block">' + esc(ec.before.obs_text) + '</div>';
  html += '</div>';
  return html;
}

/* ── Overview panel renderer ── */
function renderOverviewPanel(step) {
  const ov = step.overview || {};
  const ec = step.environment_change || {};
  const hasOv = Object.values(ov).some(v => v != null && v !== '');
  if (!hasOv && !ec.action && !ec.success) {
    return '<div class="call-card"><p style="color:#94a3b8;font-style:italic">No overview recorded for this step.</p></div>';
  }
  const rows = [];
  function row(key, val, cls) {
    const v = (val == null || val === '')
      ? '<span class="ov-val muted">(none)</span>'
      : '<div class="ov-val ' + (cls || '') + '">' + esc(val) + '</div>';
    rows.push('<div class="ov-row"><div class="ov-key">' + esc(key) + '</div>' + v + '</div>');
  }
  row('Subgoal',  ov.subgoal, 'code');
  row('Thought',  ov.thought);
  if (ov.cannot_reason) {
    row('CANNOT', ov.cannot_reason, 'cannot');
  } else {
    row('Action', ov.action, 'code');
  }
  const isDone = !!ec.success;
  let envText = ov.env_outcome || (ec.action ? ('action ' + ec.action + '; task_done=' + isDone) : null);
  row('Env Result', envText, isDone ? 'success' : '');
  return '<div class="call-card">' + rows.join('') + '</div>';
}

/* ── Context (step observation) panel renderer ── */
function renderContextPanel(obs) {
  if (!obs || Object.keys(obs).length === 0) {
    return '<div class="call-card"><p style="color:#94a3b8;font-style:italic">No observation data.</p></div>';
  }
  let html = '<div class="call-card">';
  for (const [k, v] of Object.entries(obs)) {
    html += '<div class="ctx-field"><div class="ctx-key">' + esc(k) + '</div>';
    if (v == null || v === '' || v === '(none)') {
      html += '<span style="color:#94a3b8;font-style:italic;font-size:12px">(none)</span>';
    } else if (typeof v === 'string') {
      html += '<div class="obs-block">' + esc(v) + '</div>';
    } else {
      html += '<div class="obs-block">' + esc(JSON.stringify(v, null, 2)) + '</div>';
    }
    html += '</div>';
  }
  html += '</div>';
  return html;
}

/* ── State ── */
let curTask      = null;
let curTaskIdx   = 0;
let curTaskModel = '';  /* set in loadTask; used by computeCost */

/* ── Sidebar ── */
/* ── Suite aggregate bar (combined reports, ALL_TASKS.length > 1) ── */
function initSuitebar() {
  const bar = document.getElementById('suiteBar');
  if (ALL_TASKS.length <= 1) return;
  bar.classList.add('multi');

  /* Aggregate across all tasks */
  let totalTasks = ALL_TASKS.length;
  let successCount = 0;
  let totalSteps = 0, totalGtSteps = 0, gtCount = 0;
  let totalCost = 0, totalIn = 0, totalOut = 0, totalReason = 0, totalLatMs = 0;
  let totalCalls = 0;

  ALL_TASKS.forEach(task => {
    const m  = task.metadata || {};
    const sm = task.summary  || {};
    const steps = task.steps || [];
    const model = m.lm_id || '';

    if (sm.success) successCount++;
    totalSteps += sm.total_steps || 0;
    if (m.ground_truth_steps != null) {
      totalGtSteps += Number(m.ground_truth_steps);
      gtCount++;
    }

    /* Cost: recompute from pricing table when possible */
    let taskCost = 0;
    let taskCostKnown = !!model && !!MODEL_PRICING[model];
    steps.forEach(s => (s.llm_calls || []).forEach(c => {
      totalCalls++;
      const t = c.tokens || {};
      totalIn    += t.prompt     || 0;
      totalOut   += t.completion || 0;
      totalReason += t.reasoning || 0;
      totalLatMs  += c.latency_ms || 0;
      if (taskCostKnown) {
        const cc = computeCost(model, t.prompt || 0, t.completion || 0);
        if (cc != null) taskCost += cc;
      }
    }));
    totalCost += taskCostKnown ? taskCost : (sm.total_cost || 0);
  });

  const successPct = totalTasks ? (successCount / totalTasks * 100) : 0;
  const avgSteps   = totalTasks ? (totalSteps / totalTasks) : 0;
  const avgGt      = gtCount    ? (totalGtSteps / gtCount)  : 0;
  const stepRatio  = avgGt      ? (avgSteps / avgGt)        : null;
  const pctCls     = successPct >= 60 ? 'ok' : (successPct >= 30 ? 'neu' : 'bad');

  function grp(label, val, sub) {
    return '<div class="suite-group">' +
      '<div class="suite-label">' + esc(label) + '</div>' +
      '<div class="suite-value neu">' + esc(val) + '</div>' +
      (sub ? '<div class="suite-sub">' + esc(sub) + '</div>' : '') +
      '</div>';
  }

  bar.innerHTML =
    /* Big success-rate block on the left */
    '<div class="suite-rate">' +
      '<div class="suite-label">Success Rate</div>' +
      '<div class="suite-rate-pct ' + pctCls + '">' + successPct.toFixed(0) + '%</div>' +
      '<div class="suite-rate-frac">' + successCount + ' / ' + totalTasks + ' tasks</div>' +
    '</div>' +
    grp('Avg Steps',    avgSteps.toFixed(1),
        avgGt ? 'gt avg ' + avgGt.toFixed(1) + (stepRatio ? ' (' + stepRatio.toFixed(2) + '×)' : '') : '') +
    grp('Total Cost',   '$' + totalCost.toFixed(4)) +
    grp('In Tokens',    totalIn.toLocaleString()) +
    grp('Out Tokens',   totalOut.toLocaleString()) +
    (totalReason > 0 ? grp('Reason Tokens', totalReason.toLocaleString()) : '') +
    grp('Total Calls',  totalCalls.toLocaleString()) +
    grp('Latency',      totalLatMs > 0 ? (totalLatMs / 1000).toFixed(0) + 's' : 'N/A');
}

function initSidebar() {
  const sidebar = document.getElementById('taskSidebar');
  if (ALL_TASKS.length <= 1) {
    sidebar.classList.add('solo');
    return;
  }
  ALL_TASKS.forEach((task, i) => {
    const m  = task.metadata || {};
    const sm = task.summary  || {};
    const ok = !!sm.success;
    const card = document.createElement('div');
    card.className = 'task-card' + (i === 0 ? ' sel' : '');
    card.id = 'tc' + i;
    card.innerHTML =
      '<div class="tc-id">Task ' + esc(m.task_id != null ? m.task_id : i) + '</div>' +
      '<div class="tc-name">' + esc(m.task_name || '—') + '</div>' +
      '<div class="tc-meta">' +
        '<span class="' + (ok ? 'tc-ok' : 'tc-err') + '">' + (ok ? '&#10003;' : '&#10007;') + '</span>' +
        '<span>' + (sm.total_steps || 0) + ' steps</span>' +
      '</div>';
    card.onclick = () => loadTask(i);
    sidebar.appendChild(card);
  });
}

/* ── Load task ── */
function loadTask(idx) {
  curTaskIdx   = idx;
  curTask      = ALL_TASKS[idx] || {};
  const m      = curTask.metadata || {};
  const sm     = curTask.summary  || {};
  const steps  = curTask.steps   || [];
  curTaskModel = m.lm_id || '';

  document.querySelectorAll('.task-card').forEach((c, i) => c.classList.toggle('sel', i === idx));

  /* Header */
  const methodKey  = (m.method || '').toLowerCase();
  const methodName = METHOD_DISPLAY[methodKey] || m.method || 'Experiment';
  const lmId       = m.lm_id || '';
  document.getElementById('hdrTitle').textContent = methodName;
  document.getElementById('hdrSub').textContent   = [
    lmId,
    'env' + (m.env_id != null ? m.env_id : '?'),
    'Task ' + (m.task_id != null ? m.task_id : '?'),
    (m.timestamp || '').slice(0, 16).replace('T', ' '),
  ].filter(Boolean).join(' · ');

  const multi = ALL_TASKS.length > 1;
  const badge = document.getElementById('hdrBadge');
  if (multi) {
    badge.className   = 'hdr-badge multi';
    badge.textContent = ALL_TASKS.length + ' tasks';
  } else {
    badge.className   = 'hdr-badge ' + (sm.success ? 'success' : 'fail');
    badge.textContent = sm.success ? '\\u2713 Success' : '\\u2717 Failure';
  }

  /* Metrics */
  const totalCalls = steps.reduce((a, s) => a + (s.llm_calls || []).length, 0);
  const latMs      = sm.total_latency_ms;

  /* Reasoning-token total (sum across all calls) and re-computed total cost
     from embedded pricing table; fall back to stored sm.total_cost only if
     the model isn't in the table. */
  let totalReasoning = 0;
  let recomputedCost = 0;
  let costKnown      = !!curTaskModel && !!MODEL_PRICING[curTaskModel];
  steps.forEach(s => (s.llm_calls || []).forEach(c => {
    const t = c.tokens || {};
    if (t.reasoning) totalReasoning += t.reasoning;
    if (costKnown) {
      const cc = computeCost(curTaskModel, t.prompt || 0, t.completion || 0);
      if (cc != null) recomputedCost += cc;
    }
  }));
  const effectiveCost = costKnown ? recomputedCost : Number(sm.total_cost || 0);

  const metricRows = [
    ['Result',     sm.success ? 'Success' : 'Failure',                            sm.success ? 'green' : 'red'],
    ['Steps',      (sm.total_steps || 0) + ' / ' + (m.ground_truth_steps || '?'), ''],
    ['LLM Calls',  totalCalls,                                                    ''],
    ['In Tokens',  Number(sm.total_tokens_in  || 0).toLocaleString(),             ''],
    ['Out Tokens', Number(sm.total_tokens_out || 0).toLocaleString(),             ''],
  ];
  if (totalReasoning > 0) {
    metricRows.push(['Reasoning Tokens', totalReasoning.toLocaleString(), '']);
  }
  metricRows.push(['Cost', '$' + effectiveCost.toFixed(4), '']);
  metricRows.push(['Latency', latMs != null ? (latMs / 1000).toFixed(1) + 's' : 'N/A', '']);
  document.getElementById('metrics').innerHTML = metricRows.map(([k, v, c]) =>
    '<div class="metric">' +
      '<div class="metric-k">' + esc(k) + '</div>' +
      '<div class="metric-v ' + c + '">' + esc(v) + '</div>' +
    '</div>'
  ).join('');

  /* Step list */
  const sl = document.getElementById('stepList');
  sl.innerHTML = '';
  steps.forEach((step, i) => {
    const ec     = step.environment_change || {};
    const ov     = step.overview || {};
    const calls  = step.llm_calls || [];
    const isDone = !!ec.success;
    const lat    = calls.reduce((a, c) => a + (c.latency_ms || 0), 0);

    /* First try to find the agent-selection call's parsed_action / response snippet as last-resort hint */
    const lastCall = calls.length ? calls[calls.length - 1] : null;
    const lastResp = lastCall ? (lastCall.response || '') : '';
    const lastRespSnip = lastResp
      ? lastResp.replace(/\s+/g, ' ').slice(0, 80) + (lastResp.length > 80 ? '…' : '')
      : '';

    /* Cascade: pick the most informative one-line action summary we have.
       Never fall back to '—' as long as any signal is available. */
    let actionLine, actionCls = '';
    let fbLine = '';
    if (ov.cannot_reason) {
      actionLine = 'CANNOT';
      actionCls  = 'cannot';
      fbLine = '<div class="si-feedback cannot">' + esc(ov.cannot_reason) + '</div>';
    } else if (ec.action) {
      actionLine = ec.action;
      if (isDone) actionCls = 'done';
      if (ec.success === false && ov.env_outcome && /error|failed|unparsed/i.test(ov.env_outcome)) {
        fbLine = '<div class="si-feedback warn">' + esc(ov.env_outcome) + '</div>';
      }
    } else if (ov.action) {
      actionLine = ov.action;
    } else if (ov.env_outcome) {
      actionLine = ov.env_outcome.replace(/\s+/g, ' ').slice(0, 80);
      actionCls  = 'cannot';
    } else if (lastRespSnip) {
      actionLine = lastRespSnip;
      actionCls  = 'muted';
    } else {
      actionLine = '(no action recorded)';
      actionCls  = 'muted';
    }

    /* Optional second hint line for non-CANNOT steps: show subgoal snippet */
    if (!fbLine && ov.subgoal) {
      const sg = ov.subgoal.replace(/\s+/g, ' ').slice(0, 80);
      fbLine = '<div class="si-feedback">' + esc(sg) + '</div>';
    }

    const badgeCls = isDone ? 'b-ok' : (ov.cannot_reason ? 'b-no' : 'b-gray');
    const badgeGlyph = isDone ? '&#10003;' : (ov.cannot_reason ? '&#10007;' : '&#8230;');

    const el = document.createElement('div');
    el.className = 'step-item' + (i === 0 ? ' sel' : '');
    el.id = 'si' + i;
    el.innerHTML =
      '<div class="si-row1">' +
        '<span class="si-num">Step ' + esc(step.step_id != null ? step.step_id : i) + '</span>' +
        '<span class="badge ' + badgeCls + '" style="font-size:10px;padding:1px 6px">' + badgeGlyph + '</span>' +
        (lat > 0 ? '<span class="si-lat">' + (lat / 1000).toFixed(1) + 's</span>' : '') +
      '</div>' +
      '<div class="si-action ' + actionCls + '">' + esc(actionLine) + '</div>' +
      fbLine;
    el.onclick = () => loadStep(i);
    sl.appendChild(el);
  });

  loadStep(0);
}

/* ── Load step ── */
function loadStep(idx) {
  const steps = curTask.steps || [];
  const step  = steps[idx];

  document.querySelectorAll('.step-item').forEach((el, i) => el.classList.toggle('sel', i === idx));

  const tabBar    = document.getElementById('tabBar');
  const panelWrap = document.getElementById('panelWrap');
  tabBar.innerHTML    = '';
  panelWrap.innerHTML = '';

  if (!step) {
    panelWrap.innerHTML = '<p style="padding:20px;color:#94a3b8;font-style:italic">No step data.</p>';
    return;
  }

  const calls = step.llm_calls || [];

  /* Build tab list: Overview → Context → per-call tabs → Environment */
  const tabs = [];
  tabs.push({ label: 'Overview', content: renderOverviewPanel(step) });
  tabs.push({ label: 'Context',  content: renderContextPanel(step.observation) });

  /* Deduplicate tab labels when same call_type appears multiple times */
  const typeCount = {};
  calls.forEach(c => { typeCount[c.call_type] = (typeCount[c.call_type] || 0) + 1; });
  const typeSeen = {};
  calls.forEach((c, i) => {
    typeSeen[c.call_type] = (typeSeen[c.call_type] || 0) + 1;
    const base  = CALL_LABELS[c.call_type] || c.call_type;
    const label = typeCount[c.call_type] > 1 ? base + ' (' + typeSeen[c.call_type] + ')' : base;
    tabs.push({ label, content: renderCallPanel(c) });
  });
  tabs.push({ label: 'Environment', content: renderEnvPanel(step.environment_change) });

  tabs.forEach((tab, i) => {
    const btn = document.createElement('button');
    btn.className   = 'tab-btn' + (i === 0 ? ' act' : '');
    btn.textContent = tab.label;
    btn.onclick = () => {
      document.querySelectorAll('.tab-btn').forEach((b, j) => b.classList.toggle('act', j === i));
      document.querySelectorAll('.panel').forEach((p, j) => p.classList.toggle('act', j === i));
    };
    tabBar.appendChild(btn);

    const panel = document.createElement('div');
    panel.className = 'panel' + (i === 0 ? ' act' : '');
    panel.innerHTML = tab.content;
    panelWrap.appendChild(panel);
  });
}

initSidebar();
initSuitebar();
loadTask(0);
</script>
</body>
</html>
"""
