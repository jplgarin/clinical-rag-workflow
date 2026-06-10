"use strict";

const $ = (id) => document.getElementById(id);
const api = (path) => `/api${path}`;

const els = {
  domain: $("domain"),
  patient: $("patient"),
  findings: $("findings"),
  generate: $("generate"),
  sample: $("sample"),
  copy: $("copy"),
  download: $("download"),
  report: $("report"),
  error: $("error"),
  statusDot: $("status-dot"),
  reportActions: document.querySelector(".report-actions"),
};

let lastReport = null;

// Minimal per-domain sample payloads so "Load sample" works offline.
const SAMPLES = {
  adhd_neuraxis: {
    patient: { age: 12, sex: "M" },
    findings: {
      patient_age: 12,
      patient_sex: "M",
      adhd_probability: 0.73,
      top_shap_features: [
        { name: "F3_theta_rel", value: 0.28, shap_value: 0.15 },
        { name: "Fz_theta_beta_ratio", value: 3.2, shap_value: 0.12 },
        { name: "Cz_beta_rel", value: 0.11, shap_value: -0.08 },
      ],
      normative_comparison: {
        feature: "TBR_Fz",
        patient_value: 3.2,
        norm_mean: 2.1,
        norm_std: 0.4,
        z_score: 2.75,
      },
    },
  },
};

async function init() {
  await checkHealth();
  await loadDomains();
  bindEvents();
}

async function checkHealth() {
  try {
    const res = await fetch(api("/health"));
    const data = await res.json();
    setStatus(data.status === "ok", `v${data.version}`);
  } catch {
    setStatus(false, "offline");
  }
}

function setStatus(ok, title) {
  els.statusDot.className = `dot ${ok ? "ok" : "bad"}`;
  els.statusDot.title = title;
}

async function loadDomains() {
  try {
    const res = await fetch(api("/domains"));
    const domains = await res.json();
    els.domain.innerHTML = "";
    for (const d of domains) {
      const opt = document.createElement("option");
      opt.value = d.domain;
      opt.textContent = d.metadata?.display_name || d.domain;
      els.domain.appendChild(opt);
    }
    if (domains.length) loadSample();
  } catch {
    showError("Could not load domains. Is the API running?");
  }
}

function bindEvents() {
  els.generate.addEventListener("click", generate);
  els.sample.addEventListener("click", loadSample);
  els.copy.addEventListener("click", copyReport);
  els.download.addEventListener("click", downloadReport);
}

function loadSample() {
  const sample = SAMPLES[els.domain.value];
  if (!sample) return;
  els.patient.value = JSON.stringify(sample.patient, null, 2);
  els.findings.value = JSON.stringify(sample.findings, null, 2);
}

function parseJson(text, label) {
  try {
    return JSON.parse(text);
  } catch (e) {
    throw new Error(`${label} is not valid JSON: ${e.message}`);
  }
}

async function generate() {
  clearError();
  let body;
  try {
    body = {
      domain: els.domain.value,
      patient: parseJson(els.patient.value || "{}", "Patient context"),
      findings: parseJson(els.findings.value || "{}", "Findings"),
    };
  } catch (e) {
    return showError(e.message);
  }

  setLoading(true);
  try {
    const res = await fetch(api("/generate"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
    lastReport = data.report;
    renderReport(data.report);
  } catch (e) {
    showError(e.message);
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  els.generate.disabled = on;
  els.generate.innerHTML = on
    ? '<span class="spinner"></span>Generating'
    : "Generate report";
}

function renderReport(report) {
  els.report.innerHTML = "";
  els.reportActions.hidden = false;

  const meta = document.createElement("div");
  meta.className = "report-meta";
  meta.appendChild(badge(`confidence ${pct(report.overall_confidence)}`));
  meta.appendChild(badge(`${report.sections.length} sections`));
  meta.appendChild(badge(`${report.sources_used.length} sources`));
  if (report.warnings.length) {
    meta.appendChild(badge(`${report.warnings.length} warnings`, "warn"));
  }
  els.report.appendChild(meta);

  for (const section of report.sections) {
    els.report.appendChild(renderSection(section));
  }

  if (report.warnings.length) {
    els.report.appendChild(renderWarnings(report.warnings));
  }
}

function renderSection(section) {
  const wrap = document.createElement("div");
  wrap.className = "section";

  const h = document.createElement("h3");
  h.textContent = section.title;
  const conf = document.createElement("span");
  conf.className = "conf";
  conf.textContent = pct(section.confidence_score);
  h.appendChild(conf);
  wrap.appendChild(h);

  const body = document.createElement("div");
  body.className = "body";
  for (const para of section.content.split(/\n{2,}/)) {
    if (!para.trim()) continue;
    const p = document.createElement("p");
    p.innerHTML = withCitations(escapeHtml(para.trim()));
    body.appendChild(p);
  }
  wrap.appendChild(body);

  if (section.supporting_chunks?.length) {
    wrap.appendChild(renderSources(section.supporting_chunks));
  }
  return wrap;
}

function renderSources(chunks) {
  const d = document.createElement("details");
  d.className = "sources";
  const s = document.createElement("summary");
  s.textContent = `Sources (${chunks.length})`;
  d.appendChild(s);
  const ol = document.createElement("ol");
  for (const c of chunks) {
    const li = document.createElement("li");
    const name = c.metadata?.citation || c.source;
    li.innerHTML =
      `<span class="src-name">${escapeHtml(name)}</span> &middot; ` +
      `relevance ${pct(c.relevance_score)}<br>${escapeHtml(snippet(c.content))}`;
    ol.appendChild(li);
  }
  d.appendChild(ol);
  return d;
}

function renderWarnings(warnings) {
  const wrap = document.createElement("div");
  wrap.className = "warnings";
  wrap.innerHTML =
    "<h4>Verification warnings</h4><ul>" +
    warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("") +
    "</ul>";
  return wrap;
}

function withCitations(text) {
  return text.replace(/\[(\d+)\]/g, '<sup class="cite">[$1]</sup>');
}

function copyReport() {
  if (!lastReport) return;
  const text = lastReport.sections
    .map((s) => `# ${s.title}\n\n${s.content}`)
    .join("\n\n");
  navigator.clipboard.writeText(text);
  flash(els.copy, "Copied");
}

function downloadReport() {
  if (!lastReport) return;
  const blob = new Blob([JSON.stringify(lastReport, null, 2)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `report-${els.domain.value}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function flash(btn, label) {
  const original = btn.textContent;
  btn.textContent = label;
  setTimeout(() => (btn.textContent = original), 1200);
}

function badge(text, cls = "") {
  const b = document.createElement("span");
  b.className = `badge ${cls}`.trim();
  b.textContent = text;
  return b;
}

const pct = (x) => `${Math.round((x ?? 0) * 100)}%`;
const snippet = (t) => (t.length > 160 ? t.slice(0, 159) + "…" : t);

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function showError(msg) {
  els.error.textContent = msg;
  els.error.hidden = false;
}
function clearError() {
  els.error.hidden = true;
}

init();
