// Single-page UI: talks only to the JSON API under /api.
const $ = (id) => document.getElementById(id);
let currentRun = null;

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x) => (x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`);
const when = (iso) => { try { return new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }); } catch { return iso; } };

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (res.status === 204) return null;
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = body.detail;
    throw new Error(Array.isArray(d) ? d.map((x) => (typeof x === "string" ? x : x.msg)).join("\n") : d || res.statusText);
  }
  return body;
}

// ---------- model status ----------
async function loadHealth() {
  const badge = $("model-badge"), text = $("model-text");
  try {
    const h = await api("/api/health");
    const m = h.model || {};
    if (!h.model_loaded) { badge.className = "model down"; text.textContent = "No model yet: run train.py"; return; }
    if (m.trained_on === "synthetic") { badge.className = "model demo"; text.textContent = "Demo model · synthetic data"; return; }
    badge.className = "model ready";
    text.textContent = m.holdout
      ? `FlyRank model · AUC ${m.holdout.auc} on ${m.holdout.clients} held-out clients`
      : `FlyRank model · AUC ${m.auc} in grouped CV · ${m.clients} clients`;
  } catch { badge.className = "model down"; text.textContent = "API not reachable"; }
}

// ---------- saved runs ----------
async function loadRuns() {
  const runs = await api("/api/runs");
  $("runs").innerHTML = runs.length ? runs.map((r) => `
    <li><button type="button" data-id="${r.id}" class="${currentRun === r.id ? "active" : ""}">
      <span class="name">#${r.id} · ${esc(r.filename)}</span>
      <small>${when(r.created_at)} · ${r.n_pages} pages · reviewed ${r.reviewed} of ${r.queue_size}</small>
      <span class="prog"><i style="width:${Math.min(100, Math.round((100 * r.reviewed) / Math.max(1, r.queue_size)))}%"></i></span>
    </button></li>`).join("") : '<li class="muted">No runs yet. Upload a CSV or try the demo data.</li>';
  $("runs").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => showRun(+b.dataset.id)));
  return runs;
}

// ---------- one run ----------
function tile(value, label, cls = "", extra = "") {
  return `<div class="tile ${cls}"><span class="v">${value}</span><span class="k">${label}</span>${extra}</div>`;
}

function render(run) {
  currentRun = run.id;
  $("empty").hidden = true; $("run").hidden = false;
  $("run-meta").textContent = `Run #${run.id} · ${when(run.created_at)} · top ${run.k}, max ${run.per_client} per client`;
  $("run-title").textContent = run.filename;
  $("download").href = `/api/runs/${run.id}/queue.csv`;
  resetDelete();

  const s = run.summary;
  let tiles = tile(run.n_pages.toLocaleString(), "pages uploaded")
    + tile(s.queue_size, "in the review queue")
    + tile(`${s.reviewed}<small class="muted"> / ${s.queue_size}</small>`, "reviewed so far", "",
           `<span class="prog"><i style="width:${Math.round((100 * s.reviewed) / Math.max(1, s.queue_size))}%"></i></span>`)
    + tile(pct(s.precision_from_feedback), "of reviewed pages really declined");
  if (s.precision_observed !== null) {
    tiles += tile(pct(s.precision_observed), s.file_base_rate !== null
      ? `of the queue declined, vs ${pct(s.file_base_rate)} of all pages${s.lift ? ` · ${s.lift}× the random rate` : ""}`
      : "of the queue declined (outcome column)", "key");
  }
  $("tiles").innerHTML = tiles;

  const counts = { high: 0, medium: 0, low: 0 };
  run.results.forEach((r) => { counts[r.band] += 1; });
  const n = Math.max(1, run.results.length);
  $("mix").innerHTML = `<span>Queue by risk</span>
    <span class="mixbar">${["high", "medium", "low"].map((b) => `<i style="width:${(100 * counts[b]) / n}%;background:var(--${b})"></i>`).join("")}</span>
    <span class="legend">${["high", "medium", "low"].map((b) => `<span style="color:var(--${b})"><span style="color:var(--muted)">${counts[b]} ${b}</span></span>`).join("")}</span>`;

  const top = run.results.length ? run.results[0].score : 0;
  $("notice").hidden = top >= 0.5;
  $("notice").textContent = "No page in this file scores above 0.5. The queue is still ordered by risk, but these pages look less like the ones that declined in training. If this is the synthetic demo file and the model was trained on FlyRank data, that mismatch is expected.";

  $("rows").innerHTML = run.results.map((r) => `
    <tr>
      <td class="rank">${r.rank}</td>
      <td class="page" data-l="Page"><span class="pid">${esc(r.content_id)}</span><br><span class="client">${esc(r.client_id)}</span></td>
      <td data-l="Risk"><div class="risk">
        <div class="top-line"><span class="score">${r.score.toFixed(3)}</span><span class="band ${r.band}">${r.band}</span></div>
        <div class="meter"><i class="${r.band}" style="width:${Math.round(r.score * 100)}%"></i></div></div></td>
      <td data-l="Why it's in the queue">${r.reasons.length ? `<ul class="why">${r.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : '<span class="muted">—</span>'}</td>
      <td data-l="Check">${r.review_flags.length ? `<div class="flags">${r.review_flags.map((x) => `<span class="flag">${esc(x)}</span>`).join("")}</div>` : '<span class="muted">—</span>'}</td>
      <td data-l="After review"><div class="seg" role="group" aria-label="Review result for ${esc(r.content_id)}">
        <button type="button" data-c="${esc(r.content_id)}" data-v="1" class="${r.feedback === 1 ? "on" : ""}" aria-pressed="${r.feedback === 1}">Declined</button>
        <button type="button" data-c="${esc(r.content_id)}" data-v="0" class="${r.feedback === 0 ? "on" : ""}" aria-pressed="${r.feedback === 0}">Held up</button>
      </div></td>
    </tr>`).join("");
  $("rows").querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => sendFeedback(b.dataset.c, b.dataset.v === "1")));
}

async function showRun(id) { render(await api(`/api/runs/${id}`)); loadRuns(); }

async function sendFeedback(contentId, declined) {
  await api(`/api/runs/${currentRun}/feedback`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content_id: contentId, declined }),
  });
  const y = window.scrollY;
  await showRun(currentRun);
  window.scrollTo(0, y);
}

// ---------- upload ----------
function showError(msg) { $("upload-error").textContent = msg; $("upload-error").hidden = !msg; }

async function submitRun(formData) {
  showError("");
  const btns = [$("rank-btn"), $("demo-btn"), $("demo-btn-2")];
  btns.forEach((b) => { b.disabled = true; });
  $("rank-btn").textContent = "Ranking…";
  try { const run = await api("/api/runs", { method: "POST", body: formData }); render(run); loadRuns(); }
  catch (err) { showError(err.message); }
  finally { btns.forEach((b) => { b.disabled = false; }); $("rank-btn").textContent = "Rank pages"; }
}

$("upload-form").addEventListener("submit", (e) => { e.preventDefault(); submitRun(new FormData(e.target)); });

async function runDemo() {
  const blob = await (await fetch("/demo_pages.csv")).blob();
  const fd = new FormData();
  fd.append("file", blob, "demo_pages.csv");
  fd.append("k", $("k").value); fd.append("per_client", $("per_client").value);
  submitRun(fd);
}
$("demo-btn").addEventListener("click", runDemo);
$("demo-btn-2").addEventListener("click", runDemo);

const drop = $("drop"), fileInput = $("file");
function showFile() {
  const f = fileInput.files[0];
  drop.classList.toggle("has-file", !!f);
  $("drop-text").innerHTML = f ? `<b>${esc(f.name)}</b> · ${(f.size / 1024).toFixed(0)} KB` : "<b>Choose a CSV</b> or drop it here";
}
fileInput.addEventListener("change", showFile);
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; showFile(); } });

// ---------- delete with an in-page confirm ----------
let armTimer = null;
function resetDelete() { clearTimeout(armTimer); $("delete-run").classList.remove("armed"); $("delete-run").textContent = "Delete run"; }
$("delete-run").addEventListener("click", async () => {
  const b = $("delete-run");
  if (!b.classList.contains("armed")) {
    b.classList.add("armed"); b.textContent = "Click again to delete";
    armTimer = setTimeout(resetDelete, 3000);
    return;
  }
  resetDelete();
  await api(`/api/runs/${currentRun}`, { method: "DELETE" });
  currentRun = null; $("run").hidden = true; $("empty").hidden = false;
  const runs = await loadRuns();
  if (runs.length) showRun(runs[0].id);
});

// ---------- start: open the latest run if there is one ----------
(async () => {
  loadHealth();
  const runs = await loadRuns();
  if (runs.length) showRun(runs[0].id);
})();
