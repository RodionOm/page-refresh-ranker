// Single-page UI: talks only to the JSON API under /api.
const $ = (id) => document.getElementById(id);
let currentRun = null;

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

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

async function loadHealth() {
  try {
    const h = await api("/api/health");
    const m = h.model || {};
    $("model-badge").textContent = !h.model_loaded ? "no model: run train.py"
      : m.trained_on === "synthetic" ? "demo model · synthetic data"
      : `model · AUC ${m.auc ?? "?"} on unseen clients`;
  } catch { $("model-badge").textContent = "API not reachable"; }
}

async function loadRuns() {
  const runs = await api("/api/runs");
  $("runs").innerHTML = runs.length ? runs.map((r) => `
    <li><button data-id="${r.id}" class="${currentRun === r.id ? "active" : ""}">
      #${r.id} · ${esc(r.filename)}
      <small>${r.n_pages} pages · top ${r.k} · reviewed ${r.reviewed}</small>
    </button></li>`).join("") : '<li class="muted">No runs yet.</li>';
  $("runs").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => showRun(+b.dataset.id)));
}

function tile(value, label) { return `<div class="tile"><b>${value}</b><span>${label}</span></div>`; }
const pct = (x) => (x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`);

function render(run) {
  currentRun = run.id;
  $("empty").hidden = true; $("run").hidden = false;
  $("run-title").textContent = `Run #${run.id} · ${run.filename}`;
  const s = run.summary;
  $("tiles").innerHTML = tile(run.n_pages, "pages uploaded") + tile(s.queue_size, `in queue (max ${run.per_client} per client)`)
    + tile(`${s.reviewed}`, "reviewed so far") + tile(pct(s.precision_from_feedback), "of reviewed really declined")
    + (s.precision_observed !== null ? tile(pct(s.precision_observed), "declined (from the file's outcome column)") : "");
  $("rows").innerHTML = run.results.map((r) => `
    <tr>
      <td class="num">${r.rank}</td>
      <td>${esc(r.content_id)}</td>
      <td>${esc(r.client_id)}</td>
      <td class="num">${r.score.toFixed(3)} <span class="chip ${r.band}">${r.band}</span>
          <div class="bar"><i style="width:${Math.round(r.score * 100)}%"></i></div></td>
      <td>${r.reasons.map(esc).join("<br>") || '<span class="muted">—</span>'}</td>
      <td class="flags">${r.review_flags.map(esc).join("<br>")}</td>
      <td class="fb">
        <button data-c="${esc(r.content_id)}" data-v="1" class="${r.feedback === 1 ? "on" : ""}">declined</button>
        <button data-c="${esc(r.content_id)}" data-v="0" class="${r.feedback === 0 ? "on" : ""}">held up</button>
      </td>
    </tr>`).join("");
  $("rows").querySelectorAll(".fb button").forEach((b) => b.addEventListener("click", () => sendFeedback(b.dataset.c, b.dataset.v === "1")));
}

async function showRun(id) { render(await api(`/api/runs/${id}`)); loadRuns(); }

async function sendFeedback(contentId, declined) {
  await api(`/api/runs/${currentRun}/feedback`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content_id: contentId, declined }),
  });
  showRun(currentRun);
}

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("upload-error").hidden = true;
  try {
    const run = await api("/api/runs", { method: "POST", body: new FormData(e.target) });
    render(run); loadRuns();
  } catch (err) { $("upload-error").textContent = err.message; $("upload-error").hidden = false; }
});

$("delete-run").addEventListener("click", async () => {
  await api(`/api/runs/${currentRun}`, { method: "DELETE" });
  currentRun = null; $("run").hidden = true; $("empty").hidden = false; loadRuns();
});

loadHealth(); loadRuns();
