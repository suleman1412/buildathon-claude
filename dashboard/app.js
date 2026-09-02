async function loadRuns() {
  const res = await fetch("data/runs.json", { cache: "no-store" });
  if (!res.ok) return [];
  try {
    return await res.json();
  } catch {
    return [];
  }
}

function fmtNum(n) {
  if (n === null || n === undefined) return "N/A";
  return Number(n).toFixed(2);
}

function fmtMoney(n) {
  if (n === null || n === undefined) return "$0.00";
  return "$" + Number(n).toFixed(4);
}

function fmtTime(iso) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function renderStats(runs) {
  const total = runs.length;
  const passed = runs.filter((r) => r.passed).length;
  const totalSpend = runs.reduce((sum, r) => sum + (r.spend_usd || 0), 0);

  document.getElementById("stats").innerHTML = `
    <div class="stat">
      <div class="label">Total Runs</div>
      <div class="value">${total}</div>
    </div>
    <div class="stat">
      <div class="label">Passed</div>
      <div class="value" style="color: var(--pass)">${passed}/${total || 0}</div>
    </div>
    <div class="stat">
      <div class="label">Est. Spend</div>
      <div class="value">${fmtMoney(totalSpend)}</div>
    </div>
  `;
}

function barRow(label, value, max, cls) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return `
    <div class="bar-row">
      <div>${label}</div>
      <div class="bar-track"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
      <div>${fmtNum(value)}</div>
    </div>
  `;
}

function renderRun(r) {
  const status = r.passed ? "pass" : "fail";
  const claimed = r.claimed_metric_value || 0;
  const reproduced = r.reproduced_value;
  const max = Math.max(claimed, reproduced || 0, 100);

  const bars = `
    <div class="bars">
      ${barRow("Claimed", claimed, max, "claimed")}
      ${reproduced !== null && reproduced !== undefined
        ? barRow("Reproduced", reproduced, max, `reproduced ${status}`)
        : `<div class="bar-row"><div>Reproduced</div><div class="bar-track"></div><div>N/A</div></div>`}
    </div>
  `;

  return `
    <div class="run">
      <div class="run-head">
        <div>
          <div class="run-title">${r.paper_title || "Untitled"}</div>
          <div class="run-meta">
            ${r.github_repo ? `<a href="${r.github_repo}" target="_blank" style="color:var(--accent)">${r.github_repo}</a> · ` : ""}
            ${r.dataset ? `${r.dataset} · ` : ""}
            ${r.claimed_metric_name || "metric"} ·
            ${r.backend || "?"}${r.model ? ` (${r.model})` : ""} ·
            ${fmtTime(r.timestamp)}
          </div>
        </div>
        <span class="badge ${status}">${status === "pass" ? "PASS" : "FAIL"}</span>
      </div>
      ${bars}
      <details class="reason">
        <summary>Details / spend: ${fmtMoney(r.spend_usd)}</summary>
        <pre>${(r.reason || "").replace(/</g, "&lt;")}</pre>
      </details>
    </div>
  `;
}

async function render() {
  const runs = await loadRuns();
  renderStats(runs);

  const container = document.getElementById("runs");
  if (runs.length === 0) {
    container.innerHTML = `<div class="empty">No runs yet. Run <code>python3 validate.py paper.pdf</code> on the compute machine, then redeploy.</div>`;
    return;
  }

  const sorted = [...runs].reverse(); // newest first
  container.innerHTML = sorted.map(renderRun).join("");
}

render();
