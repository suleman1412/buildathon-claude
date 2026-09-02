// ---------- Icons (inline SVG, no external deps) ----------

const ICONS = {
  runs: '<path d="M9 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h4"/><path d="M9 3h10a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H9"/><path d="M9 3v18"/><path d="M13 8l3 4-3 4"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
  dollar: '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
  check: '<path d="M20 6L9 17l-5-5"/>',
  x: '<path d="M18 6L6 18"/><path d="M6 6l12 12"/>',
  repo: '<path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/>',
  db: '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
  cpu: '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',
  moon: '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
  refresh: '<path d="M21 2v6h-6"/><path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M3 22v-6h6"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/>',
  inbox: '<path d="M22 12h-6l-2 3h-4l-2-3H2"/><path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>',
  chevron: '<path d="M9 18l6-6-6-6"/>',
};

function icon(name, cls) {
  return `<svg class="${cls || ""}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[name]}</svg>`;
}

// ---------- Theme ----------

function initTheme() {
  const stored = localStorage.getItem("theme");
  if (stored) document.documentElement.setAttribute("data-theme", stored);
  updateThemeIcon();
}

function updateThemeIcon() {
  const stored = localStorage.getItem("theme");
  const isDark = stored ? stored === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.getElementById("theme-icon").innerHTML = isDark ? ICONS.sun : ICONS.moon;
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  const isDark = current ? current === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
  const next = isDark ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
  updateThemeIcon();
}

// ---------- Data + formatting ----------

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
  const v = Number(n);
  // compact, but keep meaningful precision for sub-1 metrics (e.g. F1 scores)
  if (Math.abs(v) > 0 && Math.abs(v) < 1) return v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  return v.toFixed(2).replace(/\.00$/, "");
}

function fmtMoney(n) {
  if (n === null || n === undefined) return "$0.0000";
  return "$" + Number(n).toFixed(4);
}

function fmtTime(iso) {
  try {
    const d = new Date(iso);
    const now = new Date();
    const diffMs = now - d;
    const diffMin = diffMs / 60000;
    if (diffMin < 60) return `${Math.max(1, Math.round(diffMin))}m ago`;
    if (diffMin < 60 * 24) return `${Math.round(diffMin / 60)}h ago`;
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) + " · " +
           d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  } catch {
    return iso;
  }
}

function repoName(url) {
  if (!url) return null;
  const m = url.match(/github\.com\/([^/]+\/[^/]+?)(\.git)?\/?$/);
  return m ? m[1] : url;
}

const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// animates a number from 0 -> target with ease-out, formatting each frame
function animateValue(el, target, format, duration = 800) {
  if (prefersReducedMotion() || !(target > 0)) {
    el.textContent = format(target);
    return;
  }
  const start = performance.now();
  function tick(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = format(target * eased);
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ---------- Stats ----------

function renderStats(runs) {
  const total = runs.length;
  const passed = runs.filter((r) => r.passed).length;
  const totalSpend = runs.reduce((sum, r) => sum + (r.spend_usd || 0), 0);
  const passRate = total > 0 ? Math.round((passed / total) * 100) : 0;

  document.getElementById("stats").innerHTML = `
    <div class="stat">
      <div>
        <div class="stat-label">Total runs</div>
        <div class="stat-value" id="stat-total">0</div>
        <div class="stat-sub">${new Set(runs.map((r) => r.paper_title)).size} paper${runs.length === 1 ? "" : "s"} tested</div>
      </div>
      <div class="stat-icon tone-neutral">${icon("runs")}</div>
    </div>
    <div class="stat">
      <div>
        <div class="stat-label">Pass rate</div>
        <div class="stat-value" id="stat-passrate">0%</div>
        <div class="stat-sub">${passed} of ${total || 0} verified</div>
      </div>
      <div class="stat-icon tone-good">${icon("target")}</div>
    </div>
    <div class="stat">
      <div>
        <div class="stat-label">Est. spend</div>
        <div class="stat-value" id="stat-spend">$0.0000</div>
        <div class="stat-sub">across all runs</div>
      </div>
      <div class="stat-icon tone-accent">${icon("dollar")}</div>
    </div>
  `;

  animateValue(document.getElementById("stat-total"), total, (v) => Math.round(v).toString());
  animateValue(document.getElementById("stat-passrate"), passRate, (v) => Math.round(v) + "%");
  animateValue(document.getElementById("stat-spend"), totalSpend, (v) => fmtMoney(v));
}

// ---------- Bullet comparison ----------

function renderCompare(r) {
  const claimed = r.claimed_metric_value;
  const reproduced = r.reproduced_value;
  const hasRepro = reproduced !== null && reproduced !== undefined;
  const status = r.passed ? "pass" : hasRepro ? "fail" : "na";
  const unit = r.claimed_metric_unit || "";

  const max = Math.max(claimed || 0, reproduced || 0) * 1.15 || 1;
  const claimedPct = Math.min(100, ((claimed || 0) / max) * 100);
  const reproPct = hasRepro ? Math.min(100, (reproduced / max) * 100) : 0;

  const delta = hasRepro && claimed
    ? `${(((reproduced - claimed) / claimed) * 100).toFixed(1)}%`
    : null;

  return `
    <div class="compare">
      <div class="compare-label-row">
        <span class="compare-metric">${r.claimed_metric_name || "metric"}</span>
        ${delta
          ? `<span class="compare-delta ${status}">${reproduced >= claimed ? "+" : ""}${delta} vs. claim</span>`
          : `<span class="compare-delta na">no reproduced value</span>`}
      </div>
      <div class="bullet">
        ${hasRepro ? `<div class="bullet-fill ${status}" data-target-width="${reproPct}" style="width:0%"></div>` : ""}
        ${hasRepro
          ? `<span class="bullet-value" style="left:${Math.min(reproPct, 88)}%; ${reproPct > 88 ? "color:#fff;transform:translate(-105%,-50%)" : "transform:translate(calc(100% + 8px),-50%)"}">${fmtNum(reproduced)}${unit}</span>`
          : `<span class="bullet-value" style="left:8px; color:var(--text-muted)">N/A</span>`}
        <div class="bullet-tick" style="left:calc(${claimedPct}% - 1px)"></div>
        <span class="bullet-tick-label" style="left:${claimedPct}%">${fmtNum(claimed)}${unit}</span>
      </div>
      <div class="compare-legend">
        <span><span class="swatch claim"></span>Claimed</span>
        <span><span class="swatch repro-${status}"></span>Reproduced</span>
      </div>
    </div>
  `;
}

// ---------- Run card ----------

function renderRun(r) {
  const status = r.passed ? "pass" : "fail";
  const repo = repoName(r.github_repo);

  const tags = [
    r.github_repo
      ? `<span class="tag">${icon("repo")}<a href="${r.github_repo}" target="_blank" rel="noopener">${repo}</a></span>`
      : "",
    r.dataset ? `<span class="tag">${icon("db")}${r.dataset}</span>` : "",
    r.model ? `<span class="tag">${icon("cpu")}${r.model}</span>` : (r.backend ? `<span class="tag">${icon("cpu")}${r.backend}</span>` : ""),
  ].filter(Boolean).join("");

  return `
    <div class="run">
      <div class="run-head">
        <div>
          <div class="run-title">${r.paper_title || "Untitled"}</div>
          <div class="run-tags">
            ${tags}
            <span class="run-time">${icon("clock", "")} ${fmtTime(r.timestamp)}</span>
          </div>
        </div>
        <span class="status-pill ${status}">${icon(status === "pass" ? "check" : "x")}${status === "pass" ? "Pass" : "Fail"}</span>
      </div>
      ${renderCompare(r)}
      <details class="reason">
        <summary>
          ${icon("chevron", "chevron")}
          <span>Diagnostic details</span>
          <span class="spend-badge">spend: ${fmtMoney(r.spend_usd)}</span>
        </summary>
        <pre>${(r.reason || "").replace(/</g, "&lt;")}</pre>
      </details>
    </div>
  `;
}

// ---------- Filtering ----------

let currentFilter = "all";
let allRuns = [];

function applyFilter() {
  const filtered = allRuns.filter((r) => {
    if (currentFilter === "pass") return r.passed;
    if (currentFilter === "fail") return !r.passed;
    return true;
  });

  const container = document.getElementById("runs");
  document.getElementById("result-count").textContent =
    `${filtered.length} of ${allRuns.length} run${allRuns.length === 1 ? "" : "s"}`;

  if (filtered.length === 0) {
    container.innerHTML = `<div class="empty">No ${currentFilter === "all" ? "" : currentFilter + " "}runs to show.</div>`;
    return;
  }

  container.innerHTML = filtered.map((r, i) => {
    const html = renderRun(r);
    return html.replace('class="run"', `class="run" style="animation-delay:${Math.min(i * 60, 400)}ms"`);
  }).join("");

  // bullet fills start at width:0 in markup so the CSS transition has a
  // state change to animate from -- flip to the real width one frame later
  // (double rAF: the first frame just commits the 0% layout so the browser
  // doesn't collapse the two writes into a single, un-transitioned jump)
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      container.querySelectorAll(".bullet-fill[data-target-width]").forEach((el) => {
        el.style.width = el.dataset.targetWidth + "%";
      });
    });
  });
}

function setupFilterBar() {
  document.getElementById("filter-segmented").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-filter]");
    if (!btn) return;
    currentFilter = btn.dataset.filter;
    document.querySelectorAll("#filter-segmented button").forEach((b) => b.classList.toggle("active", b === btn));
    applyFilter();
  });
}

// ---------- Boot ----------

async function render() {
  allRuns = (await loadRuns()).slice().reverse(); // newest first
  renderStats(allRuns);

  if (allRuns.length === 0) {
    document.querySelector(".filter-bar").style.display = "none";
    document.getElementById("runs").innerHTML = `
      <div class="empty">
        ${icon("inbox", "empty-icon")}
        <div>No runs yet.</div>
        <code>python3 validate.py paper.pdf</code>
      </div>`;
    return;
  }

  applyFilter();
}

initTheme();
setupFilterBar();
document.getElementById("theme-toggle").addEventListener("click", toggleTheme);
document.getElementById("refresh-btn").addEventListener("click", render);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (!localStorage.getItem("theme")) updateThemeIcon();
});
render();
