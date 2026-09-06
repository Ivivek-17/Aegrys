/* Aegrys dashboard.
 *
 * Runs in two modes from the same file:
 *   LOCAL  — the control API answered, so start/stop and log streaming work.
 *   DOCS   — served as a static site, no API. Control screens go read-only;
 *            setup, architecture and benchmarks remain fully usable.
 *
 * Mode is discovered by probing, never assumed, so the same build can be
 * deployed to a domain and run on localhost with no configuration.
 */
(() => {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  // null, not false: setMode() early-returns when the mode is unchanged, so
  // starting at `false` made the very first setMode(false) a no-op and the
  // docs-mode banner never appeared on a statically hosted build.
  let ONLINE = null;
  let statusTimer = null;
  let logStream = null;

  /* ------------------------------------------------------------- routing */

  const SCREENS = ["control", "monitor", "setup", "docs", "bench"];

  function show(name) {
    if (!SCREENS.includes(name)) name = "control";
    $$(".screen").forEach(s => s.classList.toggle("active", s.id === `screen-${name}`));
    $$("#nav button").forEach(b =>
      b.setAttribute("aria-selected", String(b.dataset.screen === name)));
    if (location.hash.slice(1) !== name) history.replaceState(null, "", `#${name}`);
    window.scrollTo(0, 0);
    document.title = `AEGRYS — ${name}`;
  }

  $$("#nav button").forEach(b =>
    b.addEventListener("click", () => show(b.dataset.screen)));
  $$("[data-goto]").forEach(a =>
    a.addEventListener("click", e => { e.preventDefault(); show(a.dataset.goto); }));
  window.addEventListener("hashchange", () => show(location.hash.slice(1)));

  // Number keys jump between screens — it is a terminal, after all.
  document.addEventListener("keydown", e => {
    if (e.target.matches("input, select, textarea") || e.metaKey || e.ctrlKey) return;
    const n = parseInt(e.key, 10);
    if (n >= 1 && n <= SCREENS.length) show(SCREENS[n - 1]);
  });

  /* -------------------------------------------------------------- copy */

  $$(".cmd").forEach(box => {
    const btn = document.createElement("button");
    btn.className = "copy";
    btn.textContent = "COPY";
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText($("pre", box).innerText.trim());
        btn.textContent = "COPIED";
      } catch { btn.textContent = "SELECT + ⌘C"; }
      setTimeout(() => (btn.textContent = "COPY"), 1400);
    });
    box.appendChild(btn);
  });

  /* ------------------------------------------------------------ api glue */

  async function api(path, opts) {
    const r = await fetch(path, {
      headers: { "Content-Type": "application/json" }, ...opts,
    });
    if (!r.ok) throw new Error(`${r.status}`);
    return r.json();
  }

  function setMode(online) {
    if (online === ONLINE) return;
    ONLINE = online;
    const pill = $("#modePill");
    pill.textContent = online ? "LOCAL · CONNECTED" : "DOCS MODE";
    pill.dataset.mode = online ? "LOCAL" : "DOCS";
    $("#footMode").textContent = online ? "local control active" : "docs mode";
    $("#offlineNotice").hidden = online;
    ["btnStart", "btnStop", "btnBackend"].forEach(id => {
      const el = $(`#${id}`);
      if (!online) el.disabled = true;
    });
    if (online) { startLogs(); }
  }

  /* ----------------------------------------------------------- status */

  function dot(on) { return `<span class="dot ${on ? "on" : "off"}">${on ? "●" : "○"}</span>`; }

  function fmtUptime(s) {
    if (s == null) return "—";
    const m = Math.floor(s / 60), sec = Math.floor(s % 60);
    return m ? `${m}m ${sec}s` : `${sec}s`;
  }

  async function pollStatus() {
    try {
      const s = await api("/api/status");
      setMode(true);

      const a = s.assistant, b = s.backend;
      $("#stAssistant").innerHTML =
        `${dot(a.running)} ${a.running ? `running · pid ${a.pid}` : "stopped"}`;
      $("#stBackend").innerHTML =
        `${dot(s.llm_alive)} ${s.llm_alive ? "ready" : (b.running ? "starting…" : "stopped")}`;
      $("#stUptime").textContent = fmtUptime(a.uptime_s);

      // Starting without the backend just makes the assistant exit with
      // "LLM backend unreachable", so gate on it and say why.
      $("#btnStart").disabled = a.running || !s.llm_alive;
      $("#btnStart").title = s.llm_alive ? "" : "Start the LLM backend first";
      $("#btnStart").textContent = s.llm_alive || a.running
        ? "▶ START ASSISTANT"
        : "▶ START — BACKEND NEEDED";
      $("#btnStop").disabled = !a.running;
      // Server is the source of truth, so a page reload restores the input box
      // rather than leaving it disabled while the assistant waits on stdin.
      const serverTextMode = !!a.text_mode;
      if (serverTextMode !== textMode) {
        textMode = serverTextMode;
        setPromptEnabled(textMode);
      }
      $("#btnBackend").disabled = false;
      $("#btnBackend").textContent = s.llm_alive ? "BACKEND READY" : "START BACKEND";
      $("#btnBackend").disabled = s.llm_alive;

      $("#pollDot").className = "dot on";
      $("#pollDot").textContent = "●";
    } catch {
      setMode(false);
      $("#pollDot").className = "dot off";
      $("#pollDot").textContent = "○";
    }
  }

  async function loadPreflight() {
    let data;
    try { data = await api("/api/preflight"); }
    catch { renderPreflightOffline(); return; }

    const box = $("#checks");
    box.innerHTML = "";
    for (const c of data.checks) {
      const el = document.createElement("div");
      el.className = `check ${c.ok ? "good" : "bad"}`;
      el.innerHTML =
        `<span class="mark">${c.ok ? "✓" : "✗"}</span>` +
        `<span class="name">${esc(c.label)}</span>` +
        `<span class="detail">${esc(c.detail)}</span>` +
        (c.ok || !c.fix ? "" : `<span class="fix">→ ${esc(c.fix)}</span>`);
      box.appendChild(el);
    }
    const n = data.checks.filter(c => c.ok).length;
    $("#preflightSummary").textContent = `${n}/${data.checks.length}`;
  }

  function renderPreflightOffline() {
    $("#checks").innerHTML =
      `<div class="check"><span class="mark">·</span>` +
      `<span class="name">unavailable in docs mode</span>` +
      `<span class="detail">run aegrys-dash locally</span></div>`;
    $("#preflightSummary").textContent = "";
  }

  /* ------------------------------------------------------------ actions */

  function runOpts() {
    return {
      mode: $("#optText").checked ? "text" : "voice",
      no_tools: $("#optNoTools").checked,
      no_barge_in: $("#optNoBarge").checked,
      stt_model: $("#optStt").value || null,
    };
  }

  let textMode = false;

  $("#btnStart").addEventListener("click", async () => {
    $("#btnStart").disabled = true;
    appendLine("[starting assistant…]", "warn");
    const opts = runOpts();
    try {
      const r = await api("/api/start", { method: "POST", body: JSON.stringify(opts) });
      textMode = !!r.text_mode;
    } catch { appendLine("[failed to start]", "warn"); }
    setPromptEnabled(textMode);
    show("monitor");
    if (textMode) $("#sendInput").focus();
    pollStatus();
  });

  function setPromptEnabled(on) {
    $("#sendInput").disabled = !on;
    $("#sendBtn").disabled = !on;
    $("#sendInput").placeholder = on
      ? "type a message and press enter…"
      : "text mode only — enable it on the CONTROL screen";
  }

  $("#sendForm").addEventListener("submit", async e => {
    e.preventDefault();
    const input = $("#sendInput");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    try {
      await api("/api/send", { method: "POST", body: JSON.stringify({ text }) });
    } catch {
      appendLine("[assistant is not accepting input]", "warn");
    }
  });

  $("#btnStop").addEventListener("click", async () => {
    $("#btnStop").disabled = true;
    try { await api("/api/stop", { method: "POST" }); } catch {}
    appendLine("[stopped]", "warn");
    pollStatus();
  });

  $("#btnBackend").addEventListener("click", async () => {
    $("#btnBackend").disabled = true;
    $("#btnBackend").textContent = "STARTING…";
    try { await api("/api/backend/start", { method: "POST" }); } catch {}
    setTimeout(() => { pollStatus(); loadPreflight(); }, 2500);
  });

  /* ------------------------------------------------------------ console */

  const consoleEl = $("#console");
  let cleared = false;

  function esc(s) {
    return String(s).replace(/[&<>"]/g, c =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  function classify(line) {
    if (/^\s*you:/.test(line)) return "you";
    if (/^\s*aegrys:/.test(line)) return "bot";
    if (/^\s*⚙/.test(line)) return "tool";
    if (/^\s*turn \d/.test(line)) return "trace";
    if (/^\s*(⏰|⨯|\[)/.test(line)) return "warn";
    return "";
  }

  function appendLine(text, cls) {
    if (!cleared) { consoleEl.innerHTML = ""; cleared = true; }
    const follow = $("#autoscroll").checked;
    const atBottom =
      consoleEl.scrollHeight - consoleEl.scrollTop - consoleEl.clientHeight < 40;
    const el = document.createElement("div");
    el.className = `ln ${cls ?? classify(text)}`;
    el.textContent = text;
    consoleEl.appendChild(el);
    while (consoleEl.childElementCount > 600) consoleEl.firstElementChild.remove();
    if (follow && atBottom) consoleEl.scrollTop = consoleEl.scrollHeight;
    harvest(text);
  }

  $("#btnClear").addEventListener("click", () => {
    consoleEl.innerHTML = `<span class="empty">cleared</span>`;
    cleared = false;
  });

  function startLogs() {
    if (logStream) return;
    try {
      logStream = new EventSource("/api/logs?source=assistant");
      logStream.onmessage = e => {
        try { appendLine(JSON.parse(e.data)); } catch {}
      };
      logStream.onerror = () => { logStream.close(); logStream = null; };
    } catch {}
  }

  /* -------------------------------------------------------- turn metrics */

  const turns = [];
  let bargeIns = 0;

  function harvest(line) {
    if (/⨯ barge-in/.test(line)) {
      bargeIns++;
      $("#mBarge").textContent = bargeIns;
      return;
    }
    const m = line.match(/^\s*turn (\d+)/);
    if (!m) return;
    const num = (re) => { const x = line.match(re); return x ? +x[1] : null; };
    turns.push({
      n: +m[1],
      ttfa: num(/TTFA (\d+)ms/),
      stt: num(/stt (\d+)ms/),
      tok1: num(/llm_tok1 (\d+)ms/),
      tool: (line.match(/tool=(\w+)/) || [])[1] || null,
      text: (line.match(/text=(.+?)(?: \| |$)/) || [])[1] || "",
    });
    renderTurns();
  }

  function renderTurns() {
    $("#mTurns").textContent = turns.length;
    const ttfas = turns.map(t => t.ttfa).filter(Number.isFinite).sort((a, b) => a - b);
    $("#mTtfa").innerHTML = ttfas.length
      ? `${ttfas[Math.floor(ttfas.length / 2)]}<span class="unit">ms</span>`
      : `—<span class="unit">ms</span>`;

    const rows = turns.slice(-12).reverse();
    $("#turnRows").innerHTML = rows.length ? rows.map(t => `
      <tr>
        <td class="k">${t.n}</td>
        <td>${t.ttfa != null ? t.ttfa + " ms" : "—"}</td>
        <td>${t.stt != null ? t.stt + " ms" : "—"}</td>
        <td>${t.tok1 != null ? t.tok1 + " ms" : "—"}</td>
        <td>${t.tool ? esc(t.tool) : "—"}</td>
        <td>${esc(t.text).slice(0, 46)}</td>
      </tr>`).join("")
      : `<tr><td colspan="6" style="color:var(--dimmer)">no turns yet</td></tr>`;
  }

  /* --------------------------------------------------------------- boot */

  show(location.hash.slice(1) || "control");
  pollStatus();
  loadPreflight();
  statusTimer = setInterval(pollStatus, 2000);
  setInterval(() => { if (ONLINE) loadPreflight(); }, 15000);
})();
