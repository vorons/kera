// @ts-check
// Kera's local web page. Type-checked as TypeScript in CI (tsc --checkJs); no build step for users.
"use strict";

/**
 * @typedef {{ path: string, label: string, measured: boolean, read_gbps: number, note: string }} Drive
 * @typedef {{ name: string, backend: string }} Gpu
 * @typedef {{
 *   measured_at: string, chip: string, ram_gb: number, usable_gb: number,
 *   ram_gbps: { gpu: number, cpu: number, both: number },
 *   gpu_q4_gbps: number, cpu_q4_gbps: number, gpus: Gpu[], drives: Drive[]
 * }} Profile
 * @typedef {{
 *   path: string, file: string, vision?: boolean, image?: boolean, ollama?: string, size_gb?: number, moe?: boolean, fits?: boolean,
 *   ceiling_tok_s?: number, ceiling_overlap_tok_s?: number, lines?: string[], error?: string
 * }} Model
 * @typedef {{ state: string, model: string, file: string, seconds: number, error: string, url?: string }} Run
 * @typedef {{
 *   version: string, commit: string,
 *   profile: Profile | null, explain: string[],
 *   setup: { running: boolean, progress: number, step: string, error: string }
 * }} Status
 */

/** @param {string} id */
function el(id) {
  const e = document.getElementById(id);
  if (!e) throw new Error("missing element " + id);
  return e;
}

/** @param {number} v */
function gbps(v) {
  return v >= 10 ? v.toFixed(0) + " GB/s" : v.toFixed(1) + " GB/s";
}

/** @param {Status} s */
function render(s) {
  el("version").textContent = "Kera " + s.version + " (" + s.commit + ")";

  const progress = el("setup-progress");
  const machine = el("machine");
  const recheck = /** @type {HTMLButtonElement} */ (el("recheck"));
  const error = el("error");

  progress.hidden = !s.setup.running;
  if (s.setup.running) {
    const pct = Math.round(100 * s.setup.progress);
    el("setup-step").textContent = s.setup.step || "Measuring…";
    el("setup-bar").style.width = pct + "%";
    el("setup-bar").parentElement?.setAttribute("aria-valuenow", String(pct));
  }

  error.hidden = !s.setup.error;
  error.textContent = s.setup.error || "";

  recheck.hidden = s.setup.running;
  recheck.disabled = s.setup.running;

  if (!s.profile) {
    machine.hidden = true;
    return;
  }
  machine.hidden = s.setup.running;

  const lines = el("machine-lines");
  lines.replaceChildren(...s.explain.map((t) => {
    const li = document.createElement("li");
    li.textContent = t;
    return li;
  }));

  const p = s.profile;
  /** @type {[string, string][]} */
  const numbers = [];
  const ssd = p.drives.filter((d) => d.measured).sort((a, b) => b.read_gbps - a.read_gbps)[0];
  if (ssd) numbers.push(["SSD read, " + (ssd.label || ssd.path), gbps(ssd.read_gbps)]);
  if (p.ram_gbps.gpu > 0) numbers.push(["Graphics chip memory", gbps(p.ram_gbps.gpu)]);
  if (p.ram_gbps.cpu > 0) numbers.push(["CPU memory", gbps(p.ram_gbps.cpu)]);
  numbers.push(["Memory for models", p.usable_gb.toFixed(0) + " GB"]);

  el("machine-numbers").replaceChildren(...numbers.map(([k, v]) => {
    const div = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = k;
    dd.textContent = v;
    div.append(dt, dd);
    return div;
  }));

  el("measured-at").textContent = "Measured " + new Date(p.measured_at).toLocaleString() + ".";
}

/** @type {number | undefined} */
let timer;

async function refresh() {
  try {
    const r = await fetch("api/status", { cache: "no-store" });
    if (!r.ok) throw new Error("status " + r.status);
    /** @type {Status} */
    const s = await r.json();
    render(s);
    clearTimeout(timer);
    timer = window.setTimeout(refresh, s.setup.running ? 700 : 5000);
  } catch (e) {
    const err = el("error");
    err.hidden = false;
    err.textContent = "Kera isn't answering. If you closed its window, open Kera again.";
    clearTimeout(timer);
    timer = window.setTimeout(refresh, 3000);
  }
}

el("recheck").addEventListener("click", async () => {
  const b = /** @type {HTMLButtonElement} */ (el("recheck"));
  b.disabled = true;
  // the custom header keeps other web pages from starting a check (browsers can't send it cross-site)
  // (if Kera has stopped, the status poll says so)
  await fetch("api/setup", { method: "POST", headers: { "X-Kera": "1" } }).catch(() => undefined);
  refresh();
});

el("quit").addEventListener("click", async () => {
  await fetch("api/quit", { method: "POST", headers: { "X-Kera": "1" } }).catch(() => undefined);
  clearTimeout(timer);
  document.body.replaceChildren();
  const p = document.createElement("p");
  p.style.padding = "24px";
  p.textContent = "Kera has stopped. You can close this tab.";
  document.body.append(p);
});

/** @param {number} v */
function tokS(v) {
  return v >= 10 ? v.toFixed(0) : v.toFixed(1);
}

async function refreshModels() {
  try {
    const r = await fetch("api/models", { cache: "no-store" });
    if (!r.ok) return;
    /** @type {{ models: Model[] }} */
    const j = await r.json();
    el("models-empty").hidden = j.models.length > 0;
    el("models").replaceChildren(...j.models.map((m) => {
      const li = document.createElement("li");
      const head = document.createElement("div");
      head.className = "model-head";
      const name = document.createElement("strong");
      name.textContent = m.file;
      head.append(name);
      /** @type {[boolean, string, string][]} */
      const tags = [[!!m.ollama, "Ollama", "One of Ollama's models, used where it is"],
                    [!!m.vision, "sees images", "Send it pictures: it can describe and read them"],
                    [!!m.image, "makes images", "It makes pictures from text (OpenAI's images API)"]];
      for (const [on, text, title] of tags) {
        if (!on) continue;
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = text;
        tag.title = title;
        head.append(tag);
      }
      if (m.ceiling_tok_s !== undefined) {
        const speed = document.createElement("span");
        speed.className = "speed";
        speed.textContent = "up to " + tokS(m.ceiling_tok_s) + " words/s";
        head.append(speed);
      }
      if (!m.error) {
        const run = document.createElement("button");
        run.type = "button";
        run.className = "run-model";
        run.textContent = "Run";
        run.addEventListener("click", () => runModel(m.path));
        head.append(run);
      }
      li.append(head);
      const lines = m.error ? ["Kera can't read this file: " + m.error] : (m.lines || []);
      for (const line of lines) {
        const p = document.createElement("p");
        p.className = "small";
        p.textContent = line;
        li.append(p);
      }
      return li;
    }));
  } catch (e) {
    // the status poll reports a lost connection
  }
}

let starting = false;

/** @param {string} path */
async function runModel(path) {
  // one start at a time: a second click while the first is under way does nothing
  if (starting) return;
  starting = true;
  document.querySelectorAll(".run-model").forEach((b) => { /** @type {HTMLButtonElement} */ (b).disabled = true; });
  try {
    await fetch("api/run", {
      method: "POST",
      headers: { "X-Kera": "1", "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    }).catch(() => undefined);
  } finally {
    starting = false;
    document.querySelectorAll(".run-model").forEach((b) => { /** @type {HTMLButtonElement} */ (b).disabled = false; });
  }
  refreshRun();
}

/** @type {number | undefined} */
let runTimer;

async function refreshRun() {
  try {
    const r = await fetch("api/run", { cache: "no-store" });
    if (!r.ok) return;
    /** @type {Run} */
    const s = await r.json();
    el("run").hidden = s.state === "stopped" && !s.error;
    const state = el("run-state");
    if (s.state === "loading") {
      state.textContent = "Loading " + s.file + "… " + s.seconds.toFixed(0) + " s";
    } else if (s.state === "ready") {
      state.textContent = s.file + " is loaded and ready (it took " + s.seconds.toFixed(0) + " s).";
    } else if (s.state === "failed") {
      state.textContent = s.file + " stopped.";
    } else {
      state.textContent = "";
    }
    el("run-where").hidden = s.state !== "ready";
    el("run-url").textContent = s.url || "";
    const err = el("run-error");
    err.hidden = !s.error;
    err.textContent = s.error;
    clearTimeout(runTimer);
    runTimer = window.setTimeout(refreshRun, s.state === "loading" ? 700 : 4000);
  } catch (e) {
    clearTimeout(runTimer);
    runTimer = window.setTimeout(refreshRun, 4000);
  }
}

el("run-stop").addEventListener("click", async () => {
  await fetch("api/run/stop", { method: "POST", headers: { "X-Kera": "1" } }).catch(() => undefined);
  refreshRun();
});

el("add-folder").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const input = /** @type {HTMLInputElement} */ (el("folder"));
  const r = await fetch("api/models/folder", {
    method: "POST",
    headers: { "X-Kera": "1", "Content-Type": "application/json" },
    body: JSON.stringify({ folder: input.value.trim() }),
  }).catch(() => null);
  if (!r) return; // Kera has stopped: the status poll says so
  if (r.ok) {
    input.value = "";
    refreshModels();
  } else {
    input.setCustomValidity("That isn't a folder Kera can open.");
    input.reportValidity();
    input.setCustomValidity("");
  }
});

refresh();
refreshModels();
refreshRun();
