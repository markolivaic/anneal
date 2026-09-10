/*
 * Wiring: catalogue -> record column -> edit -> streamed relaxation.
 *
 * The relaxation arrives as Server-Sent Events, one per optimizer step, and is
 * applied to the viewer and the trace as it comes. Nothing waits for the run to
 * finish. The run IS the animation.
 */

import { LatticeViewer, currentThemeIsDark } from "./viewer.js";
import { ForceTrace } from "./trace.js";
import { elementColor, assertPaletteDiscipline } from "./elements.js";

const API = "/api";

const state = {
  sessionId: null,
  structure: null,
  meta: null,
  selected: null,
  baselineEnergy: null,
  relaxing: false,
  abort: null,
};

const viewer = new LatticeViewer(document.getElementById("viewport"));
const trace = new ForceTrace(document.getElementById("trace-canvas"));

viewer.onPick = (index) => selectSite(index);

// The palette rule is checked at boot rather than trusted. If element chroma
// ever creeps up toward the accent, this says so out loud.
const paletteProblems = assertPaletteDiscipline();
if (paletteProblems.length) {
  console.error("palette discipline broken:", paletteProblems);
}

// ------------------------------------------------------------------- helpers

async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail;
    try {
      detail = (await response.json()).detail;
    } catch {
      detail = await response.text();
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json();
}

const fmt = (x, places = 4) =>
  x === null || x === undefined ? "--" : Number(x).toFixed(places);

// -------------------------------------------------------------------- record

function renderRecord() {
  const body = document.getElementById("record-body");
  if (!state.structure) return;

  const s = state.structure;
  const lattice = s.lattice;
  const elements = [...new Set(s.sites.map((site) => site.element))];

  body.innerHTML = `
    <div class="entry-head">
      <span class="cod-ref">COD <a href="${state.sourceUrl}" target="_blank"
        rel="noopener">${state.codId}</a>, CC0</span>
      <p class="formula">${s.formula_reduced}</p>
      <div class="hm">${state.spacegroup ?? ""}</div>
    </div>

    <h2 class="rule-heading">cell</h2>
    <div class="constants">
      <div><span class="tag">_cell_length_a</span><span class="val">${fmt(lattice.a, 4)}</span></div>
      <div><span class="tag">_cell_angle_alpha</span><span class="val">${fmt(lattice.alpha, 2)}</span></div>
      <div><span class="tag">_cell_length_b</span><span class="val">${fmt(lattice.b, 4)}</span></div>
      <div><span class="tag">_cell_angle_beta</span><span class="val">${fmt(lattice.beta, 2)}</span></div>
      <div><span class="tag">_cell_length_c</span><span class="val">${fmt(lattice.c, 4)}</span></div>
      <div><span class="tag">_cell_angle_gamma</span><span class="val">${fmt(lattice.gamma, 2)}</span></div>
      <div><span class="tag">_cell_volume</span><span class="val">${fmt(s.volume_a3, 2)}</span></div>
      <div><span class="tag">_sites</span><span class="val">${s.n_sites}</span></div>
    </div>

    <h2 class="rule-heading">atom sites <span class="count">${s.n_sites}</span></h2>
    <div class="sites-scroll">
    <table class="sites">
      <thead>
        <tr><th>label</th><th>x</th><th>y</th><th>z</th></tr>
      </thead>
      <tbody id="site-rows">
        ${s.sites
          .map(
            (site, i) => `
          <tr data-index="${i}">
            <td><span class="swatch" style="background:${elementColor(
              site.element,
              currentThemeIsDark()
            )}"></span>${site.element}${i + 1}</td>
            <td>${site.frac[0].toFixed(4)}</td>
            <td>${site.frac[1].toFixed(4)}</td>
            <td>${site.frac[2].toFixed(4)}</td>
          </tr>`
          )
          .join("")}
      </tbody>
    </table>
    </div>

    <h2 class="rule-heading">edit</h2>
    <div class="edit-block" id="edit-block">
      <p class="hint" id="edit-hint">Select a site in the table or the cell.</p>
      <div class="edit-row">
        <select id="substitute-element">
          ${PERIODIC.map((e) => `<option value="${e}">${e}</option>`).join("")}
        </select>
        <button id="do-substitute" disabled>substitute</button>
        <button id="do-vacancy" disabled>vacancy</button>
      </div>
      <div class="edit-row">
        <input type="range" id="strain-slider" min="-8" max="8" value="0" step="0.5" />
        <span style="font-family:var(--mono);font-size:11px;min-width:46px"
          id="strain-value">0.0%</span>
        <button id="do-strain">strain</button>
      </div>
      <div class="edit-row">
        <button id="do-relax" class="primary">relax</button>
        <button id="do-stop" disabled>stop</button>
        <button id="do-reset">reset</button>
      </div>
      <div id="edit-error"></div>
      <div class="history" id="history"></div>
    </div>

    <h2 class="rule-heading">catalogue</h2>
    <div class="edit-row">
      <input type="text" id="catalog-search" placeholder="formula, COD id, symbol"
        style="flex:1" />
    </div>
    <div class="catalog-list" id="catalog-list"></div>
  `;

  document.getElementById("legend").innerHTML = elements
    .map(
      (e) =>
        `<span><i class="swatch" style="background:${elementColor(
          e,
          currentThemeIsDark()
        )}"></i>${e}</span>`
    )
    .join("");

  bindRecordEvents();
  renderHistory();
}

const PERIODIC = [
  "H","Li","Be","B","C","N","O","F","Na","Mg","Al","Si","P","S","Cl","K","Ca",
  "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br",
  "Rb","Sr","Y","Zr","Nb","Mo","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I",
  "Cs","Ba","La","Ce","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Yb","Lu","Hf",
  "Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Th","U",
];

function bindRecordEvents() {
  document.querySelectorAll("#site-rows tr").forEach((row) => {
    row.addEventListener("click", () => selectSite(Number(row.dataset.index)));
  });

  document.getElementById("do-substitute").onclick = () =>
    mutate({
      kind: "substitute",
      site_index: state.selected,
      element: document.getElementById("substitute-element").value,
    });

  document.getElementById("do-vacancy").onclick = () =>
    mutate({ kind: "vacancy", site_index: state.selected });

  const slider = document.getElementById("strain-slider");
  slider.oninput = () => {
    document.getElementById("strain-value").textContent =
      `${Number(slider.value).toFixed(1)}%`;
  };
  document.getElementById("do-strain").onclick = () =>
    mutate({ kind: "strain", strain: Number(slider.value) / 100 });

  document.getElementById("do-relax").onclick = startRelax;
  document.getElementById("do-stop").onclick = stopRelax;
  document.getElementById("do-reset").onclick = resetStructure;

  const search = document.getElementById("catalog-search");
  search.oninput = debounce(() => loadCatalog(search.value), 220);
  loadCatalog("");
}

function selectSite(index) {
  state.selected = index;
  viewer.setSelected(index);
  document.querySelectorAll("#site-rows tr").forEach((row) => {
    row.classList.toggle("selected", Number(row.dataset.index) === index);
  });
  const site = state.structure.sites[index];
  document.getElementById("edit-hint").textContent =
    `Site ${index + 1}: ${site.element} at (${site.frac.map((v) => v.toFixed(3)).join(", ")})`;
  document.getElementById("do-substitute").disabled = false;
  document.getElementById("do-vacancy").disabled = false;
}

function renderHistory() {
  const box = document.getElementById("history");
  if (!box) return;
  box.innerHTML = (state.history ?? [])
    .map((h) => `<div>${h.kind}: ${h.detail}</div>`)
    .join("");
}

function showError(message) {
  const box = document.getElementById("edit-error");
  if (box) box.innerHTML = message ? `<div class="error">${message}</div>` : "";
}

// ------------------------------------------------------------------ actions

async function loadStructure(codId) {
  showError("");
  try {
    const data = await api("/load", {
      method: "POST",
      body: JSON.stringify({ cod_id: codId }),
    });
    state.sessionId = data.session_id;
    state.structure = data.structure;
    state.codId = data.cod_id;
    state.sourceUrl = data.source_url;
    state.estimate = data.estimate;
    state.history = [];
    state.selected = null;
    state.baselineEnergy = null;
    viewer.clearEdited();
    viewer.render(data.structure);
    trace.reset();
    renderRecord();
    showEstimate(data.estimate);
    document.getElementById("readout").hidden = true;
  } catch (error) {
    showError(`could not load ${codId}: ${error.message}`);
  }
}

async function mutate(request) {
  showError("");
  try {
    const data = await api("/mutate", {
      method: "POST",
      body: JSON.stringify({ session_id: state.sessionId, ...request }),
    });
    state.structure = data.structure;
    state.history = data.history;
    state.suggestedRelaxCell = data.mutation.suggested_relax_cell;
    // Only a substitution leaves an edited atom to mark. After a vacancy the
    // edited atom is gone and every index above it has shifted down, so marking
    // the index would paint an untouched neighbour in the colour reserved for
    // model output -- the one thing the palette rule must never do.
    if (request.kind === "substitute" && request.site_index != null) {
      viewer.markEdited(request.site_index);
    } else {
      viewer.clearEdited();
    }

    viewer.render(data.structure, { keepCamera: true });
    renderRecord();
    if (state.selected != null && state.selected < data.structure.n_sites) {
      selectSite(state.selected);
    }
    showEnergy(data.energy.energy_per_atom_ev);
    showEstimate(data.estimate);
    trace.reset();
  } catch (error) {
    showError(error.message);
  }
}

async function resetStructure() {
  const data = await api(`/reset?session_id=${state.sessionId}`, { method: "POST" });
  state.structure = data.structure;
  state.history = [];
  state.baselineEnergy = null;
  viewer.clearEdited();
  viewer.render(data.structure, { keepCamera: true });
  renderRecord();
  trace.reset();
  document.getElementById("readout").hidden = true;
}

function showEnergy(perAtom) {
  const readout = document.getElementById("readout");
  readout.hidden = false;
  document.getElementById("energy-value").textContent = `${fmt(perAtom, 4)} eV`;
  if (state.baselineEnergy === null) {
    state.baselineEnergy = perAtom;
    document.getElementById("energy-delta").textContent = "";
  } else {
    const delta = perAtom - state.baselineEnergy;
    document.getElementById("energy-delta").textContent =
      `${delta >= 0 ? "+" : ""}${fmt(delta, 4)} eV/atom vs start`;
  }
}

function showEstimate(estimate) {
  const box = document.getElementById("relax-estimate");
  if (!estimate) {
    box.textContent = "no benchmark on this machine, run scripts/benchmark.py";
    return;
  }
  // Cost is driven by graph edges, not atom count, so both are shown. The
  // headline figure is the median relaxation; the bracket is the honest spread.
  box.textContent =
    `${estimate.atoms} atoms, ${estimate.atom_graph_edges.toLocaleString()} edges` +
    `, typically ~${estimate.seconds_typical} s` +
    ` (${estimate.seconds_low}-${estimate.seconds_high} s)`;
}

// -------------------------------------------------------- streamed relaxation

async function startRelax() {
  if (state.relaxing) return;
  state.relaxing = true;
  showError("");
  trace.reset();
  document.getElementById("do-relax").disabled = true;
  document.getElementById("do-stop").disabled = false;
  setStatus("relaxing…");

  const controller = new AbortController();
  state.abort = controller;

  try {
    const response = await fetch(`${API}/relax`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        relax_cell: state.suggestedRelaxCell ?? null,
      }),
      signal: controller.signal,
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      let split;
      while ((split = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        handleFrame(frame);
      }
    }
  } catch (error) {
    if (error.name !== "AbortError") showError(`relaxation failed: ${error.message}`);
  } finally {
    state.relaxing = false;
    state.abort = null;
    document.getElementById("do-relax").disabled = false;
    document.getElementById("do-stop").disabled = true;
  }
}

function handleFrame(frame) {
  const eventLine = frame.split("\n").find((l) => l.startsWith("event: "));
  const dataLine = frame.split("\n").find((l) => l.startsWith("data: "));
  if (!eventLine || !dataLine) return;

  const kind = eventLine.slice(7).trim();
  const payload = JSON.parse(dataLine.slice(6));

  if (kind === "step") {
    // The strip opens on the first real step, never before there is a line to
    // draw in it.
    document.getElementById("trace-strip").classList.remove("is-idle");
    trace.push(payload);
    viewer.updatePositions(payload.positions, payload.lattice);
    // The arrows are the accent language made literal: what CHGNet is pushing
    // on, drawn on top of the deposited structure while it happens.
    viewer.showForces(payload.forces, payload.positions);
    showEnergy(payload.energy_per_atom_ev);
    setStatus(
      `step ${payload.step}, max force ${payload.max_force_ev_per_a.toFixed(3)} eV/Å`
    );
  } else if (kind === "done") {
    state.structure = payload.structure;
    viewer.render(payload.structure, { keepCamera: true });
    renderRecord();
    setStatus(payload.summary);
  } else if (kind === "error") {
    showError(payload.message);
    setStatus("failed");
  }
}

async function stopRelax() {
  document.getElementById("do-stop").disabled = true;
  setStatus("stopping…");
  await api(`/relax/stop?session_id=${state.sessionId}`, { method: "POST" });
}

function setStatus(text) {
  document.getElementById("relax-status").textContent = text;
}

// ----------------------------------------------------------------- catalogue

async function loadCatalog(query) {
  const list = document.getElementById("catalog-list");
  if (!list) return;
  const data = await api(`/catalog?q=${encodeURIComponent(query)}&limit=40`);
  list.innerHTML = data.items
    .map(
      (item) => `
      <button data-cod="${item.cod_id}">
        ${item.formula_reduced}
        <span class="sg">${item.spacegroup_symbol} / ${item.n_sites}</span>
      </button>`
    )
    .join("");
  list.querySelectorAll("button").forEach((button) => {
    button.onclick = () => loadStructure(Number(button.dataset.cod));
  });
}

function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

// --------------------------------------------------------------------- boot

document.getElementById("theme-toggle").onclick = () => {
  const dark = !currentThemeIsDark();
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  viewer.setTheme(dark);
  trace.draw();
  if (state.structure) renderRecord();
};

(async function boot() {
  state.meta = await api("/meta");
  const list = document.getElementById("catalog-list");
  if (list) loadCatalog("");

  // Open on something worth looking at rather than an empty viewport.
  await bootstrapCatalog();
})();

async function bootstrapCatalog() {
  const data = await api("/catalog?limit=1");
  if (data.items.length) {
    await loadStructure(data.items[0].cod_id);
  } else {
    document.getElementById("record-body").innerHTML =
      `<div class="empty-state">No catalogue on disk. Run
       <code>python scripts/build_catalog.py</code>.</div>`;
  }
}
