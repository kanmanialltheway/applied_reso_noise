// Library view: everything stored in the database, with downloads and deletion.
import { api } from "./api.js";
import { icon } from "./icons.js";
import { $, $$, download, esc, fmtDate, fmtDb, fmtSize, fmtTime, toast } from "./ui.js";

export async function renderLibrary(root, { open, openResult, classes }) {
  root.innerHTML = `<div class="studio-head"><h1>Library</h1><p>Every recording, analysis and processed file is stored in the local database, so you can come back to it later.</p></div><div id="lib-body"><div class="progress"><span class="spinner"></span>Loading…</div></div>`;
  let data;
  try { data = await api.library(); } catch (e) { $("#lib-body").innerHTML = `<div class="notice error">${icon("alert")}<span>${esc(e.message)}</span></div>`; return; }
  const { recordings, stats } = data;
  const label = (k) => (classes.find((c) => c.key === k) || {}).label;
  $("#lib-body").innerHTML = `
    <div class="stat-row"><div class="stat"><b>${stats.recordings}</b><span>Recordings</span></div><div class="stat"><b>${stats.analyses}</b><span>Analyses</span></div>
      <div class="stat"><b>${stats.processed}</b><span>Processed files</span></div><div class="stat"><b>${stats.labels}</b><span>Corrections for the classifier</span></div></div>
    ${recordings.length ? recordings.map((r) => `
      <article class="lib-item" data-id="${r.id}"><div class="lib-main">
        <div class="name">${esc(r.name)}</div>
        ${r.source_class ? `<span class="badge">${esc(label(r.source_class) || r.source_class)}</span>` : '<span class="badge plain">Not analysed</span>'}
        <span class="badge plain">${r.source === "sample" ? "Sample" : r.source === "recording" ? "Microphone" : "Upload"}</span>
        <span class="meta">${fmtTime(r.duration)} · ${(r.sample_rate / 1000).toFixed(r.sample_rate % 1000 ? 1 : 0)} kHz · ${fmtSize(r.size_bytes)} · ${fmtDate(r.created_at)}</span>
        <div class="row"><button class="btn small" data-act="open">${icon("studio")}Open</button>
          <button class="btn small" data-act="dl" title="Download original">${icon("download")}Original</button>
          <button class="btn small danger" data-act="del" aria-label="Delete ${esc(r.name)}">${icon("trash")}</button></div></div>
        ${r.n_processed ? `<div class="lib-sub" data-sub><span class="small muted">Loading processed versions…</span></div>` : ""}
      </article>`).join("") : `<div class="card empty"><h3>Nothing here yet</h3><p>Record or upload something in the Studio and it will appear here.</p></div>`}`;

  $$(".lib-item").forEach(async (el) => {
    const id = el.dataset.id;
    el.addEventListener("click", async (e) => {
      const b = e.target.closest("[data-act],[data-pact]"); if (!b) return;
      if (b.dataset.act === "open") open(id);
      if (b.dataset.act === "dl") download(api.audioUrl(id, true));
      if (b.dataset.act === "del") {
        if (!confirm("Delete this recording and all of its processed versions?")) return;
        try { await api.remove(id); toast("Recording deleted.", "good"); renderLibrary(root, { open, openResult, classes }); } catch (err) { toast(err.message, "error"); }
      }
      if (b.dataset.pact) {
        const pid = b.dataset.pid;
        if (b.dataset.pact === "view") openResult(id, pid);
        if (b.dataset.pact === "dl") download(api.processedUrl(pid, true));
        if (b.dataset.pact === "rep") download(api.reportUrl(pid));
        if (b.dataset.pact === "del") { try { await api.removeProcessed(pid); renderLibrary(root, { open, openResult, classes }); } catch (err) { toast(err.message, "error"); } }
      }
    });
    const sub = el.querySelector("[data-sub]");
    if (sub) {
      try {
        const d = await api.detail(id);
        sub.innerHTML = d.processed.map((p) => `<div class="r"><span class="badge sage">${esc(label(p.profile) || "Custom")}</span>
          <span>${fmtDate(p.created_at)}</span><span class="muted">noise floor ${fmtDb(-p.metrics.noise_floor_reduction_db)}, ${p.metrics.energy_removed_pct.toFixed(0)}% energy removed</span>
          <span class="row" style="margin-left:auto"><button class="btn small" data-pact="view" data-pid="${p.id}">View result</button>
          <button class="btn small" data-pact="dl" data-pid="${p.id}">${icon("download")}WAV</button>
          <button class="btn small" data-pact="rep" data-pid="${p.id}">Report</button>
          <button class="btn small danger" data-pact="del" data-pid="${p.id}" aria-label="Delete processed file">${icon("trash")}</button></span></div>`).join("");
      } catch (err) { sub.textContent = err.message; }
    }
  });
}
