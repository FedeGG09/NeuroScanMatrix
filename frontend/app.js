
const API = "http://localhost:8000";
let currentDocumentId = null;
let currentResult = null;

const uploadBtn = document.getElementById("uploadBtn");
const extractBtn = document.getElementById("extractBtn");
const fileInput = document.getElementById("fileInput");
const apiStatus = document.getElementById("apiStatus");
const uploadInfo = document.getElementById("uploadInfo");
const summary = document.getElementById("summary");
const jsonOutput = document.getElementById("jsonOutput");
const rowsTable = document.getElementById("rowsTable");
const rowsCount = document.getElementById("rowsCount");

async function checkHealth() {
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    apiStatus.textContent = `API: ${data.status}`;
    apiStatus.style.background = "#e5f7ef";
    apiStatus.style.color = "#16623f";
  } catch (e) {
    apiStatus.textContent = "API: desconectada";
    apiStatus.style.background = "#fbeaea";
    apiStatus.style.color = "#9b1c1c";
  }
}

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function asBool(value) {
  return value === true || value === 1 || value === "true" || value === "True" || value === "sí" || value === "si";
}

function renderDownloadButtons(result) {
  const container = document.getElementById("downloadLinks");
  const csvUrl = result.csv_download_url ? `${API}${result.csv_download_url}` : null;
  const xlsxUrl = result.xlsx_download_url ? `${API}${result.xlsx_download_url}` : null;

  container.innerHTML = `
    ${csvUrl ? `<button class="secondary" id="csvBtn">Descargar CSV</button>` : ""}
    ${xlsxUrl ? `<button class="secondary" id="xlsxBtn">Descargar Excel</button>` : ""}
  `;

  if (csvUrl) document.getElementById("csvBtn").onclick = () => window.open(csvUrl, "_blank");
  if (xlsxUrl) document.getElementById("xlsxBtn").onclick = () => window.open(xlsxUrl, "_blank");
}

function normalizeRow(row) {
  if (!row) return {};
  if (typeof row?.model_dump === "function") return row.model_dump();
  if (typeof row?.dict === "function") return row.dict();
  return row;
}

function buildColumns(rows) {
  const preferred = [
    "page",
    "column",
    "band_index",
    "block_index",
    "nro_orden",
    "apellido_nombre",
    "apellido_y_nombre",
    "domicilio",
    "documento",
    "dni_tipo",
    "anio_nacimiento",
    "voto",
    "voto_score",
    "confidence",
    "bbox",
    "voto_bbox",
    "raw_text",
  ];

  const seen = new Set();
  const cols = [];
  for (const key of preferred) {
    if (rows.some(r => Object.prototype.hasOwnProperty.call(r, key))) {
      cols.push(key);
      seen.add(key);
    }
  }
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!seen.has(key)) {
        cols.push(key);
        seen.add(key);
      }
    }
  }
  return cols;
}

function renderRows(rows) {
  const normalizedRows = (rows || []).map(normalizeRow);
  if (!normalizedRows.length) {
    rowsTable.innerHTML = `<div class="hint">No se detectaron filas.</div>`;
    rowsCount.textContent = "0 filas";
    return;
  }

  rowsCount.textContent = `${normalizedRows.length} filas extraídas`;
  const cols = buildColumns(normalizedRows);
  const header = cols.map(c => `<th>${escapeHtml(c)}</th>`).join("");
  const body = normalizedRows.map(row => `
    <tr>
      ${cols.map(c => {
        const v = row[c];
        if (c === "voto") {
          const isChecked = asBool(v);
          return `<td>${isChecked ? '<span class="badge-yes">Sí</span>' : '<span class="badge-no">No</span>'}</td>`;
        }
        if (c === "confidence" || c === "voto_score") {
          return `<td>${Number(v ?? 0).toFixed(3)}</td>`;
        }
        if (Array.isArray(v) || (v && typeof v === "object")) {
          return `<td><code>${escapeHtml(JSON.stringify(v))}</code></td>`;
        }
        return `<td>${escapeHtml(v)}</td>`;
      }).join("")}
    </tr>
  `).join("");

  rowsTable.innerHTML = `
    <table>
      <thead><tr>${header}</tr></thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function renderResult(result) {
  currentResult = result;
  jsonOutput.textContent = JSON.stringify(result, null, 2);

  const rows = (result.rows || []).map(normalizeRow);
  const checked = rows.filter(r => asBool(r.voto)).length;

  summary.innerHTML = `
    <div class="kpi"><span class="label">Tipo</span><span class="value">${escapeHtml(result.document_type ?? "-")}</span></div>
    <div class="kpi"><span class="label">Confianza global</span><span class="value">${Number(result.confidence_global ?? 0).toFixed(3)}</span></div>
    <div class="kpi"><span class="label">Revisión</span><span class="value">${result.needs_review ? "Sí" : "No"}</span></div>
    <div class="kpi"><span class="label">Páginas</span><span class="value">${result.pages_processed ?? 0}</span></div>
    <div class="kpi"><span class="label">Filas</span><span class="value">${rows.length}</span></div>
    <div class="kpi"><span class="label">Votaron</span><span class="value">${checked}</span></div>
  `;

  renderRows(rows);
  renderDownloadButtons(result);
}

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("Elegí un archivo primero");
    return;
  }
  const form = new FormData();
  form.append("file", file);

  uploadInfo.textContent = "Subiendo...";
  const res = await fetch(`${API}/upload`, { method: "POST", body: form });
  if (!res.ok) {
    uploadInfo.textContent = "Error al subir";
    return;
  }
  const data = await res.json();
  currentDocumentId = data.document_id;
  uploadInfo.textContent = `Documento cargado: ${data.file_name} (${currentDocumentId})`;
  extractBtn.disabled = false;
});

extractBtn.addEventListener("click", async () => {
  if (!currentDocumentId) return;
  extractBtn.disabled = true;
  extractBtn.textContent = "Extrayendo...";
  const res = await fetch(`${API}/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: currentDocumentId })
  });
  extractBtn.textContent = "Extraer filas";
  extractBtn.disabled = false;
  if (!res.ok) {
    const err = await res.text();
    alert(`Error: ${err}`);
    return;
  }
  const data = await res.json();
  renderResult(data);
});

checkHealth();
