const $ = (id) => document.getElementById(id);
const form = $("forecast-form");
const fileInput = $("csv-file");
const dropzone = $("dropzone");
let lastForecast = [];

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  $("file-title").textContent = file ? file.name : "Drop your CSV here";
  $("file-hint").textContent = file ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ready to process` : "or browse files · up to 25 MB";
});
for (const event of ["dragenter", "dragover"]) dropzone.addEventListener(event, (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
for (const event of ["dragleave", "drop"]) dropzone.addEventListener(event, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); });
dropzone.addEventListener("drop", (e) => { const file = e.dataTransfer.files[0]; if (file?.name.toLowerCase().endsWith(".csv")) { fileInput.files = e.dataTransfer.files; fileInput.dispatchEvent(new Event("change")); } });
$("theme-toggle").addEventListener("click", () => { document.body.classList.toggle("dark"); localStorage.setItem("current-theme", document.body.classList.contains("dark") ? "dark" : "light"); if (lastForecast.length) renderChart(lastForecast); });
if (localStorage.getItem("current-theme") === "dark") document.body.classList.add("dark");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;
  const data = new FormData();
  data.append("file", file);
  data.append("horizon", $("horizon").value);
  const button = form.querySelector("button[type=submit]");
  button.disabled = true;
  $("error-message").classList.add("hidden");
  $("status-banner").classList.remove("hidden");
  $("status-message").textContent = "Uploading your data and starting the forecast…";
  try {
    const base = (window.CURRENT_API_URL || "http://localhost:8000").replace(/\/$/, "");
    const response = await fetch(`${base}/forecast`, { method: "POST", body: data });
    const started = await response.json();
    if (!response.ok) throw new Error(started.detail || started.error || `Request failed (${response.status})`);
    let result;
    while (!result) {
      await new Promise((resolve) => setTimeout(resolve, 3000));
      const statusResponse = await fetch(`${base}/forecast/${encodeURIComponent(started.job_id)}`);
      const status = await statusResponse.json();
      if (!statusResponse.ok) throw new Error(status.detail || status.error || `Status request failed (${statusResponse.status})`);
      if (status.status === "failed") throw new Error(status.error || "The forecast could not be completed.");
      if (status.status === "completed") {
        result = status.result;
      } else if (status.status === "training") {
        const loss = Number.isFinite(status.validation_loss) ? ` · validation loss ${status.validation_loss.toFixed(5)}` : "";
        $("status-message").textContent = `Training your LSTM · epoch ${status.epoch} of up to ${status.total_epochs}${loss}`;
      } else {
        $("status-message").textContent = status.message || "Preparing your forecast…";
      }
    }
    showResults(result);
  } catch (error) {
    console.error("Forecast request failed:", error);
    $("error-message").textContent = `Forecast failed: ${error.message}`;
    $("error-message").classList.remove("hidden");
  } finally {
    button.disabled = false;
    $("status-banner").classList.add("hidden");
  }
});

function formatDate(value, options = { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) { return new Intl.DateTimeFormat(undefined, options).format(new Date(value)); }
function showResults(data) {
  lastForecast = data;
  const { metrics } = data;
  const metric = (value) => Number(value).toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 4 });
  $("metric-mae").textContent = metric(metrics.mae);
  $("metric-rmse").textContent = metric(metrics.rmse);
  $("metric-mse").textContent = metric(metrics.mse);
  $("metric-r2").textContent = metric(metrics.r2);
  $("trained-label").textContent = `${data.source} · ${data.rows_used.toLocaleString()} rows · best epoch ${metrics.epochs} · test ${metrics.test_rows.toLocaleString()}`;
  renderChart(data);
  $("chart-start").textContent = formatDate(data.observed[0].datetime, { month: "short", day: "numeric" });
  $("chart-end").textContent = formatDate(data.forecast.at(-1).datetime, { month: "short", day: "numeric" });
  const rows = data.forecast.map((item, index) => `<tr><td>${String(index + 1).padStart(2, "0")}</td><td>${formatDate(item.datetime)}</td><td>${Number(item.value).toLocaleString(undefined, { maximumFractionDigits: 3 })}</td><td>FORECAST</td></tr>`).join("");
  $("forecast-rows").innerHTML = rows;
  $("results").classList.remove("hidden");
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderChart(data) {
  const svg = $("chart"), observed = data.observed, future = data.forecast;
  const values = [...observed.map((d) => d.value), ...future.map((d) => d.value)];
  let min = Math.min(...values), max = Math.max(...values); if (min === max) { min -= 1; max += 1; }
  const pad = (max - min) * 0.12; min -= pad; max += pad;
  const W = 900, H = 330, L = 64, R = 14, T = 14, B = 35, plotW = W - L - R, plotH = H - T - B;
  const x = (i) => L + i / (values.length - 1) * plotW;
  const y = (v) => T + (max - v) / (max - min) * plotH;
  const poly = (arr, offset) => arr.map((d, i) => `${i ? "L" : "M"}${x(offset + i).toFixed(1)},${y(d.value).toFixed(1)}`).join(" ");
  const ticks = 4;
  let content = "";
  for (let i = 0; i <= ticks; i++) { const val = min + (max - min) * i / ticks, yy = y(val); content += `<line class="grid-line" x1="${L}" y1="${yy}" x2="${W - R}" y2="${yy}"/><text class="axis-label" x="${L - 9}" y="${yy + 3}" text-anchor="end">${Number(val).toLocaleString(undefined, { maximumFractionDigits: 1 })}</text>`; }
  const splitX = x(observed.length - 1);
  const futurePath = poly(future, observed.length);
  const area = `${futurePath} L${x(values.length - 1)},${T + plotH} L${splitX},${T + plotH} Z`;
  content += `<path class="forecast-area" d="${area}"/><line class="grid-line" x1="${splitX}" y1="${T}" x2="${splitX}" y2="${T + plotH}" stroke-dasharray="3 5"/><path class="observed-line" d="${poly(observed, 0)}"/><path class="forecast-line" d="M${splitX},${y(observed.at(-1).value)} ${futurePath.slice(futurePath.indexOf(" ") + 1)}"/>`;
  for (let i = 0; i < future.length; i += Math.max(1, Math.floor(future.length / 7))) content += `<circle class="chart-dot" cx="${x(observed.length + i)}" cy="${y(future[i].value)}" r="3.5"><title>${formatDate(future[i].datetime)} · ${future[i].value.toFixed(2)}</title></circle>`;
  const first = formatDate(observed[0].datetime, { month: "short", day: "numeric" }), mid = formatDate(observed[Math.floor(observed.length / 2)].datetime, { month: "short", day: "numeric" }), last = formatDate(future.at(-1).datetime, { month: "short", day: "numeric" });
  content += `<text class="axis-label" x="${L}" y="${H - 7}">${first}</text><text class="axis-label" x="${splitX}" y="${H - 7}" text-anchor="middle">${mid} · NOW</text><text class="axis-label" x="${W - R}" y="${H - 7}" text-anchor="end">${last}</text>`;
  svg.innerHTML = content;
}

$("download-csv").addEventListener("click", () => {
  if (!lastForecast.length) return;
  const rows = [["Date and Hour", "Forecast Production"], ...lastForecast.forecast.map((item) => [item.datetime, item.value])];
  const csv = rows.map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\n");
  const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); link.download = "renewables-forecast.csv"; link.click(); URL.revokeObjectURL(link.href);
});

