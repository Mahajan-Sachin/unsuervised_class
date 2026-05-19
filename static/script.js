const dropZone   = document.getElementById('dropZone');
const csvInput   = document.getElementById('csvInput');
const fileInfo   = document.getElementById('fileInfo');
const fileName   = document.getElementById('fileName');
const clearFile  = document.getElementById('clearFile');
const analyzeBtn = document.getElementById('analyzeBtn');
const analyzeLabel = document.getElementById('analyzeLabel');

let selectedFile = null;

// ── Open file picker on drop zone click ───────────────────
dropZone.addEventListener('click', () => csvInput.click());

csvInput.addEventListener('change', () => {
  if (csvInput.files[0]) setFile(csvInput.files[0]);
});

// ── Drag & Drop ───────────────────────────────────────────
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f && f.name.endsWith('.csv')) setFile(f);
  else alert('Please drop a .csv file');
});

function setFile(f) {
  selectedFile = f;
  fileName.textContent = `${f.name}  (${(f.size / 1024).toFixed(1)} KB)`;
  fileInfo.classList.remove('hidden');
  analyzeBtn.disabled = false;
}

clearFile.addEventListener('click', () => {
  selectedFile = null;
  csvInput.value = '';
  fileInfo.classList.add('hidden');
  analyzeBtn.disabled = true;
});

// ── Analyze Button ────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  analyzeLabel.innerHTML = '<span class="spinner"></span> Analyzing...';

  const form = new FormData();
  form.append('file', selectedFile);

  try {
    const res  = await fetch('/upload', { method: 'POST', body: form });
    const data = await res.json();

    if (data.error) { alert('Error: ' + data.error); return; }
    showResults(data);
  } catch (err) {
    alert('Network error: ' + err.message);
  } finally {
    analyzeBtn.disabled = false;
    analyzeLabel.textContent = '🔍 Analyze Transactions';
  }
});

// ── Show Results ──────────────────────────────────────────
function showResults(data) {
  // Summary
  document.getElementById('sTotal').textContent  = data.total;
  document.getElementById('sNormal').textContent = data.normal_count;
  document.getElementById('sFraud').textContent  = data.fraud_count;
  document.getElementById('sFraudPct').textContent = data.fraud_pct + '%';

  // Fraud alert
  const alert = document.getElementById('fraudAlert');
  const alertText = document.getElementById('fraudAlertText');
  if (data.fraud_count > 0) {
    alertText.textContent = `${data.fraud_count} out of ${data.total}`;
    alert.classList.remove('hidden');
  } else {
    alert.classList.add('hidden');
  }

  // Table rows
  const tbody = document.getElementById('resultsBody');
  tbody.innerHTML = '';

  data.transactions.forEach((t) => {
    const tr = document.createElement('tr');
    if (t.is_anomaly) tr.classList.add('row-fraud');

    const isoTag = tag(t.isolation_forest);
    const aeTag   = tag(t.autoencoder);
    const statusTag = t.is_anomaly
      ? '<span class="tag tag-anomaly">🚨 Fraud</span>'
      : '<span class="tag tag-normal">✅ Normal</span>';

    tr.innerHTML = `
      <td>${t.row}</td>
      <td>${t.time}</td>
      <td>$${t.amount}</td>
      <td>${isoTag}</td>
      <td>${aeTag}</td>
      <td>${statusTag}</td>
    `;
    tbody.appendChild(tr);
  });

  // update sticky bar info
  document.getElementById('stickyInfo').textContent =
    `${data.fraud_count} fraud / ${data.total} transactions`;

  document.getElementById('uploadCard').classList.add('hidden');
  const rc = document.getElementById('resultsCard');
  rc.classList.remove('hidden');
  rc.classList.add('fade-in');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function tag(val) {
  return val === 'Anomaly'
    ? '<span class="tag tag-anomaly">Anomaly</span>'
    : '<span class="tag tag-normal">Normal</span>';
}

// ── Reset ─────────────────────────────────────────────────
document.getElementById('resetBtn').addEventListener('click', reset);
document.getElementById('stickyReset').addEventListener('click', reset);

function reset() {
  document.getElementById('resultsCard').classList.add('hidden');
  document.getElementById('uploadCard').classList.remove('hidden');
  clearFile.click();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
