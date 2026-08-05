const NT_GEOM = @@NT_GEOMETRY_DATA@@;
const RETRIEVAL_ATLAS = @@RETRIEVAL_SCORE_DATA@@;

const ntControls = {
  model: document.getElementById('nt-model'),
  layer: document.getElementById('nt-layer'),
  basis: document.getElementById('nt-basis'),
  points: document.getElementById('nt-points'),
  pairs: document.getElementById('nt-pairs'),
  x: document.getElementById('nt-axis-x'),
  y: document.getElementById('nt-axis-y'),
  z: document.getElementById('nt-axis-z')
};
const ntCamera = {yaw: -.72, pitch: .42, zoom: 1};
let ntScreenPoints = [];

function ntActiveKey() {
  return ntControls.model.value + '|answer_query|' + ntControls.layer.value;
}

function drawNtGeometry() {
  const dataset = NT_GEOM.datasets[ntActiveKey()];
  const stat = NT_GEOM.statistics[ntActiveKey()];
  if (!dataset || !stat) return;
  const canvas = document.getElementById('nt-answer-canvas');
  const sized = resizeCanvas(canvas);
  const ctx = sized.ctx;
  const w = sized.w;
  const h = sized.h;
  const basis = ntControls.basis.value;
  const axes = [+ntControls.x.value, +ntControls.y.value, +ntControls.z.value];
  const rows = dataset.rows;
  const all = [];
  for (const row of rows) {
    all.push(rowCoords(row, 'cue_present', basis));
    all.push(rowCoords(row, 'cue_absent', basis));
  }
  const tf = transformFactory(all, axes, w, h, ntCamera);
  ctx.fillStyle = '#15112B';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(255,255,255,.10)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(w * .08, h / 2);
  ctx.lineTo(w * .92, h / 2);
  ctx.moveTo(w / 2, h * .08);
  ctx.lineTo(w / 2, h * .92);
  ctx.stroke();

  ntScreenPoints = [];
  if (ntControls.pairs.checked) {
    ctx.strokeStyle = 'rgba(255,255,255,.10)';
    ctx.lineWidth = .8;
    for (const row of rows) {
      const present = tf(rowCoords(row, 'cue_present', basis));
      const absent = tf(rowCoords(row, 'cue_absent', basis));
      ctx.beginPath();
      ctx.moveTo(present.x, present.y);
      ctx.lineTo(absent.x, absent.y);
      ctx.stroke();
    }
  }

  if (ntControls.points.value === 'all') {
    const items = [];
    for (const row of rows) {
      for (const condition of ['cue_present', 'cue_absent']) {
        items.push({row, condition, q: tf(rowCoords(row, condition, basis))});
      }
    }
    items.sort((a, b) => a.q.z - b.q.z);
    for (const item of items) {
      const count = item.row[1];
      const correct = item.condition === 'cue_present' ? item.row[2] : item.row[3];
      ctx.globalAlpha = correct ? .58 : .22;
      ctx.fillStyle = COUNT_COLORS[count - 1];
      ctx.strokeStyle = item.condition === 'cue_present' ? '#FFFDF8' : '#F6E36A';
      ctx.lineWidth = item.condition === 'cue_present' ? 1.1 : 1.4;
      if (item.condition === 'cue_present') {
        ctx.beginPath();
        ctx.arc(item.q.x, item.q.y, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.fillRect(item.q.x - 3, item.q.y - 3, 6, 6);
        ctx.strokeRect(item.q.x - 3, item.q.y - 3, 6, 6);
      }
      ntScreenPoints.push({x: item.q.x, y: item.q.y, row: item.row, condition: item.condition});
    }
    ctx.globalAlpha = 1;
  }

  for (const condition of ['cue_present', 'cue_absent']) {
    const centers = centroids(rows, condition, basis)
      .map((point, index) => ({count: index + 1, q: tf(point)}));
    ctx.strokeStyle = condition === 'cue_present' ? '#FFFDF8' : '#F6E36A';
    ctx.lineWidth = 2.6;
    ctx.setLineDash(condition === 'cue_present' ? [] : [7, 5]);
    ctx.beginPath();
    centers.forEach((item, index) => {
      if (index) ctx.lineTo(item.q.x, item.q.y);
      else ctx.moveTo(item.q.x, item.q.y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    for (const item of centers) {
      ctx.fillStyle = COUNT_COLORS[item.count - 1];
      ctx.strokeStyle = condition === 'cue_present' ? '#FFFDF8' : '#F6E36A';
      ctx.lineWidth = 1.4;
      if (condition === 'cue_present') {
        ctx.beginPath();
        ctx.arc(item.q.x, item.q.y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.fillRect(item.q.x - 5, item.q.y - 5, 10, 10);
        ctx.strokeRect(item.q.x - 5, item.q.y - 5, 10, 10);
      }
    }
  }

  const evr = basis === 'raw' ? dataset.evr_raw : dataset.evr_cue_centered;
  const accuracyPresent = rows.reduce((sum, row) => sum + row[2], 0) / rows.length;
  const accuracyAbsent = rows.reduce((sum, row) => sum + row[3], 0) / rows.length;
  document.getElementById('nt-answer-stats').innerHTML =
    statHtml(stat, evr) + '<br><strong>accuracy</strong> ' +
    format(accuracyPresent * 100, 0) + '% → ' + format(accuracyAbsent * 100, 0) +
    '% (' + (accuracyAbsent >= accuracyPresent ? '+' : '') +
    format((accuracyAbsent - accuracyPresent) * 100, 0) + ' pp)';
  document.getElementById('nt-selected-conclusion').innerHTML =
    '<strong>当前层结论</strong><span>' + ntControls.model.value + ' · L' +
    ntControls.layer.value + ' · count-strength ' +
    (+stat.count_eta_q < .05 ? '显著' : '不显著') + ' (q=' +
    pformat(stat.count_eta_q) + ')；count×cue interaction ' +
    (+stat.interaction_q < .05 ? '显著' : '不显著') + ' (q=' +
    pformat(stat.interaction_q) + ')。</span>';
}

function drawNtSweep() {
  const canvas = document.getElementById('nt-sweep-canvas');
  const sized = resizeCanvas(canvas);
  const ctx = sized.ctx;
  const w = sized.w;
  const h = sized.h;
  const model = ntControls.model.value;
  const layers = Object.values(NT_GEOM.datasets)
    .filter(row => row.model === model && row.site === 'answer_query')
    .map(row => +row.layer)
    .sort((a, b) => a - b);
  const pad = {l: 54, r: 22, t: 25, b: 34};
  const gap = 24;
  const panelH = (h - pad.t - pad.b - gap * 2) / 3;
  const x = layer => pad.l + (w - pad.l - pad.r) *
    (layer - layers[0]) / (layers[layers.length - 1] - layers[0]);
  const panels = [
    {title: 'Centroid CKA', min: 0, max: 1, value: stat => stat.centroid_cka},
    {title: 'Δ count η² (absent−present)', min: -.25, max: .25, value: stat => stat.count_eta_delta},
    {title: '−log10 interaction q', min: 0, max: 4, value: stat => Math.min(4, -Math.log10(Math.max(+stat.interaction_q, 1e-4)))}
  ];
  ctx.fillStyle = '#FFFDF8';
  ctx.fillRect(0, 0, w, h);
  panels.forEach((panel, panelIndex) => {
    const top = pad.t + panelIndex * (panelH + gap);
    const bottom = top + panelH;
    const y = value => bottom - (bottom - top) * (value - panel.min) / (panel.max - panel.min);
    ctx.strokeStyle = '#C9C2B6';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, top);
    ctx.lineTo(pad.l, bottom);
    ctx.lineTo(w - pad.r, bottom);
    ctx.stroke();
    ctx.fillStyle = '#20242D';
    ctx.font = '12px Segoe UI';
    ctx.fillText(panel.title, pad.l, top - 8);
    for (let tick = 0; tick <= 4; tick++) {
      const value = panel.min + (panel.max - panel.min) * tick / 4;
      const yy = y(value);
      ctx.strokeStyle = 'rgba(94,102,114,.18)';
      ctx.beginPath();
      ctx.moveTo(pad.l, yy);
      ctx.lineTo(w - pad.r, yy);
      ctx.stroke();
      ctx.fillStyle = '#5E6672';
      ctx.font = '10px Consolas';
      ctx.fillText(format(value, 2), 5, yy + 3);
    }
    if (panelIndex === 2) {
      const threshold = y(-Math.log10(.05));
      ctx.strokeStyle = '#D6B52C';
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(pad.l, threshold);
      ctx.lineTo(w - pad.r, threshold);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    const stats = layers.map(layer => NT_GEOM.statistics[model + '|answer_query|' + layer]);
    ctx.strokeStyle = '#D94B86';
    ctx.lineWidth = 2;
    ctx.beginPath();
    stats.forEach((stat, index) => {
      const xx = x(layers[index]);
      const yy = y(clamp(panel.value(stat), panel.min, panel.max));
      if (index) ctx.lineTo(xx, yy);
      else ctx.moveTo(xx, yy);
    });
    ctx.stroke();
    stats.forEach((stat, index) => {
      const xx = x(layers[index]);
      const yy = y(clamp(panel.value(stat), panel.min, panel.max));
      const significant = panelIndex === 1 ? +stat.count_eta_q < .05 :
        panelIndex === 2 ? +stat.interaction_q < .05 : false;
      ctx.fillStyle = significant ? '#D94B86' : '#FFFDF8';
      ctx.strokeStyle = '#D94B86';
      ctx.beginPath();
      ctx.arc(xx, yy, significant ? 3.4 : 2.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      if (panelIndex === 1) {
        const low = y(clamp(+stat.count_eta_delta_ci_low, panel.min, panel.max));
        const high = y(clamp(+stat.count_eta_delta_ci_high, panel.min, panel.max));
        ctx.globalAlpha = .22;
        ctx.beginPath();
        ctx.moveTo(xx, low);
        ctx.lineTo(xx, high);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });
    const selected = x(+ntControls.layer.value);
    ctx.strokeStyle = '#23165C';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(selected, top);
    ctx.lineTo(selected, bottom);
    ctx.stroke();
    ctx.setLineDash([]);
  });
  ctx.fillStyle = '#5E6672';
  ctx.font = '10px Consolas';
  for (let index = 0; index < layers.length; index += 5) {
    ctx.fillText('L' + layers[index], x(layers[index]) - 7, h - 8);
  }
}

function updateNtSignificance() {
  const model = ntControls.model.value;
  const layers = Object.values(NT_GEOM.datasets)
    .filter(row => row.model === model && row.site === 'answer_query')
    .map(row => +row.layer)
    .sort((a, b) => a - b);
  const stats = layers.map(layer => NT_GEOM.statistics[model + '|answer_query|' + layer]);
  document.getElementById('nt-sig-table').innerHTML = '<tr><td>answer query</td><td>' +
    ranges(layers.filter((layer, index) => +stats[index].count_eta_q < .05)) +
    '</td><td>' +
    ranges(layers.filter((layer, index) => +stats[index].interaction_q < .05)) +
    '</td></tr>';
}

function refreshNtLayers() {
  const model = ntControls.model.value;
  const layers = Object.values(NT_GEOM.datasets)
    .filter(row => row.model === model && row.site === 'answer_query')
    .map(row => +row.layer)
    .sort((a, b) => a - b);
  const hadOptions = ntControls.layer.options.length > 0;
  const old = +ntControls.layer.value;
  ntControls.layer.innerHTML = '';
  for (const layer of layers) {
    const option = document.createElement('option');
    option.value = String(layer);
    option.textContent = 'L' + layer +
      (layer === NT_GEOM.landmarks[model].display ? ' · prior display landmark' : '') +
      (layer === NT_GEOM.landmarks[model].probe ? ' · prior probe landmark' : '');
    ntControls.layer.appendChild(option);
  }
  ntControls.layer.value = String(
    hadOptions && layers.includes(old) ? old : NT_GEOM.landmarks[model].display
  );
}

function drawNtAll() {
  drawNtGeometry();
  drawNtSweep();
  updateNtSignificance();
}

function attachNtGeometry() {
  const canvas = document.getElementById('nt-answer-canvas');
  const tooltip = document.getElementById('nt-answer-tooltip');
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener('pointerdown', event => {
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointerup', () => dragging = false);
  canvas.addEventListener('pointermove', event => {
    if (dragging) {
      ntCamera.yaw += (event.clientX - lastX) * .008;
      ntCamera.pitch = clamp(ntCamera.pitch + (event.clientY - lastY) * .008, -1.45, 1.45);
      lastX = event.clientX;
      lastY = event.clientY;
      drawNtGeometry();
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let best = null;
    let distance = Infinity;
    for (const point of ntScreenPoints) {
      const value = (point.x - x) ** 2 + (point.y - y) ** 2;
      if (value < distance) {
        distance = value;
        best = point;
      }
    }
    if (best && distance < 90) {
      tooltip.style.display = 'block';
      tooltip.style.left = Math.min(rect.width - 250, x + 13) + 'px';
      tooltip.style.top = Math.max(8, y - 12) + 'px';
      const correct = best.condition === 'cue_present' ? best.row[2] : best.row[3];
      tooltip.innerHTML = '<strong>count ' + best.row[1] + ' · seed ' + best.row[0] +
        '</strong><br>' + best.condition + ' · ' + (correct ? 'correct' : 'wrong') +
        '<br>answer query · L' + ntControls.layer.value;
    } else {
      tooltip.style.display = 'none';
    }
  });
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    ntCamera.zoom = clamp(ntCamera.zoom * Math.exp(-event.deltaY * .001), .55, 2.8);
    drawNtGeometry();
  }, {passive: false});
}

const headControls = {
  model: document.getElementById('head-model'),
  mode: document.getElementById('head-mode')
};

function headMatrix(condition) {
  const modeData = RETRIEVAL_ATLAS.models[headControls.model.value]
    .modes[headControls.mode.value];
  const source = modeData.conditions[condition].layer_head_score;
  return Array.from({length: modeData.heads}, (_, head) =>
    modeData.layers.map((_, layerIndex) => source[layerIndex][head])
  );
}

function atlasQuantile(values, probability) {
  const sorted = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (!sorted.length) return 0;
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function atlasSequential(value, maxValue) {
  const t = clamp(value / Math.max(maxValue, 1e-12), 0, 1);
  const start = [247, 243, 234];
  const middle = [88, 139, 210];
  const end = [35, 22, 92];
  const local = t < .55 ? t / .55 : (t - .55) / .45;
  const left = t < .55 ? start : middle;
  const right = t < .55 ? middle : end;
  return 'rgb(' + left.map((value, index) =>
    Math.round(value + (right[index] - value) * local)).join(',') + ')';
}

function atlasDiverging(value, maxValue) {
  const t = clamp(value / Math.max(maxValue, 1e-12), -1, 1);
  const center = [247, 243, 234];
  const end = t < 0 ? [49, 91, 199] : [181, 61, 102];
  const amount = Math.abs(t);
  return 'rgb(' + center.map((value, index) =>
    Math.round(value + (end[index] - value) * amount)).join(',') + ')';
}

function drawHeadAtlas(canvasId, matrix, options) {
  const canvas = document.getElementById(canvasId);
  const sized = resizeCanvas(canvas);
  const ctx = sized.ctx;
  const w = sized.w;
  const h = sized.h;
  const rows = matrix.length;
  const cols = matrix[0].length;
  const left = 42;
  const right = 10;
  const top = 12;
  const bottom = 38;
  const plotW = w - left - right;
  const plotH = h - top - bottom;
  const cellW = plotW / cols;
  const cellH = plotH / rows;
  ctx.fillStyle = '#FFFDF8';
  ctx.fillRect(0, 0, w, h);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const value = matrix[row][col];
      ctx.fillStyle = value == null ? '#DED8CE' :
        options.diverging ? atlasDiverging(value, options.max) :
          atlasSequential(value, options.max);
      ctx.fillRect(
        left + col * cellW,
        top + row * cellH,
        Math.ceil(cellW + .25),
        Math.ceil(cellH + .25)
      );
    }
  }
  ctx.strokeStyle = '#C9C2B6';
  ctx.lineWidth = 1;
  ctx.strokeRect(left, top, plotW, plotH);
  ctx.fillStyle = '#5E6672';
  ctx.font = '10px Consolas';
  const layers = options.layers;
  for (let col = 0; col < cols; col += 5) {
    ctx.save();
    ctx.translate(left + (col + .5) * cellW, h - bottom + 6);
    ctx.rotate(-Math.PI / 3);
    ctx.fillText('L' + layers[col], 0, 0);
    ctx.restore();
  }
  const headStep = rows <= 12 ? 1 : 4;
  for (let row = 0; row < rows; row += headStep) {
    ctx.fillText('H' + row, 7, top + (row + .65) * cellH);
  }
  canvas.onmousemove = event => {
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * w / rect.width;
    const y = (event.clientY - rect.top) * h / rect.height;
    const col = Math.floor((x - left) / cellW);
    const row = Math.floor((y - top) / cellH);
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      const value = matrix[row][col];
      document.getElementById('head-atlas-hover').textContent =
        options.label + ' · L' + layers[col] + ' H' + row + ' · score ' +
        (value == null ? 'N/A' : format(value, 7));
    }
  };
}

function drawHeadAtlases() {
  const present = headMatrix('cue_present');
  const absent = headMatrix('cue_absent');
  const delta = matrixDelta(present, absent);
  const actualValues = [...finiteValues(present), ...finiteValues(absent)];
  const deltaValues = finiteValues(delta).map(Math.abs);
  const actualMax = Math.max(atlasQuantile(actualValues, .995), 1e-12);
  const deltaMax = Math.max(atlasQuantile(deltaValues, .995), 1e-12);
  const modeData = RETRIEVAL_ATLAS.models[headControls.model.value]
    .modes[headControls.mode.value];
  drawHeadAtlas('head-atlas-present', present, {
    max: actualMax,
    layers: modeData.layers,
    label: 'cue-present'
  });
  drawHeadAtlas('head-atlas-absent', absent, {
    max: actualMax,
    layers: modeData.layers,
    label: 'cue-absent'
  });
  drawHeadAtlas('head-atlas-delta', delta, {
    max: deltaMax,
    layers: modeData.layers,
    label: 'absent−present',
    diverging: true
  });
  document.querySelectorAll('.head-actual-max').forEach(node =>
    node.textContent = 'p99.5 cap ' + format(actualMax, 5));
  document.querySelectorAll('.head-delta-min').forEach(node =>
    node.textContent = '−' + format(deltaMax, 5));
  document.querySelectorAll('.head-delta-max').forEach(node =>
    node.textContent = '+' + format(deltaMax, 5));

  const changes = [];
  for (let head = 0; head < delta.length; head++) {
    for (let layerIndex = 0; layerIndex < delta[head].length; layerIndex++) {
      if (delta[head][layerIndex] == null) continue;
      changes.push({
        layer: modeData.layers[layerIndex],
        head,
        present: present[head][layerIndex],
        absent: absent[head][layerIndex],
        delta: delta[head][layerIndex]
      });
    }
  }
  changes.sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta));
  document.getElementById('head-change-table').innerHTML = changes.length ? changes.slice(0, 12)
    .map(row => '<tr><td>L' + row.layer + ' / H' + row.head + '</td><td>' +
      format(row.present, 7) + '</td><td>' + format(row.absent, 7) +
      '</td><td>' + (row.delta >= 0 ? '+' : '') + format(row.delta, 7) +
      '</td></tr>').join('') :
    '<tr><td colspan="4">N/A: no target token is visible to the selected trace queries.</td></tr>';
  const presentValid = modeData.conditions.cue_present.valid_samples_by_layer;
  const absentValid = modeData.conditions.cue_absent.valid_samples_by_layer;
  const presentRange = Math.min(...presentValid) + '–' + Math.max(...presentValid);
  const absentRange = Math.min(...absentValid) + '–' + Math.max(...absentValid);
  const unavailable = Math.max(...presentValid, ...absentValid) === 0;
  const structurallyZero = actualValues.length > 0 &&
    Math.max(...actualValues.map(Math.abs)) <= 1e-12;
  document.getElementById('head-atlas-caption').textContent =
    (headControls.mode.value === 'nonthinking' ? 'non-thinking' : 'native thinking') +
    ' · ' + headControls.model.value + ' · ' + modeData.score_definition.name +
    ' · site: ' + modeData.score_definition.site +
    ' · ' + modeData.score_definition.formula +
    ' · finite n/layer: present ' + presentRange + ', absent ' + absentRange +
    ' (scheduled ' + modeData.conditions.cue_present.samples +
    ' each); color is capped at the within-mode p99.5 for readability.' +
    (unavailable ?
      ' N/A means the architecture window exposes no original needle token to any saved trace query; this is structural, not missing data.' :
      structurallyZero ?
        ' All direct retrieval scores are zero under the captured attention masks; this is structural, not missing data.' : '');
}

for (const key of ['layer', 'basis', 'points', 'pairs', 'x', 'y', 'z']) {
  ntControls[key].addEventListener('change', drawNtAll);
}
ntControls.model.addEventListener('change', () => {
  refreshNtLayers();
  drawNtAll();
});
document.getElementById('nt-reset').addEventListener('click', () => {
  ntCamera.yaw = -.72;
  ntCamera.pitch = .42;
  ntCamera.zoom = 1;
  drawNtGeometry();
});
document.getElementById('nt-sweep-canvas').addEventListener('click', event => {
  const canvas = event.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const model = ntControls.model.value;
  const layers = Object.values(NT_GEOM.datasets)
    .filter(row => row.model === model && row.site === 'answer_query')
    .map(row => +row.layer)
    .sort((a, b) => a - b);
  const left = 54;
  const right = 22;
  const index = Math.round(
    (event.clientX - rect.left - left) / (rect.width - left - right) *
    (layers.length - 1)
  );
  ntControls.layer.value = String(layers[clamp(index, 0, layers.length - 1)]);
  drawNtAll();
});
for (const key of ['model', 'mode']) {
  headControls[key].addEventListener('change', drawHeadAtlases);
}

attachNtGeometry();
refreshNtLayers();
drawNtAll();
drawHeadAtlases();
window.addEventListener('resize', () => {
  drawNtAll();
  drawHeadAtlases();
});
