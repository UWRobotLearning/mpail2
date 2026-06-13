// ─── Palette ─────────────────────────────────────────────────────────────────
const ALGO_STYLES = {
    // Efficiency
    'MPAIL2':       { line: '#F25912', fill: 'rgba(242,89,18,0.15)',   dash: [],    width: 2.5 },
    '[−P] (MAIRL)': { line: '#E49BA6', fill: 'rgba(228,155,166,0.15)', dash: [],    width: 2   },
    '[−PM] (DAC)':  { line: '#E69F00', fill: 'rgba(230,159,0,0.12)',   dash: [],    width: 2   },
    'GAIL':         { line: '#56B4E9', fill: 'rgba(86,180,233,0.12)',  dash: [],    width: 2   },
    'RLPD':         { line: '#009E73', fill: 'rgba(0,158,115,0.12)',   dash: [],    width: 2   },
    'MPAIL':        { line: '#CC79A7', fill: 'rgba(204,121,167,0.12)', dash: [],    width: 2   },

    // BP Transfer – MPAIL2 family (orange-red)
    'MPAIL2 (Full Transfer)':           { line: '#F25912', fill: 'rgba(242,89,18,0.18)',       dash: [],    width: 2.5 },
    'MPAIL2 (Dynamics Transfer)':       { line: '#F25912', fill: 'rgba(242,89,18,0.10)',       dash: [7,4], width: 2   },
    'MPAIL2 (From Scratch)':            { line: '#F25912', fill: 'rgba(242,89,18,0.06)',       dash: [3,5], width: 2   },
    // BP Transfer – [−P] (MAIRL) family (pink)
    '[−P] (MAIRL) (Full Transfer)':     { line: '#E49BA6', fill: 'rgba(228,155,166,0.15)',     dash: [],    width: 2   },
    '[−P] (MAIRL) (Dynamics Transfer)': { line: '#E49BA6', fill: 'rgba(228,155,166,0.08)',     dash: [7,4], width: 2   },
    '[−P] (MAIRL) (From Scratch)':      { line: '#E49BA6', fill: 'rgba(228,155,166,0.05)',     dash: [3,5], width: 2   },
    // PnP Transfer
    'MPAIL2 (Transferred)':             { line: '#F25912', fill: 'rgba(242,89,18,0.18)',       dash: [],    width: 2.5 },
    '[−P] (MAIRL) (Transferred)':       { line: '#E49BA6', fill: 'rgba(228,155,166,0.15)',     dash: [],    width: 2   },
    '[−P] (MAIRL) (From Scratch) ':     { line: '#E49BA6', fill: 'rgba(228,155,166,0.05)',     dash: [3,5], width: 2   },
    // BC (grey)
    'BC (Transferred)':  { line: '#999999', fill: 'rgba(153,153,153,0.08)', dash: [],    width: 1.8 },
    'BC (From Scratch)': { line: '#999999', fill: 'rgba(153,153,153,0.05)', dash: [3,5], width: 1.8 },
    // Init-phase helpers: shown on chart, hidden from legend
    '_init_mpail2':     { line: '#F25912', fill: 'rgba(242,89,18,0.18)',       dash: [], width: 2.5 },
    '_init_[-p]_mairl': { line: '#E49BA6', fill: 'rgba(228,155,166,0.15)',     dash: [], width: 2   },
    '_bc_real':         { line: '#999999', fill: 'rgba(153,153,153,0.08)',     dash: [], width: 1.8 },
};
const FALLBACK = { line: '#94a3b8', fill: 'rgba(148,163,184,0.10)', dash: [], width: 2 };

function styleFor(label) { return ALGO_STYLES[label] || FALLBACK; }

// Labels shown on chart but excluded from legend
const LEGEND_HIDDEN = new Set([
    '_init_mpail2', '_init_[-p]_mairl', '_bc_real',
    '_init_mpail2__upper', '_init_[-p]_mairl__upper', '_bc_real__upper',
    '_init_mpail2__lower', '_init_[-p]_mairl__lower', '_bc_real__lower',
]);

// ─── Build Chart.js datasets ──────────────────────────────────────────────────
// Use two-dataset fill (+1) so the shaded region covers only the SE band,
// not the full area down to y=0.
function buildDatasets(seriesMap, bands = true) {
    const datasets = [];
    for (const [label, series] of Object.entries(seriesMap)) {
        const c = styleFor(label);
        const meanPts  = series.mean.map((m, i) => ({ x: series.x[i], y: m }));
        const upperPts = series.mean.map((m, i) => ({ x: series.x[i], y: +(m + series.std[i]).toFixed(4) }));
        const lowerPts = series.mean.map((m, i) => ({ x: series.x[i], y: Math.max(0, +(m - series.std[i]).toFixed(4)) }));

        if (bands) {
            // Upper boundary – fills DOWN to the lower boundary dataset (+1 index)
            datasets.push({
                label: `${label}__upper`,
                data: upperPts,
                fill: '+1', backgroundColor: c.fill, borderWidth: 0,
                pointRadius: 0, pointHoverRadius: 0,
                tension: 0.3, order: 10,
            });
            // Lower boundary – just the boundary line, no fill
            datasets.push({
                label: `${label}__lower`,
                data: lowerPts,
                fill: false, borderWidth: 0,
                pointRadius: 0, pointHoverRadius: 0,
                tension: 0.3, order: 10,
            });
        }
        datasets.push({
            label,
            data: meanPts,
            fill: false, borderColor: c.line, backgroundColor: c.line,
            borderWidth: c.width, borderDash: c.dash || [],
            pointRadius: 0, pointHoverRadius: 5, tension: 0.3, order: 1,
        });
    }
    return datasets;
}

// ─── HTML/SVG legend (reliable dash/dotted rendering) ────────────────────────
// Chart.js 4.x drawPoint('line') ignores borderDash, so we render the legend
// as HTML with SVG lines using stroke-dasharray, which maps 1-to-1 to canvas.

// Sort order for transfer: Row 1 = MPAIL2 + BC (Transferred), Row 2 = [−P] (MAIRL) + BC (From Scratch)
function legendSortKey(label) {
    if (label.startsWith('MPAIL2'))           return 0;
    if (label === 'BC (Transferred)')         return 0;
    if (label.startsWith('[−P] (MAIRL)'))     return 1;
    if (label === 'BC (From Scratch)')        return 1;
    return 2;
}

// Explicit ordering for efficiency legend
const EFFICIENCY_LEGEND_ORDER = ['MPAIL2', '[−P] (MAIRL)', '[−PM] (DAC)', 'RLPD'];

function renderHtmlLegend(containerId, seriesMap, mode = 'transfer') {
    const el = document.getElementById(containerId);
    if (!el) return;
    const isDark = document.body.classList.contains('dark-mode');
    const txtColor = isDark ? '#cbd5e1' : '#374151';

    let entries = Object.entries(seriesMap)
        .filter(([k]) => !k.includes('__') && !LEGEND_HIDDEN.has(k));

    if (mode === 'efficiency') {
        // Sort by explicit order; unlisted entries go at the end
        entries = entries.sort(([a], [b]) => {
            const ia = EFFICIENCY_LEGEND_ORDER.indexOf(a);
            const ib = EFFICIENCY_LEGEND_ORDER.indexOf(b);
            return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
        });
    } else {
        entries = entries.sort(([a], [b]) => legendSortKey(a) - legendSortKey(b));
    }

    // Insert a row-break between MPAIL2 group and the rest
    let html = '';
    let prevGroup = null;
    for (const [label] of entries) {
        const group = legendSortKey(label);
        // Insert row break after MPAIL2 group in transfer mode only
        if (mode === 'transfer' && prevGroup === 0 && group > 0) {
            html += `<div style="flex-basis:100%;height:0"></div>`;
        }
        prevGroup = group;
        const c = styleFor(label);
        const da = c.dash.length ? c.dash.join(',') : 'none';
        html += `<div class="chart-legend-item" style="color:${txtColor}">
            <svg width="28" height="10" style="flex-shrink:0;display:block">
                <line x1="2" y1="5" x2="26" y2="5"
                      stroke="${c.line}" stroke-width="${Math.max(c.width, 1.5)}"
                      stroke-dasharray="${da}" stroke-linecap="round"/>
            </svg>
            <span>${label}</span>
        </div>`;
    }
    el.innerHTML = `<div class="chart-html-legend">${html}</div>`;
}

// ─── "New Task" vertical line plugin ─────────────────────────────────────────
const newTaskLinePlugin = {
    id: 'newTaskLine',
    afterDraw(chart) {
        const xVal = chart.options._newTaskX;
        if (xVal == null) return;
        const { ctx, chartArea, scales } = chart;
        const xPx = scales.x.getPixelForValue(xVal);
        if (xPx < chartArea.left || xPx > chartArea.right) return;
        ctx.save();
        ctx.setLineDash([8, 4]);
        ctx.strokeStyle = 'rgba(60,60,60,0.6)';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(xPx, chartArea.top); ctx.lineTo(xPx, chartArea.bottom); ctx.stroke();
        ctx.setLineDash([]);
        if (chart.options._newTaskLabel !== false) {
            ctx.fillStyle = 'rgba(60,60,60,0.8)';
            ctx.font = '700 14px Inter, sans-serif';
            ctx.save();
            ctx.translate(xPx + 5, (chartArea.top + chartArea.bottom) / 2);
            ctx.rotate(Math.PI / 2);
            ctx.textAlign = 'center';
            ctx.fillText('New Task', 0, 0);
            ctx.restore();
        }
        ctx.restore();
    },
};
Chart.register(newTaskLinePlugin);

// ─── Axis break mark plugin ───────────────────────────────────────────────────
// Draws two diagonal slash pairs at left and right edges of the chart boundary.
// _breakEdge: 'top' → marks at chartArea.top; 'bottom' → marks at chartArea.bottom
const axisBreakMarkPlugin = {
    id: 'axisBreakMark',
    afterDraw(chart) {
        const edge = chart.options._breakEdge;
        if (!edge) return;
        const { ctx, chartArea } = chart;
        const baseY = edge === 'bottom' ? chartArea.bottom : chartArea.top;
        const hw = 9, hh = 5, gap = 7;

        const isDark = document.body.classList.contains('dark-mode');
        const slashColor = isDark ? 'rgba(200,200,200,0.9)' : 'rgba(60,60,60,0.8)';
        // Use a thick stroke in the background color to "erase" the axis spine first,
        // then draw the thin colored slash on top.
        const bgColor = isDark ? '#0d1630' : '#ffffff';

        [chartArea.left, chartArea.right].forEach(cx => {
            for (const dy of [-gap / 2, gap / 2]) {
                const cy = baseY + dy;
                ctx.save();
                // Erase the spine with a background-colored stroke
                ctx.strokeStyle = bgColor;
                ctx.lineWidth = 4;
                ctx.beginPath(); ctx.moveTo(cx - hw, cy + hh); ctx.lineTo(cx + hw, cy - hh); ctx.stroke();
                // Draw the colored slash
                ctx.strokeStyle = slashColor;
                ctx.lineWidth = 1.8;
                ctx.beginPath(); ctx.moveTo(cx - hw, cy + hh); ctx.lineTo(cx + hw, cy - hh); ctx.stroke();
                ctx.restore();
            }
        });
    },
};
Chart.register(axisBreakMarkPlugin);

// ─── Custom interaction mode: match by true x VALUE, not array index ──────────
// The default 'index' mode matches points by their array index. Our transfer
// chart has an init-phase series (x: 0→N) and a transfer series (x: N→2N) that
// share array indices but span different x ranges, so 'index' picks the wrong
// points. This mode converts the cursor pixel to a data-x value and, per dataset,
// returns the element whose parsed.x is closest to it.
Chart.Interaction.modes.xValue = function (chart, e, options) {
    const xScale = chart.scales.x;
    if (!xScale) return [];
    const cursorX = xScale.getValueForPixel(e.x);
    if (cursorX == null || Number.isNaN(cursorX)) return [];
    const items = [];
    chart.data.datasets.forEach((dataset, datasetIndex) => {
        const meta = chart.getDatasetMeta(datasetIndex);
        if (meta.hidden) return;
        const elements = meta.data;
        if (!elements || !elements.length) return;
        // Parsed data values live in meta._parsed (pixel coords are on the element)
        const parsed = meta._parsed || [];
        let bestIdx = -1, bestDist = Infinity;
        for (let i = 0; i < elements.length; i++) {
            const px = parsed[i] ? parsed[i].x : elements[i].$context?.parsed?.x;
            if (px == null) continue;
            const d = Math.abs(px - cursorX);
            if (d < bestDist) { bestDist = d; bestIdx = i; }
        }
        if (bestIdx >= 0) {
            items.push({ element: elements[bestIdx], datasetIndex, index: bestIdx });
        }
    });
    return items;
};

// Common y-axis width and right padding for top/bottom transfer charts —
// both must be identical so chartArea.left and chartArea.right line up.
const TRANSFER_Y_WIDTH    = 72;
const TRANSFER_PAD_RIGHT  = 18;

// ─── Custom HTML tooltip (SVG line indicators) ───────────────────────────────
let _tooltipEl = null;

function getTooltipEl() {
    if (!_tooltipEl) {
        _tooltipEl = document.createElement('div');
        _tooltipEl.id = 'chart-custom-tooltip';
        document.body.appendChild(_tooltipEl);
    }
    return _tooltipEl;
}

function buildExternalTooltip({ chart, tooltip }) {
    const el = getTooltipEl();

    if (tooltip.opacity === 0) {
        el.style.opacity = '0';
        return;
    }

    // Track true mouse x in canvas CSS pixels via a one-time mousemove listener
    if (!chart.canvas._tooltipListenerAdded) {
        chart.canvas._tooltipListenerAdded = true;
        chart.canvas.addEventListener('mousemove', evt => {
            const rect = chart.canvas.getBoundingClientRect();
            chart.canvas._mouseX = evt.clientX - rect.left;
        });
    }

    const isDark = document.body.classList.contains('dark-mode');

    const TOOLTIP_ORDER = ['MPAIL2', '[−P] (MAIRL)', '[−PM] (DAC)', 'RLPD', 'BC'];
    function tooltipSortKey(label) {
        const idx = TOOLTIP_ORDER.findIndex(p => label.startsWith(p));
        return idx === -1 ? 99 : idx;
    }

    const newTaskX = chart.options._newTaskX ?? null;

    // True cursor x in data space (from the raw mousemove pixel). The custom
    // 'xValue' interaction mode already matches points by x, so parsed.x values
    // are correct; we only need the cursor's data-x to decide which side of the
    // "New Task" divider the user is hovering.
    const rawMouseX = chart.canvas._mouseX ?? null;
    const hoverX = (rawMouseX != null && chart.scales?.x)
        ? chart.scales.x.getValueForPixel(rawMouseX)
        : null;
    const hasSplit = newTaskX != null && hoverX != null;
    const isLeftHover = hasSplit && hoverX < newTaskX;

    // Map hidden init/bc series to display names for the tooltip
    const INIT_DISPLAY = {
        '_init_mpail2':      'MPAIL2',
        '_init_[-p]_mairl':  '[−P] (MAIRL)',
        '_bc_real':          'BC',
    };

    const items = (tooltip.dataPoints || []).filter(dp => {
        const lbl = dp.dataset.label || '';
        if (lbl.includes('__')) return false;
        const isInit   = lbl in INIT_DISPLAY;
        const isHidden = LEGEND_HIDDEN.has(lbl);
        if (hasSplit) {
            if (isLeftHover) {
                // Left side: show only init/bc helpers
                if (!isInit) return false;
            } else {
                // Right side: show only post-transfer series
                if (isInit || isHidden) return false;
            }
        } else {
            if (isHidden) return false;
        }
        return true;
    }).sort((a, b) => {
        const la = INIT_DISPLAY[a.dataset.label] ?? a.dataset.label ?? '';
        const lb = INIT_DISPLAY[b.dataset.label] ?? b.dataset.label ?? '';
        return tooltipSortKey(la) - tooltipSortKey(lb);
    });

    if (!items.length) { el.style.opacity = '0'; return; }

    // Display the actual hovered x value (rounded)
    const stepX = hoverX != null ? Math.round(hoverX) : items[0].parsed.x;

    const rows = items.map(dp => {
        const rawLabel = dp.dataset.label || '';
        const label = INIT_DISPLAY[rawLabel] ?? rawLabel;
        const c = styleFor(label);
        const da = c.dash.length ? c.dash.join(',') : 'none';
        return `<div class="ctt-row">
            <svg width="26" height="10" style="flex-shrink:0;display:block">
                <line x1="1" y1="5" x2="25" y2="5"
                      stroke="${c.line}" stroke-width="${Math.max(c.width, 1.8)}"
                      stroke-dasharray="${da}" stroke-linecap="round"/>
            </svg>
            <span class="ctt-label">${label}</span>
            <span class="ctt-value">${dp.parsed.y.toFixed(1)}</span>
        </div>`;
    }).join('');

    el.className = isDark ? 'dark' : '';
    el.innerHTML = `<div class="ctt-title">Iteration: ${stepX}</div>${rows}`;

    // Position: right of cursor, flip left if near viewport edge
    const rect  = chart.canvas.getBoundingClientRect();
    const cx    = rect.left + (rawMouseX != null ? rawMouseX : tooltip.caretX);
    const cy    = rect.top  + tooltip.caretY;
    const APPROX_W = 260;
    const OFFSET   = 14;

    const left = (cx + OFFSET + APPROX_W > window.innerWidth - 8)
        ? cx - APPROX_W - OFFSET
        : cx + OFFSET;

    el.style.left    = `${left}px`;
    el.style.top     = `${Math.max(8, cy - 40)}px`;
    el.style.opacity = '1';
}

// ─── Tooltip options ──────────────────────────────────────────────────────────
function tooltipOpts() {
    return {
        enabled: false,
        external: buildExternalTooltip,
    };
}

// ─── Create a standard (single) chart ────────────────────────────────────────
// draws each (mean) line's own label at its right end, coloured to match — used by the aggregate view
// initial reveal: instead of the default zoom/grow-from-baseline, wipe the lines in left→right by
// clipping the datasets layer to a width that eases 0→full. Only the data layer is clipped (axes,
// grid and titles stay put). Enabled per-chart via options._grow.
const GROW_MS = 850;
const growHorizontalPlugin = {
    id: 'growHorizontal',
    beforeDatasetsDraw(chart) {
        if (!chart.options._grow) return;
        const a = chart.chartArea; if (!a) return;
        if (chart._growT0 == null) chart._growT0 = performance.now();
        const t = Math.min(1, (performance.now() - chart._growT0) / GROW_MS);
        const p = 1 - Math.pow(1 - t, 3);                  // easeOutCubic
        chart._growP = t;
        const ctx = chart.ctx;
        ctx.save(); chart._growSaved = true;
        ctx.beginPath();
        ctx.rect(a.left - 1, a.top - 2, (a.right - a.left) * p + 1, a.bottom - a.top + 4);
        ctx.clip();
        if (t < 1) requestAnimationFrame(() => { try { chart.draw(); } catch (e) {} });
    },
    afterDatasetsDraw(chart) {
        if (chart._growSaved) { chart.ctx.restore(); chart._growSaved = false; }
    },
};
Chart.register(growHorizontalPlugin);

const inlineLabelPlugin = {
    id: 'inlineLabels',
    afterDatasetsDraw(chart) {
        if (!chart.options._inlineLabels) return;
        if (chart._growP != null && chart._growP < 0.999) return;   // hold labels until the lines finish drawing
        const ctx = chart.ctx;
        const placed = [];                                  // [y] already used, to dodge overlaps
        chart.data.datasets.forEach((ds, i) => {
            if (!ds.label || ds.label.includes('__')) return;   // skip SE-band datasets
            const meta = chart.getDatasetMeta(i);
            if (!meta || meta.hidden || !meta.data || !meta.data.length) return;
            const pt = meta.data[meta.data.length - 1];
            if (!pt) return;
            let y = pt.y;
            while (placed.some(p => Math.abs(p - y) < 13)) y -= 13;   // nudge up if it collides
            placed.push(y);
            ctx.save();
            ctx.font = '700 11px Inter, sans-serif';
            ctx.fillStyle = ds.borderColor || '#333';
            ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
            ctx.fillText(ds.label, pt.x + 7, y);
            ctx.restore();
        });
    }
};

function createResultChart(canvasId, legendContainerId, title, seriesMap, bands = true, xTitle = 'Number of Iterations', inlineLabels = false, newTaskX = null, xRange = null) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    const isDark    = document.body.classList.contains('dark-mode');
    const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)';
    const tickColor = isDark ? '#94a3b8' : '#6b7280';
    const txtColor  = isDark ? '#e2e8f0' : '#1e293b';

    const chart = new Chart(canvas, {
        type: 'line',
        data: { datasets: buildDatasets(seriesMap, bands) },
        plugins: inlineLabels ? [inlineLabelPlugin] : [],
        options: {
            responsive: true,
            animation: false,                                          // default zoom off; grow plugin wipes L→R
            interaction: { mode: 'xValue', intersect: false },
            layout: inlineLabels ? { padding: { right: 116 } } : {},   // room for the on-line labels
            _inlineLabels: inlineLabels,
            _grow: true,
            _newTaskX: newTaskX,                                       // vertical "New Task" separator (transfer)
            plugins: {
                title: { display: true, text: title, color: txtColor,
                    font: { size: 14, weight: '600', family: 'Inter, sans-serif' }, padding: { bottom: 10 } },
                legend: { display: false },
                tooltip: tooltipOpts(),
            },
            scales: {
                x: {
                    type: 'linear',
                    ...(xRange ? { min: xRange.min, max: xRange.max } : {}),
                    title: { display: true, text: xTitle,
                        color: tickColor, font: { family: 'Inter, sans-serif', size: 11 } },
                    grid: { color: gridColor },
                    ticks: { color: tickColor, font: { family: 'Inter, sans-serif' } },
                },
                y: {
                    title: { display: true, text: 'Cumulative Success Count',
                        color: tickColor, font: { family: 'Inter, sans-serif', size: 11 } },
                    grid: { color: gridColor },
                    ticks: { color: tickColor, font: { family: 'Inter, sans-serif' } },
                    min: 0,
                },
            },
        },
    });
    if (inlineLabels) { const el = document.getElementById(legendContainerId); if (el) el.innerHTML = ''; }   // labels live on the lines
    else renderHtmlLegend(legendContainerId, seriesMap, 'efficiency');
    return chart;
}

// ─── Create broken-axis transfer chart (two stacked canvases) ─────────────────
function createTransferBrokenChart(trData, newTaskX) {
    const topCanvas = document.getElementById('transfer-chart-top');
    const botCanvas = document.getElementById('transfer-chart-bottom');
    if (!topCanvas || !botCanvas) return [null, null];

    const isDark    = document.body.classList.contains('dark-mode');
    const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)';
    const tickColor = isDark ? '#94a3b8' : '#6b7280';
    const txtColor  = isDark ? '#e2e8f0' : '#1e293b';

    const task    = (window.currentTask === 'pick-place') ? 'pick' : 'push';
    const trTitle = (task === 'pick') ? 'Transfer: Pick and Place' : 'Transfer: Block Push';
    const meta    = chartData?.transfer?.meta ?? {};

    // Per-task axis config
    const xMax       = (task === 'pick') ? (meta.x_max_pick ?? 400)  : (meta.x_max_push ?? 600);
    const brokenCfg  = (task === 'pick') ? meta.broken_pick           : meta.broken_push;
    const yPickRange = meta.y_pick ?? [-2, 65];

    const xRange = { min: 0, max: xMax };
    const forceYWidth = scale => { scale.width = TRANSFER_Y_WIDTH; };

    const topSeries = {};
    const botSeries = {};
    for (const [k, v] of Object.entries(trData)) {
        if (k === '_bc_real') topSeries[k] = v;
        else botSeries[k] = v;
    }

    // If no broken axis config for this task (PnP), hide top strip and use single chart
    const useBroken = !!brokenCfg;
    topCanvas.parentElement.style.display = useBroken ? '' : 'none';

    // Top strip y range (broken axis) or dummy
    const topY = brokenCfg ? brokenCfg.top_y : [0, 1];
    const topTicks = brokenCfg ? brokenCfg.top_y_ticks ?? [] : [];
    // Bottom panel y range
    const botY = brokenCfg ? brokenCfg.bot_y : yPickRange;

    // ── Top strip: BC baseline ─────────────────────────────────────────────
    const topChart = new Chart(topCanvas, {
        type: 'line',
        data: { datasets: buildDatasets(useBroken ? topSeries : {}) },
        options: {
            _newTaskX: newTaskX,
            _newTaskLabel: false,
            _breakEdge: 'bottom',
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { right: TRANSFER_PAD_RIGHT } },
            interaction: { mode: 'xValue', intersect: false },
            plugins: {
                title: { display: true, text: trTitle, color: txtColor,
                    font: { size: 14, weight: '600', family: 'Inter, sans-serif' }, padding: { bottom: 8 } },
                legend: { display: false },
                tooltip: { enabled: false },
            },
            scales: {
                x: { type: 'linear', ...xRange, display: false },
                y: {
                    min: topY[0], max: topY[1],
                    afterFit: forceYWidth,
                    grid: { color: gridColor },
                    ticks: {
                        color: tickColor, font: { family: 'Inter, sans-serif', size: 10 },
                        callback: v => topTicks.includes(v) ? v : '',
                        maxTicksLimit: 3,
                    },
                    title: { display: false },
                },
            },
        },
    });

    // ── Bottom main panel ─────────────────────────────────────────────────
    // Always include bc_real in the bottom datasets so its value appears in the
    // tooltip when hovering the left (init) side. In broken mode it sits above
    // the bottom panel's max, so Chart.js clips it and no extra line is drawn.
    const botDatasets = buildDatasets({ ...topSeries, ...botSeries });
    const botChart = new Chart(botCanvas, {
        type: 'line',
        data: { datasets: botDatasets },
        options: {
            _newTaskX: newTaskX,
            _breakEdge: useBroken ? 'top' : null,
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { right: TRANSFER_PAD_RIGHT } },
            interaction: { mode: 'xValue', intersect: false },
            plugins: {
                title: { display: !useBroken, text: trTitle, color: txtColor,
                    font: { size: 14, weight: '600', family: 'Inter, sans-serif' }, padding: { bottom: 8 } },
                legend: { display: false },
                tooltip: tooltipOpts(),
            },
            scales: {
                x: {
                    type: 'linear', ...xRange,
                    title: { display: true, text: 'Number of Iterations',
                        color: tickColor, font: { family: 'Inter, sans-serif', size: 11 } },
                    grid: { color: gridColor },
                    ticks: { color: tickColor, font: { family: 'Inter, sans-serif' } },
                },
                y: {
                    min: botY[0], max: botY[1],
                    afterFit: forceYWidth,
                    title: { display: true, text: 'Cumulative Success Count',
                        color: tickColor, font: { family: 'Inter, sans-serif', size: 11 } },
                    grid: { color: gridColor },
                    ticks: {
                        color: tickColor, font: { family: 'Inter, sans-serif' },
                    },
                },
            },
        },
    });

    // Render HTML legend (use combined series for PnP, botSeries for push)
    renderHtmlLegend('transfer-chart-legend', useBroken ? botSeries : { ...topSeries, ...botSeries });

    return [topChart, botChart];
}

// ─── State ────────────────────────────────────────────────────────────────────
let chartData           = null;
let efficiencyChart     = null;
let transferChartTop    = null;
let transferChartBottom = null;

function destroyTransferCharts() {
    if (transferChartTop)    { transferChartTop.destroy();    transferChartTop    = null; }
    if (transferChartBottom) { transferChartBottom.destroy(); transferChartBottom = null; }
}

async function initCharts() {
    if (chartData) return;
    try {
        const resp = await fetch('Media/Plots/chart_data.json');
        chartData = await resp.json();
        renderCharts();
    } catch (e) {
        console.error('Failed to load chart data:', e);
    }
}

function renderCharts() {
    if (!chartData) return;

    if (efficiencyChart) { efficiencyChart.destroy(); efficiencyChart = null; }
    destroyTransferCharts();

    const task      = (window.currentTask === 'pick-place') ? 'pick'
                    : (window.currentTask === 'mug-on-plate') ? 'mop'
                    : (window.currentTask === 'vid-bp') ? 'vid'
                    : 'push';
    const taskLabel = (task === 'pick') ? 'Pick and Place'
                    : (task === 'mop') ? 'Mug on Plate'
                    : (task === 'vid') ? 'Block Push (Video-only)'
                    : 'Block Push';

    const effData  = chartData.efficiency?.[task];
    const trData   = (task !== 'mop' && task !== 'vid') ? chartData.transfer?.[task] : null;
    const newTaskX = (task === 'pick')
        ? (chartData.transfer?.meta?.new_task_x_pick ?? null)
        : (chartData.transfer?.meta?.new_task_x_push ?? null);

    // only build the Training/Transfer charts when their (Undirected-only) section is actually visible —
    // creating a Chart on a display:none / 0-size canvas can throw and abort the rest of this function
    const visible = (id) => { const el = document.getElementById(id); return !!el && el.offsetWidth > 0; };
    try {
        if (effData && Object.keys(effData).length > 0 && visible('efficiency-chart')) {
            efficiencyChart = createResultChart(
                'efficiency-chart', 'efficiency-chart-legend',
                `Sample Efficiency – Real: ${taskLabel}`, effData
            );
        }
        if (trData && visible('transfer-chart-bottom')) {
            [transferChartTop, transferChartBottom] = createTransferBrokenChart(trData, newTaskX);
        }
    } catch (e) { console.error('training/transfer chart render failed:', e); }

    // (re)render the focused claim's plot now that data is loaded
    if (window.resultsActiveClaim) renderClaimPlot(window.resultsActiveClaim);
}

function refreshCharts() {
    if (!chartData) return;
    renderCharts();
}

// ─── Per-claim efficiency plot (Focus mode) ──────────────────────────────────
// One dynamic plot per claim, overlaying every relevant task on a single canvas:
// method → colour, task → line dash. Efficiency-based claims only.
const CLAIM_PLOT_METHODS = {
    world:       ['MPAIL2', '[−P] (MAIRL)', '[−PM] (DAC)'],
    planning:    ['MPAIL2', '[−P] (MAIRL)'],
    supervision: ['MPAIL2', '[−P] (MAIRL)', '[−PM] (DAC)', 'RLPD'],
    video:       ['MPAIL2', '[−P] (MAIRL)'],
};
const PLOT_TASKS = [['push', 'Block Push', []], ['pick', 'Pick-and-Place', [7, 4]], ['mop', 'Mug on Plate', [3, 3]], ['vid', 'Block Push (Video)', [2, 3]]];
const AGG_GRID = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];   // % of training

// sample a series at a fraction p∈[0,1] of its own training schedule (linear interpolation)
function interpAtFrac(series, p) {
    const xs = series.x, last = xs.length - 1, xt = p * xs[last];
    if (xt <= xs[0]) return { mean: series.mean[0], std: series.std[0] };
    for (let i = 1; i <= last; i++) {
        if (xt <= xs[i]) {
            const f = (xt - xs[i - 1]) / ((xs[i] - xs[i - 1]) || 1);
            return { mean: series.mean[i - 1] + (series.mean[i] - series.mean[i - 1]) * f,
                     std:  series.std[i - 1]  + (series.std[i]  - series.std[i - 1])  * f };
        }
    }
    return { mean: series.mean[last], std: series.std[last] };
}
// The aggregate view collapses the claim to its rhetorical contrast: each line is the mean over the
// listed methods × both tasks, labelled by the claim's framing (drawn on the line). Colour comes from
// the first method in each group.
const AGG_LABELS = {
    world:       [['+ Dynamics', ['MPAIL2', '[−P] (MAIRL)']], ['No Dynamics', ['[−PM] (DAC)']]],
    planning:    [['With Planning', ['MPAIL2']], ['Without Planning', ['[−P] (MAIRL)']]],
    supervision: [['Observation-only', ['MPAIL2', '[−P] (MAIRL)', '[−PM] (DAC)']], ['Obs. + Reward + Actions', ['RLPD']]],
};
function aggregateEfficiency(key) {
    const groups = AGG_LABELS[key] || (CLAIM_PLOT_METHODS[key] || []).map(m => [m, [m]]);
    const out = {};
    groups.forEach(([label, methodList]) => {
        const series = [];
        methodList.filter(m => !claimExMethods.has(m)).forEach(m => ['push', 'pick', 'mop', 'vid'].filter(t => !claimExTasks.has(t)).forEach(t => { const eff = chartData && chartData.efficiency && chartData.efficiency[t]; if (eff && eff[m]) series.push(eff[m]); }));
        if (!series.length) return;
        const x = [], mean = [], std = [];
        AGG_GRID.forEach(p => {
            let sm = 0, ss = 0;
            series.forEach(s => { const v = interpAtFrac(s, p / 100); sm += v.mean; ss += v.std; });
            x.push(p); mean.push(+(sm / series.length).toFixed(4)); std.push(+(ss / series.length).toFixed(4));
        });
        if (!ALGO_STYLES[label]) { const base = styleFor(methodList[0]); ALGO_STYLES[label] = { line: base.line, fill: base.fill, dash: [], width: 2.5 }; }
        out[label] = { x, mean, std };
    });
    return out;
}

// interpolate a series at an absolute x value
function interpAtX(s, xt) {
    const xs = s.x, last = xs.length - 1;
    if (xt <= xs[0]) return { mean: s.mean[0], std: s.std[0] };
    for (let i = 1; i <= last; i++) {
        if (xt <= xs[i]) {
            const f = (xt - xs[i - 1]) / ((xs[i] - xs[i - 1]) || 1);
            return { mean: s.mean[i - 1] + (s.mean[i] - s.mean[i - 1]) * f, std: s.std[i - 1] + (s.std[i] - s.std[i - 1]) * f };
        }
    }
    return { mean: s.mean[last], std: s.std[last] };
}
// Transfer claim, aggregated: each method's Transferred vs From-Scratch runs, averaged over both tasks
// against the new-task training progress (%) — conveys positive online transfer in one contrast.
// [label, {push, pick} series keys, colour key (method), dash]. Solid = transferred, dashed = from scratch.
const TRANSFER_AGG = [
    ['MPAIL2 · Transferred', { push: 'MPAIL2 (Full Transfer)', pick: 'MPAIL2 (Transferred)' }, 'MPAIL2', []],
    ['MPAIL2 · From Scratch', { push: 'MPAIL2 (From Scratch)', pick: 'MPAIL2 (From Scratch)' }, 'MPAIL2', [6, 4]],
    ['[−P] (MAIRL) · Transferred', { push: '[−P] (MAIRL) (Full Transfer)', pick: '[−P] (MAIRL) (Transferred)' }, '[−P] (MAIRL)', []],
    ['[−P] (MAIRL) · From Scratch', { push: '[−P] (MAIRL) (From Scratch)', pick: '[−P] (MAIRL) (From Scratch)' }, '[−P] (MAIRL)', [6, 4]],
];
function aggregateTransfer() {
    const meta = (chartData && chartData.transfer && chartData.transfer.meta) || {};
    const out = {};
    TRANSFER_AGG.forEach(([label, keys, colorKey, dash]) => {
        const runs = [];
        ['push', 'pick'].forEach(t => {
            const tr = chartData && chartData.transfer && chartData.transfer[t];
            const s = tr && tr[keys[t]];
            if (!s) return;
            const ntx = (t === 'pick') ? (meta.new_task_x_pick ?? 0) : (meta.new_task_x_push ?? 0);
            const xmax = (t === 'pick') ? (meta.x_max_pick ?? s.x[s.x.length - 1]) : (meta.x_max_push ?? s.x[s.x.length - 1]);
            runs.push({ s, ntx, xmax });
        });
        if (!runs.length) return;
        const x = [], mean = [], std = [];
        AGG_GRID.forEach(p => {
            let sm = 0, ss = 0, n = 0;
            runs.forEach(({ s, ntx, xmax }) => { const v = interpAtX(s, ntx + (p / 100) * (xmax - ntx)); if (v) { sm += v.mean; ss += v.std; n++; } });
            if (n) { x.push(p); mean.push(+(sm / n).toFixed(4)); std.push(+(ss / n).toFixed(4)); }
        });
        if (!ALGO_STYLES[label]) { const base = styleFor(colorKey); ALGO_STYLES[label] = { line: base.line, fill: base.fill, dash: dash, width: 2.5 }; }
        out[label] = { x, mean, std };
    });
    return out;
}

// table method key → efficiency-series label, and result column → its task curve
const EFF_METHOD = { mpail2: 'MPAIL2', mairl: '[−P] (MAIRL)', dac: '[−PM] (DAC)', rlpd: 'RLPD' };
const COL_TASK = {
    'res-bp':  { task: 'push', label: 'Block Push',           dash: []     },
    'res-pnp': { task: 'pick', label: 'Pick-and-Place',       dash: [7, 4] },
    'res-mop': { task: 'mop',  label: 'Mug on Plate',         dash: [3, 3] },
    'vid-bp':  { task: 'vid',  label: 'Block Push (Video)',   dash: [2, 3] },
};
// the "Claim" (aggregated) view is only meaningful with a claim selected; grey it out otherwise
function setClaimBtnEnabled(on) {
    const b = document.querySelector('#results-plot-views .plot-view-btn[data-view="claim"]');
    if (!b) return;
    b.disabled = !on;
    b.classList.toggle('is-disabled', !on);
    b.title = on ? "The claim's aggregated curves" : 'Select a finding (claim) above to enable the aggregated view';
}

let currentClaimKey = null, currentSelection = null, plotView = 'claim', claimCharts = [];
// per-claim filters: methods/tasks the viewer has toggled OFF in the claim efficiency plots
let claimExMethods = new Set(), claimExTasks = new Set();
function clearClaimCharts() { claimCharts.forEach(c => c && c.destroy()); claimCharts = []; }
function renderClaimPlot(key) {
    if (key && key[0] !== '_') {
        if (key !== currentClaimKey) { claimExMethods.clear(); claimExTasks.clear(); }   // fresh filters per claim
        currentClaimKey = key; currentSelection = null;                                  // entering claim mode
    }
    clearClaimCharts();
    const wrap = document.getElementById('results-plot');
    const hint = document.getElementById('results-plot-hint');
    const views = document.getElementById('results-plot-views');
    const show = (on) => { if (wrap) wrap.style.display = on ? '' : 'none'; if (hint) hint.style.display = on ? 'none' : ''; };
    if (!wrap) return;
    wrap.innerHTML = '';
    const hideViews = () => { if (views) views.style.display = 'none'; };
    if (document.body.classList.contains('results-undirected')) { show(false); hideViews(); return; }
    const isTransfer = (key === 'transfer');
    const methods = CLAIM_PLOT_METHODS[key];
    if (!isTransfer && !methods) { show(false); hideViews(); return; }   // e.g. the video claim has no curves

    // Transfer (Q3): "Claim" aggregates MPAIL2's Transferred vs From-Scratch over both tasks; "By task"
    // separates the two per-task plots (Block Push, Pick-and-Place), each mirroring the Undirected view —
    // a continuous x-axis with the shared initial-task training (the _init_* curves) then the new task's
    // runs shifted after it, with a dashed "New Task" separator.
    if (isTransfer) {
        if (views) views.style.display = '';
        setClaimBtnEnabled(true);
        document.querySelectorAll('#results-plot-views .plot-view-btn').forEach(b => b.classList.toggle('is-on', b.dataset.view === plotView));

        if (plotView === 'claim') {
            const agg = aggregateTransfer();
            if (!Object.keys(agg).length) { show(false); return; }
            show(true);
            wrap.insertAdjacentHTML('beforeend',
                '<div class="results-chart-container results-chart-container--claim"><canvas id="claim-chart"></canvas></div><div id="claim-chart-legend"></div>');
            // four series (MPAIL2 + MAIRL, each transferred/from-scratch) — use a legend, not on-line
            // labels, since the labels are long and the MAIRL pair runs too close to label inline
            claimCharts.push(createResultChart('claim-chart', 'claim-chart-legend',
                'Transfer — transferred vs from scratch', agg, true, 'New-task progress (%)', false));
            return;
        }
        // by task: the two per-task transfer plots, sharing ONE legend (built from the union of series)
        const meta = (chartData && chartData.transfer && chartData.transfer.meta) || {};
        let any = false; const unionSeries = {};
        PLOT_TASKS.forEach(([task, taskLabel]) => {
            const tr = chartData && chartData.transfer && chartData.transfer[task];
            if (!tr || !Object.keys(tr).length) return;
            any = true;
            Object.keys(tr).forEach(k => { if (k[0] !== '_' && !(k in unionSeries)) unionSeries[k] = tr[k]; });
            const ntx = (task === 'pick') ? (meta.new_task_x_pick ?? null) : (meta.new_task_x_push ?? null);
            const xMax = (task === 'pick') ? (meta.x_max_pick ?? null) : (meta.x_max_push ?? null);
            const cid = 'claim-chart-' + task;
            wrap.insertAdjacentHTML('beforeend',
                '<div class="results-plot__one"><div class="results-chart-container results-chart-container--claim"><canvas id="' + cid + '"></canvas></div></div>');
            claimCharts.push(createResultChart(cid, null, 'Transfer — ' + taskLabel, tr, true,    // null legend → suppressed
                'Number of Iterations', false, ntx, xMax != null ? { min: 0, max: xMax } : null));
        });
        if (any) {
            wrap.insertAdjacentHTML('beforeend', '<div id="claim-transfer-legend" class="results-shared-legend"></div>');
            renderHtmlLegend('claim-transfer-legend', unionSeries, 'efficiency');
        }
        show(any);
        return;
    }

    // Efficiency claims: a single plot — the claim's aggregate (default) or overlaid by task.
    if (views) views.style.display = '';
    setClaimBtnEnabled(true);                                   // a claim is active, so "Claim" is available
    document.querySelectorAll('#results-plot-views .plot-view-btn').forEach(b => b.classList.toggle('is-on', b.dataset.view === plotView));
    wrap.insertAdjacentHTML('beforeend',
        '<div class="results-chart-container results-chart-container--claim"><canvas id="claim-chart"></canvas></div><div id="claim-chart-legend"></div>');

    if (plotView === 'claim') {
        const agg = aggregateEfficiency(key);
        if (!Object.keys(agg).length) { show(true); claimEmptyNote(); renderClaimControls(key); return; }
        show(true);
        claimCharts.push(createResultChart('claim-chart', 'claim-chart-legend',
            'Sample Efficiency Combined', agg, true, 'Training Progress (%)', true));
        renderClaimControls(key);
        return;
    }

    // by-task overlay: one line per (method × task); colour = method, dash = task
    const merged = {};
    PLOT_TASKS.forEach(([task, taskLabel, dash]) => {
        if (claimExTasks.has(task)) return;
        const eff = chartData && chartData.efficiency && chartData.efficiency[task];
        if (!eff) return;
        methods.forEach(m => {
            if (claimExMethods.has(m) || !eff[m]) return;
            const mLabel = m + ' · ' + taskLabel;
            if (!ALGO_STYLES[mLabel]) { const base = styleFor(m); ALGO_STYLES[mLabel] = { line: base.line, fill: base.fill, dash: dash, width: base.width }; }
            merged[mLabel] = eff[m];
        });
    });
    if (!Object.keys(merged).length) { show(true); claimEmptyNote(); renderClaimControls(key); return; }
    show(true);
    claimCharts.push(createResultChart('claim-chart', 'claim-chart-legend', 'Individual Sample Efficiency ', merged, false));
    renderClaimControls(key);
}

// when method/task filters leave nothing to plot, replace the chart with a gentle note (controls stay below)
function claimEmptyNote() {
    const c = document.querySelector('#results-plot .results-chart-container--claim');
    if (c) c.innerHTML = '<p class="claim-plot-empty">No sample-efficiency curves for the current method / task selection.</p>';
}

// Filter controls under the claim efficiency plots: toggle methods (grouped by the claim line they feed)
// and tasks in/out of the plot. In the aggregate view methods are clustered under their claim-line label
// and tinted to that line's colour, so it is clear which line each method belongs to.
function renderClaimControls(key) {
    const wrap = document.getElementById('results-plot');
    if (!wrap || !CLAIM_PLOT_METHODS[key]) return;
    const claimMethods = CLAIM_PLOT_METHODS[key];
    const groups = AGG_LABELS[key] || claimMethods.map(m => [m, [m]]);
    const hasData = (m) => PLOT_TASKS.some(([t]) => chartData && chartData.efficiency && chartData.efficiency[t] && chartData.efficiency[t][m]);
    const taskHasData = (t) => chartData && chartData.efficiency && chartData.efficiency[t] && claimMethods.some(m => chartData.efficiency[t][m]);
    const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
    const chip = (kind, id, label, col) => {
        const off = (kind === 'method' ? claimExMethods : claimExTasks).has(id);
        return '<button type="button" class="cpc-chip' + (off ? ' is-off' : '') + '" data-kind="' + kind +
            '" data-id="' + esc(id) + '" style="--cc:' + col + '" aria-pressed="' + (!off) + '">' +
            '<span class="cpc-chip__dot"></span>' + esc(label) + '</button>';
    };

    let methodsHtml = '';
    if (plotView === 'claim' && AGG_LABELS[key]) {               // cluster methods under their claim line
        groups.forEach(([glabel, mlist]) => {
            const avail = mlist.filter(hasData);
            if (!avail.length) return;
            const col = styleFor(mlist[0]).line;
            methodsHtml += '<div class="cpc-group" style="--gc:' + col + '"><span class="cpc-group__label">' + glabel + '</span>' +
                avail.map(m => chip('method', m, m, col)).join('') + '</div>';
        });
    } else {                                                     // by-task: flat, each method its own colour
        methodsHtml = claimMethods.filter(hasData).map(m => chip('method', m, m, styleFor(m).line)).join('');
    }
    const tasksHtml = PLOT_TASKS.filter(([t]) => taskHasData(t)).map(([t, tl]) => chip('task', t, tl, '#8a93a8')).join('');
    if (!methodsHtml && !tasksHtml) return;

    wrap.insertAdjacentHTML('beforeend',
        '<div class="claim-plot-controls" id="claim-plot-controls">' +
        '<div class="cpc-row"><span class="cpc-row__label">Methods</span><div class="cpc-row__items">' + methodsHtml + '</div></div>' +
        '<div class="cpc-row"><span class="cpc-row__label">Tasks</span><div class="cpc-row__items">' + tasksHtml + '</div></div>' +
        '</div>');
}
window.renderClaimPlot = renderClaimPlot;

// plot the table's hand-picked cells: one efficiency curve per selected (method × task) cell that has
// data (the BP / PnP result columns). Always shown alongside a selection; the "Claim" view is greyed.
function renderSelectionPlot(cells) {
    currentSelection = cells || [];
    clearClaimCharts();
    const wrap = document.getElementById('results-plot');
    const hint = document.getElementById('results-plot-hint');
    const views = document.getElementById('results-plot-views');
    if (!wrap) return;
    wrap.innerHTML = '';
    if (document.body.classList.contains('results-undirected')) { wrap.style.display = 'none'; if (hint) hint.style.display = 'none'; if (views) views.style.display = 'none'; return; }
    if (views) views.style.display = '';
    setClaimBtnEnabled(false);                                  // no claim → aggregated view unavailable
    plotView = 'bytask';
    document.querySelectorAll('#results-plot-views .plot-view-btn').forEach(b => b.classList.toggle('is-on', b.dataset.view === 'bytask'));

    const merged = {};
    (currentSelection || []).forEach(([m, col]) => {
        const t = COL_TASK[col], label = EFF_METHOD[m];
        if (!t || !label) return;
        const eff = chartData && chartData.efficiency && chartData.efficiency[t.task];
        if (!eff || !eff[label]) return;
        const mLabel = label + ' · ' + t.label;
        if (!ALGO_STYLES[mLabel]) { const base = styleFor(label); ALGO_STYLES[mLabel] = { line: base.line, fill: base.fill, dash: t.dash, width: base.width }; }
        merged[mLabel] = eff[label];
    });
    wrap.insertAdjacentHTML('beforeend',
        '<div class="results-chart-container results-chart-container--claim"><canvas id="claim-chart"></canvas></div><div id="claim-chart-legend"></div>');
    if (!Object.keys(merged).length) {                         // selection has no efficiency curves (e.g. only MoP/transfer)
        wrap.style.display = 'none';
        if (hint) { hint.style.display = ''; hint.textContent = 'The selected cells have no sample-efficiency curves.'; }
        return;
    }
    wrap.style.display = ''; if (hint) hint.style.display = 'none';
    claimCharts.push(createResultChart('claim-chart', 'claim-chart-legend', 'Selected Sample Efficiency', merged, false));
}
window.renderSelectionPlot = renderSelectionPlot;

// plot view toggle (Claim / By task) — delegated since the buttons load with the section
document.addEventListener('click', e => {
    const b = e.target.closest('#results-plot-views .plot-view-btn');
    if (!b || b.disabled || b.classList.contains('is-disabled')) return;
    plotView = b.dataset.view;
    if (currentSelection) renderSelectionPlot(currentSelection);
    else if (currentClaimKey) renderClaimPlot(currentClaimKey);
});

// method / task filter chips under the claim efficiency plots — toggle a series in or out
document.addEventListener('click', e => {
    const chip = e.target.closest('#claim-plot-controls .cpc-chip');
    if (!chip || !currentClaimKey || currentSelection) return;
    const kind = chip.dataset.kind, id = chip.dataset.id;
    const set = kind === 'method' ? claimExMethods : claimExTasks;
    if (!set.has(id)) {                                          // turning OFF — keep at least one of this kind active
        const items = Array.prototype.slice.call(chip.closest('.cpc-row').querySelectorAll('.cpc-chip'));
        const active = items.filter(c => !c.classList.contains('is-off')).length;
        if (active <= 1) return;
    }
    if (set.has(id)) set.delete(id); else set.add(id);
    renderClaimPlot(currentClaimKey);
});

// Re-render on theme toggle
const _origToggleTheme = window.toggleTheme;
window.toggleTheme = function () {
    if (_origToggleTheme) _origToggleTheme();
    setTimeout(refreshCharts, 50);
};

// Bootstrap after dynamic sections load
document.addEventListener('DOMContentLoaded', () => {
    const observer = new MutationObserver(() => {
        if (document.getElementById('efficiency-chart')) {
            observer.disconnect();
            initCharts();
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
});
