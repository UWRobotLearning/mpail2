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
function buildDatasets(seriesMap) {
    const datasets = [];
    for (const [label, series] of Object.entries(seriesMap)) {
        const c = styleFor(label);
        const meanPts  = series.mean.map((m, i) => ({ x: series.x[i], y: m }));
        const upperPts = series.mean.map((m, i) => ({ x: series.x[i], y: +(m + series.std[i]).toFixed(4) }));
        const lowerPts = series.mean.map((m, i) => ({ x: series.x[i], y: Math.max(0, +(m - series.std[i]).toFixed(4)) }));

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
function createResultChart(canvasId, legendContainerId, title, seriesMap) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    const isDark    = document.body.classList.contains('dark-mode');
    const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)';
    const tickColor = isDark ? '#94a3b8' : '#6b7280';
    const txtColor  = isDark ? '#e2e8f0' : '#1e293b';

    const chart = new Chart(canvas, {
        type: 'line',
        data: { datasets: buildDatasets(seriesMap) },
        options: {
            responsive: true,
            interaction: { mode: 'xValue', intersect: false },
            plugins: {
                title: { display: true, text: title, color: txtColor,
                    font: { size: 14, weight: '600', family: 'Inter, sans-serif' }, padding: { bottom: 10 } },
                legend: { display: false },
                tooltip: tooltipOpts(),
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Number of Iterations',
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
    renderHtmlLegend(legendContainerId, seriesMap, 'efficiency');
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

    const task      = (window.currentTask === 'pick-place') ? 'pick' : 'push';
    const taskLabel = (task === 'pick') ? 'Pick and Place' : 'Block Push';

    const effData  = chartData.efficiency?.[task];
    const trData   = chartData.transfer?.[task];
    const newTaskX = (task === 'pick')
        ? (chartData.transfer?.meta?.new_task_x_pick ?? null)
        : (chartData.transfer?.meta?.new_task_x_push ?? null);

    if (effData && document.getElementById('efficiency-chart')) {
        efficiencyChart = createResultChart(
            'efficiency-chart', 'efficiency-chart-legend',
            `Sample Efficiency – Real: ${taskLabel}`, effData
        );
    }

    if (trData && document.getElementById('transfer-chart-bottom')) {
        [transferChartTop, transferChartBottom] = createTransferBrokenChart(trData, newTaskX);
    }
}

function refreshCharts() {
    if (!chartData) return;
    renderCharts();
}

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
