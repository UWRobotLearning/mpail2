// ─── Palette ─────────────────────────────────────────────────────────────────
const ALGO_STYLES = {
    // Efficiency
    'MPAIL2': { line: '#F25912', fill: 'rgba(242,89,18,0.15)',  dash: [],    width: 2.5 },
    'MGAIL':  { line: '#E49BA6', fill: 'rgba(228,155,166,0.15)', dash: [],   width: 2   },
    'DAC':    { line: '#E69F00', fill: 'rgba(230,159,0,0.12)',   dash: [],   width: 2   },
    'GAIL':   { line: '#56B4E9', fill: 'rgba(86,180,233,0.12)',  dash: [],   width: 2   },
    'RLPD':   { line: '#009E73', fill: 'rgba(0,158,115,0.12)',   dash: [],   width: 2   },
    'MPAIL':  { line: '#CC79A7', fill: 'rgba(204,121,167,0.12)', dash: [],   width: 2   },

    // Transfer – MPAIL2 family (orange-red)
    'MPAIL2 (Full Transfer)':      { line: '#F25912', fill: 'rgba(242,89,18,0.18)',  dash: [],    width: 2.5 },
    'MPAIL2 (Dynamics Transfer)':  { line: '#F25912', fill: 'rgba(242,89,18,0.10)',  dash: [7,4], width: 2   },
    'MPAIL2 (From Scratch)':       { line: '#F25912', fill: 'rgba(242,89,18,0.06)',  dash: [3,5], width: 2   },
    // Transfer – [-P] (MAIRL) family (pink)
    '[-P] (MAIRL) (Full Transfer)':     { line: '#E49BA6', fill: 'rgba(228,155,166,0.15)', dash: [],    width: 2   },
    '[-P] (MAIRL) (Dynamics Transfer)': { line: '#E49BA6', fill: 'rgba(228,155,166,0.08)', dash: [7,4], width: 2   },
    '[-P] (MAIRL) (From Scratch)':      { line: '#E49BA6', fill: 'rgba(228,155,166,0.05)', dash: [3,5], width: 2   },
    // BC (grey)
    'BC (Transferred)':  { line: '#999999', fill: 'rgba(153,153,153,0.08)', dash: [],    width: 1.8 },
    'BC (From Scratch)': { line: '#999999', fill: 'rgba(153,153,153,0.05)', dash: [3,5], width: 1.8 },
    // Init-phase helpers: shown on chart, hidden from legend
    '_init_mpail2': { line: '#F25912', fill: 'rgba(242,89,18,0.18)',  dash: [], width: 2.5 },
    '_init_mgail':  { line: '#E49BA6', fill: 'rgba(228,155,166,0.15)', dash: [], width: 2  },
    '_bc_real':     { line: '#999999', fill: 'rgba(153,153,153,0.08)', dash: [], width: 1.8 },
};
const FALLBACK = { line: '#94a3b8', fill: 'rgba(148,163,184,0.10)', dash: [], width: 2 };

function styleFor(label) { return ALGO_STYLES[label] || FALLBACK; }

// Labels shown on chart but excluded from legend
const LEGEND_HIDDEN = new Set([
    '_init_mpail2', '_init_mgail', '_bc_real',
    '_init_mpail2__band', '_init_mgail__band', '_bc_real__band',
]);

// ─── Build Chart.js datasets ──────────────────────────────────────────────────
function buildDatasets(seriesMap) {
    const datasets = [];
    for (const [label, series] of Object.entries(seriesMap)) {
        const c = styleFor(label);
        const meanPts  = series.mean.map((m, i) => ({ x: series.x[i], y: m }));
        const upperPts = series.mean.map((m, i) => ({ x: series.x[i], y: +(m + series.std[i]).toFixed(4) }));
        const lowerPts = series.mean.map((m, i) => ({ x: series.x[i], y: +(m - series.std[i]).toFixed(4) }));

        datasets.push({
            label: `${label}__band`,
            data: [...upperPts, ...lowerPts.slice().reverse()],
            fill: true, backgroundColor: c.fill, borderWidth: 0,
            pointRadius: 0, tension: 0.3, order: 10,
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
function renderHtmlLegend(containerId, seriesMap) {
    const el = document.getElementById(containerId);
    if (!el) return;
    const isDark = document.body.classList.contains('dark-mode');
    const txtColor = isDark ? '#cbd5e1' : '#374151';

    const entries = Object.entries(seriesMap)
        .filter(([k]) => !k.includes('__') && !LEGEND_HIDDEN.has(k));

    el.innerHTML = `<div class="chart-html-legend">
        ${entries.map(([label]) => {
            const c = styleFor(label);
            const da = c.dash.length ? c.dash.join(',') : 'none';
            return `<div class="chart-legend-item" style="color:${txtColor}">
                <svg width="28" height="10" style="flex-shrink:0;display:block">
                    <line x1="2" y1="5" x2="26" y2="5"
                          stroke="${c.line}" stroke-width="${Math.max(c.width, 1.5)}"
                          stroke-dasharray="${da}" stroke-linecap="round"/>
                </svg>
                <span>${label}</span>
            </div>`;
        }).join('')}
    </div>`;
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
            ctx.font = '700 11px Inter, sans-serif';
            ctx.save();
            ctx.translate(xPx + 12, (chartArea.top + chartArea.bottom) / 2);
            ctx.rotate(-Math.PI / 2);
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

// Common y-axis width and right padding for top/bottom transfer charts —
// both must be identical so chartArea.left and chartArea.right line up.
const TRANSFER_Y_WIDTH    = 72;
const TRANSFER_PAD_RIGHT  = 18;

// ─── Tooltip options ──────────────────────────────────────────────────────────
function tooltipOpts() {
    return {
        filter: item => {
            const lbl = item.dataset.label || '';
            return !lbl.includes('__') && !LEGEND_HIDDEN.has(lbl);
        },
        callbacks: {
            title: ctx => `Step: ${ctx[0]?.parsed.x}×10²`,
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}`,
        },
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
            interaction: { mode: 'index', intersect: false },
            plugins: {
                title: { display: true, text: title, color: txtColor,
                    font: { size: 14, weight: '600', family: 'Inter, sans-serif' }, padding: { bottom: 10 } },
                legend: { display: false },
                tooltip: tooltipOpts(),
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Number of Environment Steps (×10²)',
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
    renderHtmlLegend(legendContainerId, seriesMap);
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

    const topSeries = {};
    const botSeries = {};
    for (const [k, v] of Object.entries(trData)) {
        if (k === '_bc_real') topSeries[k] = v;
        else botSeries[k] = v;
    }

    const xRange = { min: 0, max: 600 };
    // Force identical left padding on both charts so the y-axis lines & break marks align
    const forceYWidth = scale => { scale.width = TRANSFER_Y_WIDTH; };

    // ── Top strip: BC baseline (~141) ─────────────────────────────────────────
    const topChart = new Chart(topCanvas, {
        type: 'line',
        data: { datasets: buildDatasets(topSeries) },
        options: {
            _newTaskX: newTaskX,
            _newTaskLabel: false,
            _breakEdge: 'bottom',
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { right: TRANSFER_PAD_RIGHT } },
            interaction: { mode: 'index', intersect: false },
            plugins: {
                title: { display: true, text: 'Transfer: Block Push', color: txtColor,
                    font: { size: 14, weight: '600', family: 'Inter, sans-serif' }, padding: { bottom: 8 } },
                legend: { display: false },
                tooltip: { enabled: false },
            },
            scales: {
                x: { type: 'linear', ...xRange, display: false },
                y: {
                    min: 138, max: 144,
                    afterFit: forceYWidth,
                    grid: { color: gridColor },
                    ticks: {
                        color: tickColor, font: { family: 'Inter, sans-serif', size: 10 },
                        callback: v => (v === 140 || v === 142) ? v : '',
                        maxTicksLimit: 3,
                    },
                    title: { display: false },
                },
            },
        },
    });

    // ── Bottom main panel ─────────────────────────────────────────────────────
    const botChart = new Chart(botCanvas, {
        type: 'line',
        data: { datasets: buildDatasets(botSeries) },
        options: {
            _newTaskX: newTaskX,
            _breakEdge: 'top',
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { right: TRANSFER_PAD_RIGHT } },
            interaction: { mode: 'index', intersect: false },
            plugins: {
                title: { display: false },
                legend: { display: false },
                tooltip: tooltipOpts(),
            },
            scales: {
                x: {
                    type: 'linear', ...xRange,
                    title: { display: true, text: 'Number of Environment Steps (×10²)',
                        color: tickColor, font: { family: 'Inter, sans-serif', size: 11 } },
                    grid: { color: gridColor },
                    ticks: { color: tickColor, font: { family: 'Inter, sans-serif' } },
                },
                y: {
                    min: -2, max: 80,
                    afterFit: forceYWidth,
                    title: { display: true, text: 'Cumulative Success Count',
                        color: tickColor, font: { family: 'Inter, sans-serif', size: 11 } },
                    grid: { color: gridColor },
                    ticks: {
                        color: tickColor, font: { family: 'Inter, sans-serif' },
                        callback: v => [0, 25, 50, 75].includes(v) ? v : '',
                    },
                },
            },
        },
    });

    // Render the shared HTML legend below the bottom chart
    renderHtmlLegend('transfer-chart-legend', botSeries);

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

    const effData  = chartData.efficiency?.push;
    const trData   = chartData.transfer?.push;
    const newTaskX = chartData.transfer?.meta?.new_task_x ?? null;

    if (effData && document.getElementById('efficiency-chart')) {
        efficiencyChart = createResultChart(
            'efficiency-chart', 'efficiency-chart-legend',
            'Sample Efficiency – Block Push', effData
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
