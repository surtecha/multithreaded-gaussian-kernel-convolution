import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const [csv = 'benchmarks/data/benchmark.csv', output = 'benchmarks/charts'] = process.argv.slice(2);
const [header, ...lines] = readFileSync(csv, 'utf8').trim().split(/\r?\n/);
const columns = header.split(',');
const data = lines.map(line => Object.fromEntries(
    line.split(',').map((value, index) => [columns[index], value]),
));

const hd = data.some(row => row.width === '1920' && row.height === '1080');
const [width, height] = hd ? ['1920', '1080'] : ['512', '512'];
const rows = data.filter(row => row.width === width && row.height === height);
const series = [
    { method: 'naive_2d', label: 'Naive 2D', color: '#909ba8', threads: 1 },
    { method: 'separable', label: 'Separable', color: '#78a99b', threads: 1 },
    { method: 'threaded_2d', label: 'Threaded 2D', color: '#c99b79' },
    { method: 'separable_threaded', label: 'Threaded separable', color: '#829fc4' },
    { method: 'separable_reuse', label: 'Separable with reuse', color: '#ad94b9' },
];

const threadCount = Math.max(...rows.filter(row => row.method === 'threaded_2d')
    .map(row => Number(row.threads)));
const kernels = [3, 5, 11, 15];
for (const item of series) {
    item.threads ??= threadCount;
    item.label += ` · ${item.threads}T`;
    item.samples = kernels.map(size => {
        const row = rows.find(row => row.method === item.method &&
            Number(row.threads) === item.threads && Number(row.kernel_size) === size);
        if (!row || ![row.median_ms, row.speedup].every(value => Number(value) > 0)) {
            throw new Error(`Missing or invalid benchmark data for ${item.method}, kernel ${size}`);
        }
        return row;
    });
}

const text = (x, y, value, options = '') => `<text x="${x}" y="${y}" ${options}>${value}</text>`;
const line = (x1, y1, x2, y2, options = '') =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${options}/>`;
const svg = (title, subtitle, content) => `<svg xmlns="http://www.w3.org/2000/svg" width="780" height="370" viewBox="0 0 780 370" role="img" aria-labelledby="title">
<title id="title">${title}</title>
<rect width="780" height="370" fill="#fff"/>
<g font-family="system-ui, sans-serif" font-size="13" fill="#222b38">
${text(26, 32, title, 'font-size="22" font-weight="650"')}
${text(26, 56, subtitle)}
${content.join('\n')}
</g>
</svg>
`;

function scale(max) {
    const roughStep = max / 5;
    const power = 10 ** Math.floor(Math.log10(roughStep));
    const step = [1, 2, 2.5, 5, 10].map(n => n * power).find(n => n >= roughStep);
    const limit = step * Math.ceil(max / step);
    return { limit, ticks: Array.from({ length: Math.round(limit / step) + 1 }, (_, i) => i * step) };
}

function runtimeChart() {
    const times = series.map(item => Number(item.samples[2].median_ms));
    const { limit, ticks } = scale(Math.max(...times) * 1.08);
    const start = 205;
    const span = 500;
    const marks = [];
    for (const tick of ticks) {
        const x = start + tick / limit * span;
        marks.push(line(x, 80, x, 310, 'stroke="#d9dee6"'));
        marks.push(text(x, 332, tick, 'text-anchor="middle"'));
    }
    series.forEach((item, i) => {
        const y = 94 + i * 44;
        const length = times[i] / limit * span;
        marks.push(text(start - 12, y + 16, item.label, 'text-anchor="end"'));
        marks.push(`<rect x="${start}" y="${y}" width="${length}" height="24" rx="2" fill="${item.color}"/>`);
        marks.push(text(start + length + 8, y + 16, times[i].toFixed(2)));
    });
    marks.push(text(start + span / 2, 354, 'Median runtime (ms)', 'text-anchor="middle"'));
    return svg('Runtime by implementation',
        `${width} × ${height} RGB · 11 × 11 kernel · lower is faster`, marks);
}

function speedupChart() {
    const { limit, ticks } = scale(Math.max(...series.flatMap(item => item.samples.map(row => Number(row.speedup)))));
    const x = i => 80 + i * 140;
    const y = value => 304 - value / limit * 220;
    const marks = [];
    for (const tick of ticks) {
        marks.push(line(x(0), y(tick), x(3), y(tick), 'stroke="#d9dee6"'));
        marks.push(text(x(0) - 10, y(tick) + 4, `${tick}×`, 'text-anchor="end"'));
    }
    kernels.forEach((size, i) => {
        marks.push(line(x(i), y(0), x(i), y(limit), 'stroke="#edf0f4"'));
        marks.push(text(x(i), 328, size, 'text-anchor="middle"'));
    });
    marks.push(text((x(0) + x(3)) / 2, 351, 'Kernel width (pixels)', 'text-anchor="middle"'));
    series.forEach((item, i) => {
        const points = item.samples.map((row, k) => [x(k), y(Number(row.speedup))]);
        const style = `stroke="${item.color}" stroke-width="2"`;
        marks.push(`<polyline points="${points.map(point => point.join(',')).join(' ')}" fill="none" ${style}/>`);
        points.forEach(([cx, cy]) => marks.push(`<circle cx="${cx}" cy="${cy}" r="3.5" fill="#fff" ${style}/>`));
        const keyY = 108 + i * 27;
        marks.push(line(530, keyY - 4, 552, keyY - 4, style));
        marks.push(text(560, keyY, item.label));
    });
    return svg('Speedup by kernel size',
        `${width} × ${height} RGB · relative to single-threaded naive 2D for each kernel`, marks);
}

mkdirSync(output, { recursive: true });
writeFileSync(join(output, 'runtime.svg'), runtimeChart());
writeFileSync(join(output, 'speedup.svg'), speedupChart());
