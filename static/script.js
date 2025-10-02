window.onload = () => {
    const patientId = "0";
    let ws = null;
    const MAX_DATA_POINTS = 300;
    const MAX_TREND_POINTS = 100;

    // --- Элементы DOM ---
    const statusEl = document.getElementById('status');
    const gaugeCoverEl = document.querySelector('.gauge-cover');
    const gaugeFillEl = document.querySelector('.gauge-fill');
    const hypoxiaStatusEl = document.getElementById('hypoxiaStatus');
    const riskFactorsEl = document.getElementById('riskFactors');
    const eventLogEl = document.getElementById('eventLog');
    const alertSound = document.getElementById('alertSound');

    // --- Инициализация графиков ---
    const createChart = (canvasId, label, color) => {
        const ctx = document.getElementById(canvasId).getContext('2d');
        const config = {
            type: 'line', data: { datasets: [{ label, borderColor: color, backgroundColor: `${color}80`, data: [], tension: 0.1, pointRadius: 0 }] },
            options: { scales: { x: { type: 'time', time: { unit: 'second' }, ticks: { display: true } }, y: { beginAtZero: false } }, animation: false, plugins: { legend: { display: false }, annotation: { annotations: {} } } }
        };
        if (canvasId === 'trendChart') { config.options.scales.x.ticks.display = false; config.options.scales.y.max = 1; config.options.scales.y.min = 0; }
        return new Chart(ctx, config);
    };

    const bpmChart = createChart('bpmChart', 'BPM', 'rgb(75, 192, 192)');
    const uterusChart = createChart('uterusChart', 'Uterus', 'rgb(255, 99, 132)');
    const trendChart = createChart('trendChart', 'Risk Trend', 'rgb(255, 159, 64)');

    // --- Функции-помощники ---
    const updateChart = (chart, timestamp, value, maxPoints) => { const data = chart.data.datasets[0].data; if (data.length > maxPoints) data.shift(); data.push({ x: timestamp, y: value }); chart.update('none'); };
    const addEventToLog = (message, type = 'info') => { const eventEl = document.createElement('div'); eventEl.textContent = `[${new Date().toLocaleTimeString()}] ${message}`; eventEl.classList.add(type); eventLogEl.prepend(eventEl); };
    const addAnnotation = (chart, timestamp, label, color) => { const annId = `ann_${Date.now()}`; chart.options.plugins.annotation.annotations[annId] = { type: 'line', xMin: timestamp, xMax: timestamp, borderColor: color, borderWidth: 2, label: { content: label, enabled: true, position: 'start', backgroundColor: color, font: { size: 10 } } }; chart.update('none'); };
    const updateGauge = (prob) => { gaugeCoverEl.textContent = `${(prob * 100).toFixed(2)}%`; gaugeFillEl.style.transform = `rotate(${-90 + (prob * 180)}deg)`; if (prob > 0.7) { hypoxiaStatusEl.textContent = 'Высокий риск. Требуется внимание!'; hypoxiaStatusEl.style.color = '#d9534f'; } else if (prob > 0.4) { hypoxiaStatusEl.textContent = 'Средний риск'; hypoxiaStatusEl.style.color = '#f0ad4e'; } else { hypoxiaStatusEl.textContent = 'Низкий риск'; hypoxiaStatusEl.style.color = '#5cb85c'; } };

    // --- Логика WebSocket ---
    function connect() {
        if (ws && ws.readyState === WebSocket.OPEN) return;
        ws = new WebSocket(`ws://${window.location.host}/ws/${patientId}`);
        ws.onopen = () => { statusEl.textContent = 'Получение данных...'; statusEl.style.color = 'green'; };
        ws.onclose = () => { statusEl.textContent = 'Переподключение...'; statusEl.style.color = 'orange'; setTimeout(connect, 3000); };
        ws.onerror = () => { statusEl.textContent = 'Ошибка соединения'; statusEl.style.color = 'red'; ws.close(); };

        ws.onmessage = (event) => {
            const { data, timestamp: msgTimestamp } = JSON.parse(event.data);
            const jsTimestamp = msgTimestamp * 1000;

            if (data.bpm_value !== undefined) updateChart(bpmChart, jsTimestamp, data.bpm_value, MAX_DATA_POINTS);
            if (data.uterus_value !== undefined) updateChart(uterusChart, jsTimestamp, data.uterus_value, MAX_DATA_POINTS);
            
            if (data.prob !== undefined) {
                updateGauge(data.prob);
                updateChart(trendChart, jsTimestamp, data.prob, MAX_TREND_POINTS);
                if (data.prob > 0.7) { addEventToLog(`Критический риск гипоксии: ${(data.prob * 100).toFixed(0)}%`, 'critical'); alertSound.play(); }
            }
            
            if (data.risk_factors && data.risk_factors.length > 0) {
                let factorsHtml = '<div class="risk-factors-title">Ключевые факторы риска:</div><ul>';
                data.risk_factors.forEach(factor => { factorsHtml += `<li>- ${factor}</li>`; });
                factorsHtml += '</ul>';
                riskFactorsEl.innerHTML = factorsHtml;
            } else {
                riskFactorsEl.innerHTML = '';
            }

            const processEvent = (eventData, ...args) => { if (eventData) { const ts = eventData.timestamp * 1000; addAnnotation(bpmChart, ts, ...args.slice(0, 2)); addEventToLog(`Обнаружена ${args[0].toLowerCase()}`, args[2]); if (args[3]) alertSound.play(); } };
            processEvent(data.deceleration, 'Децелерация', '#d9534f', 'critical', true);
            processEvent(data.tachycard, 'Тахикардия', '#f0ad4e', 'warning', true);
            processEvent(data.bradicard, 'Брадикардия', '#f0ad4e', 'warning', true);
            processEvent(data.acceleration, 'Акцелерация', '#5cb85c', 'info', false);
        };
    }
    connect();

    // --- ЛОГИКА АНАЛИЗА АРХИВА ---
    const bpmFileInput = document.getElementById('bpmFileInput');
    const uterusFileInput = document.getElementById('uterusFileInput');
    const analyzeArchiveButton = document.getElementById('analyzeArchiveButton');
    const archiveResultEl = document.getElementById('archiveResult');

    analyzeArchiveButton.addEventListener('click', async () => {
        const bpmFile = bpmFileInput.files[0];
        const uterusFile = uterusFileInput.files[0];
        if (!bpmFile || !uterusFile) { archiveResultEl.innerHTML = `<p class="highlight">Пожалуйста, выберите оба файла.</p>`; return; }
        archiveResultEl.innerHTML = `<p>Анализ запущен, пожалуйста, подождите...</p>`;
        analyzeArchiveButton.disabled = true;
        const formData = new FormData();
        formData.append('bpm_file', bpmFile);
        formData.append('uterus_file', uterusFile);
        try {
            const response = await fetch(`/api/v1/analyze_archive/${patientId}`, { method: 'POST', body: formData });
            if (!response.ok) { const err = await response.json(); throw new Error(err.detail || 'Ошибка сервера'); }
            const result = await response.json();
            displayArchiveResult(result);
        } catch (error) {
            archiveResultEl.innerHTML = `<p class="highlight">Ошибка: ${error.message}</p>`;
        } finally {
            analyzeArchiveButton.disabled = false;
        }
    });

    function displayArchiveResult(data) {
        const prob = data.final_probability;
        const riskLevel = prob > 0.7 ? "Высокий" : (prob > 0.4 ? "Средний" : "Низкий");
        const riskClass = prob > 0.7 ? "highlight" : "";
        let html = `<p class="result-title">Результаты анализа:</p>`;
        html += `<p><strong>Итоговый риск гипоксии:</strong> <span class="${riskClass}">${(prob * 100).toFixed(2)}% (${riskLevel})</span></p>`;
        html += `<p><strong>Базальный ритм:</strong> ${data.basal_rate ? data.basal_rate.toFixed(1) : 'N/A'} уд/мин</p>`;
        html += `<p><strong>Децелерации (всего):</strong> ${data.decelerations_count || 0}</p>`;
        html += `<p><strong>Акцелерации (всего):</strong> ${data.accelerations_count || 0}</p>`;
        html += `<p><strong>Вариабельность (LTV):</strong> ${data.ltv ? data.ltv.toFixed(2) : 'N/A'}</p>`;
        archiveResultEl.innerHTML = html;
    }
};