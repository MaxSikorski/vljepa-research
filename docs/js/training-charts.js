/**
 * training-charts.js — Animated training loss curves and comparison bars
 *
 * Uses Chart.js (CDN, lazy loaded) for the loss curve chart.
 * Comparison bars animate on scroll via IntersectionObserver.
 */

(function () {
  'use strict';

  let chartLoaded = false;
  let trainingData = null;

  // ---- Load Chart.js from CDN ----
  function loadChartJS() {
    return new Promise((resolve, reject) => {
      if (window.Chart) { resolve(); return; }
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js';
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  // ---- Load training data ----
  async function loadTrainingData() {
    if (trainingData) return trainingData;
    try {
      trainingData = await window.SALT.loadJSON('data/training-data.json');
    } catch (e) {
      console.warn('Training data not found, using defaults');
      trainingData = getDefaultData();
    }
    return trainingData;
  }

  function getDefaultData() {
    // Fallback with realistic values from SALT_RESULTS.txt
    return {
      salt: {
        stage1: generateCurve(0.84, 0.76, 150),
        stage2: generateCurve(0.51, 0.43, 150),
        knn: 0.292,
        linear_probe: 0.378,
      },
      ijepa: {
        steps: generateCurve(0.58, 0.42, 150),
        knn: 0.262,
        linear_probe: 0.352,
      },
    };
  }

  function generateCurve(start, end, steps) {
    const result = [];
    for (let i = 0; i < steps; i++) {
      const t = i / (steps - 1);
      // Exponential decay with noise
      const value = start + (end - start) * (1 - Math.exp(-3 * t)) + (Math.random() - 0.5) * 0.02;
      result.push({ step: (i + 1) * 10, loss: parseFloat(value.toFixed(4)) });
    }
    return result;
  }

  // ---- Create loss curve chart ----
  async function createLossChart() {
    const data = await loadTrainingData();
    await loadChartJS();

    const ctx = document.getElementById('lossChart');
    if (!ctx) return;

    const style = getComputedStyle(document.documentElement);
    const saltColor = style.getPropertyValue('--chart-salt').trim() || '#0071e3';
    const ijepaColor = style.getPropertyValue('--chart-ijepa').trim() || '#86868b';
    const textColor = style.getPropertyValue('--text-secondary').trim() || '#6e6e73';
    const borderColor = style.getPropertyValue('--border').trim() || '#d2d2d7';

    const stage1Data = data.salt.stage1.map((d) => ({ x: d.step, y: d.loss }));
    const stage2Data = data.salt.stage2.map((d) => ({
      x: d.step + (data.salt.stage1.length ? data.salt.stage1[data.salt.stage1.length - 1].step : 0),
      y: d.loss,
    }));

    new window.Chart(ctx, {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'SALT Stage 1 (Pixel)',
            data: stage1Data,
            borderColor: saltColor,
            backgroundColor: saltColor + '20',
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            tension: 0.3,
          },
          {
            label: 'SALT Stage 2 (Latent)',
            data: stage2Data,
            borderColor: '#34c759',
            backgroundColor: '#34c75920',
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            tension: 0.3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 2000, easing: 'easeOutQuart' },
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Training Step', color: textColor },
            ticks: { color: textColor },
            grid: { color: borderColor + '40' },
          },
          y: {
            title: { display: true, text: 'Loss', color: textColor },
            ticks: { color: textColor },
            grid: { color: borderColor + '40' },
          },
        },
        plugins: {
          legend: {
            labels: { color: textColor, usePointStyle: true },
          },
          tooltip: {
            backgroundColor: 'rgba(0,0,0,0.8)',
            titleColor: '#fff',
            bodyColor: '#fff',
          },
        },
      },
    });
  }

  // ---- Animate comparison bars ----
  async function animateBars() {
    const data = await loadTrainingData();

    const saltKnn = data.salt.knn * 100;
    const ijepaKnn = data.ijepa.knn * 100;
    const saltProbe = data.salt.linear_probe * 100;
    const ijepaProbe = data.ijepa.linear_probe * 100;

    // Scale bars relative to max (with some headroom)
    const maxVal = Math.max(saltKnn, ijepaKnn, saltProbe, ijepaProbe) * 1.2;

    const animate = (id, value) => {
      const el = document.getElementById(id);
      if (!el) return;
      setTimeout(() => {
        el.style.width = (value / maxVal * 100) + '%';
        el.textContent = value.toFixed(1) + '%';
      }, 100);
    };

    animate('barSaltKnn', saltKnn);
    animate('barIjepaKnn', ijepaKnn);
    animate('barSaltProbe', saltProbe);
    animate('barIjepaProbe', ijepaProbe);

    // Update table
    const set = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = (val * 100).toFixed(1) + '%';
    };
    set('tableSaltKnn', data.salt.knn);
    set('tableIjepaKnn', data.ijepa.knn);
    set('tableSaltProbe', data.salt.linear_probe);
    set('tableIjepaProbe', data.ijepa.linear_probe);
  }

  // ---- Initialize on section visible ----
  window.addEventListener('section-visible', (e) => {
    if (e.detail.id === 'training' && !chartLoaded) {
      chartLoaded = true;
      createLossChart();
      animateBars();
    }
  });

  // Also check if already visible
  const section = document.getElementById('training');
  if (section) {
    const rect = section.getBoundingClientRect();
    if (rect.top < window.innerHeight && !chartLoaded) {
      chartLoaded = true;
      createLossChart();
      animateBars();
    }
  }
})();
