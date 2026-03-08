/**
 * roi-calculator.js — Interactive ROI calculator for manufacturing page
 * All calculations client-side, no backend needed
 */
(function () {
  'use strict';

  const fields = ['numMachines', 'avgJobHours', 'defectRate', 'materialCost', 'jobsPerMonth', 'hourlyRate'];
  const els = {};

  fields.forEach((id) => {
    els[id] = document.getElementById(id);
    if (els[id]) {
      els[id].addEventListener('input', calculate);
    }
  });

  function getVal(id) {
    return parseFloat(els[id]?.value) || 0;
  }

  function formatNumber(n) {
    if (n >= 1000) {
      return n.toLocaleString('en-US', { maximumFractionDigits: 0 });
    }
    return n.toFixed(n < 10 ? 1 : 0);
  }

  function calculate() {
    const machines = getVal('numMachines');
    const jobHours = getVal('avgJobHours');
    const defectPct = getVal('defectRate') / 100;
    const matCost = getVal('materialCost');
    const jobsMonth = getVal('jobsPerMonth');
    const hourly = getVal('hourlyRate');

    // Total jobs per month across all machines
    const totalJobs = machines * jobsMonth;

    // Jobs that would fail without inspection
    const failedJobs = totalJobs * defectPct;

    // With SALT: detect defects at ~15% through the job on average (early detection)
    // Instead of losing the full job, you lose only 15% of time + material
    const earlyDetectionFactor = 0.15;

    // Hours saved: failed jobs × (job hours × 85% saved per failure)
    const hoursSaved = failedJobs * jobHours * (1 - earlyDetectionFactor);

    // Material saved: failed jobs × material cost × 85% saved
    const materialSaved = failedJobs * matCost * (1 - earlyDetectionFactor);

    // Labor savings from hours saved
    const laborSaved = hoursSaved * hourly;

    // Total monthly savings
    const totalSaved = materialSaved + laborSaved;

    // Update DOM
    const hoursEl = document.getElementById('roiHoursSaved');
    const matEl = document.getElementById('roiMaterialSaved');
    const totalEl = document.getElementById('roiTotalSaved');

    if (hoursEl) hoursEl.textContent = formatNumber(hoursSaved);
    if (matEl) matEl.textContent = '$' + formatNumber(materialSaved);
    if (totalEl) totalEl.textContent = '$' + formatNumber(totalSaved);
  }

  // Run on load
  calculate();
})();
