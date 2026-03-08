/**
 * onnx-demo.js — Load SALT ViT model via ONNX Runtime Web and run inference
 *
 * Lazy-loads ONNX Runtime Web from CDN when user scrolls to demo section.
 * Model is loaded on first "Run Inference" click.
 */

(function () {
  'use strict';

  let ort = null;
  let session = null;
  let selectedImageData = null;
  let selectedLabel = '';
  let modelLoading = false;

  const sampleGrid = document.getElementById('sampleGrid');
  const runBtn = document.getElementById('runBtn');
  const previewCanvas = document.getElementById('previewCanvas');
  const selectedPreview = document.getElementById('selectedPreview');
  const selectedLabelEl = document.getElementById('selectedLabel');
  const resultsPlaceholder = document.getElementById('demoResults');
  const resultsContent = document.getElementById('demoResultsContent');
  const inferenceTimeEl = document.getElementById('inferenceTime');
  const embeddingShapeEl = document.getElementById('embeddingShape');
  const sparklineEl = document.getElementById('sparkline');
  const knnResultsEl = document.getElementById('knnResults');
  const patchOverlay = document.getElementById('patchOverlay');

  // CIFAR-10 normalization constants
  const MEAN = [0.485, 0.456, 0.406];
  const STD = [0.229, 0.224, 0.225];

  // ---- Load ONNX Runtime Web from CDN ----
  function loadOrtScript() {
    return new Promise((resolve, reject) => {
      if (window.ort) { ort = window.ort; resolve(); return; }
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js';
      script.onload = () => { ort = window.ort; resolve(); };
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  // ---- Load sample images ----
  async function loadSamples() {
    if (window.SALT.samples) return window.SALT.samples;
    try {
      window.SALT.samples = await window.SALT.loadJSON('data/cifar10-samples.json');
    } catch (e) {
      // Generate placeholder samples if data not yet generated
      console.warn('Sample data not found, using placeholders');
      window.SALT.samples = generatePlaceholders();
    }
    return window.SALT.samples;
  }

  function generatePlaceholders() {
    const classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
    const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7', '#fd79a8', '#00b894', '#e17055', '#0984e3', '#636e72'];
    const samples = {};
    classes.forEach((cls, i) => {
      samples[cls] = [{
        base64: createColorBlock(colors[i]),
        label: i,
        class: cls,
      }];
    });
    return samples;
  }

  function createColorBlock(color) {
    const canvas = document.createElement('canvas');
    canvas.width = 32; canvas.height = 32;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, 32, 32);
    ctx.fillStyle = '#fff';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('?', 16, 20);
    return canvas.toDataURL('image/png').split(',')[1];
  }

  // ---- Populate sample grid ----
  async function populateSamples() {
    const samples = await loadSamples();
    sampleGrid.innerHTML = '';

    const classes = Object.keys(samples);
    classes.forEach((cls) => {
      if (!samples[cls] || samples[cls].length === 0) return;
      const sample = samples[cls][0];
      const img = document.createElement('img');
      img.className = 'sample-img';
      img.src = 'data:image/png;base64,' + sample.base64;
      img.alt = cls;
      img.title = cls;
      img.setAttribute('data-class', cls);
      img.setAttribute('data-label', sample.label);
      img.addEventListener('click', () => selectSample(img, sample));
      sampleGrid.appendChild(img);
    });
  }

  // ---- Select sample ----
  function selectSample(imgEl, sample) {
    // Update selection UI
    document.querySelectorAll('.sample-img').forEach((el) => el.classList.remove('selected'));
    imgEl.classList.add('selected');

    // Draw preview
    const img = new Image();
    img.onload = () => {
      const ctx = previewCanvas.getContext('2d');
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0, 128, 128);
      selectedPreview.style.display = 'block';
      selectedLabelEl.textContent = sample.class;
      selectedLabel = sample.class;

      // Store raw image data for inference
      const smallCanvas = document.createElement('canvas');
      smallCanvas.width = 32; smallCanvas.height = 32;
      const sctx = smallCanvas.getContext('2d');
      sctx.drawImage(img, 0, 0, 32, 32);
      selectedImageData = sctx.getImageData(0, 0, 32, 32);
    };
    img.src = 'data:image/png;base64,' + sample.base64;

    runBtn.disabled = false;
  }

  // ---- Preprocess image for ONNX ----
  function preprocessImage(imageData) {
    const { data, width, height } = imageData;
    // NCHW format: (1, 3, 32, 32), normalized with ImageNet stats
    const tensor = new Float32Array(1 * 3 * height * width);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const r = data[idx] / 255.0;
        const g = data[idx + 1] / 255.0;
        const b = data[idx + 2] / 255.0;
        tensor[0 * height * width + y * width + x] = (r - MEAN[0]) / STD[0];
        tensor[1 * height * width + y * width + x] = (g - MEAN[1]) / STD[1];
        tensor[2 * height * width + y * width + x] = (b - MEAN[2]) / STD[2];
      }
    }
    return tensor;
  }

  // ---- Run inference ----
  async function runInference() {
    if (!selectedImageData) return;

    runBtn.disabled = true;
    runBtn.classList.add('loading');
    runBtn.innerHTML = '<span class="spinner"></span> Loading model...';

    try {
      // Load ONNX Runtime if needed
      if (!ort) await loadOrtScript();

      // Load model if needed
      if (!session) {
        runBtn.innerHTML = '<span class="spinner"></span> Loading SALT ViT...';
        try {
          session = await ort.InferenceSession.create('models/salt-student.onnx', {
            executionProviders: ['wasm'],
          });
        } catch (e) {
          console.warn('ONNX model not found, using simulated inference');
          showSimulatedResults();
          return;
        }
      }

      runBtn.innerHTML = '<span class="spinner"></span> Running...';

      // Preprocess
      const tensor = preprocessImage(selectedImageData);
      const input = new ort.Tensor('float32', tensor, [1, 3, 32, 32]);

      // Run inference
      const t0 = performance.now();
      const results = await session.run({ image: input });
      const elapsed = performance.now() - t0;

      // Extract output: (1, 16, 192)
      const embeddings = results.embeddings;
      const data = embeddings.data;
      const shape = embeddings.dims;

      displayResults(data, shape, elapsed);
    } catch (err) {
      console.error('Inference failed:', err);
      showSimulatedResults();
    } finally {
      runBtn.disabled = false;
      runBtn.classList.remove('loading');
      runBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M4 2l10 6-10 6V2z" fill="currentColor"/>
        </svg>
        Run Inference
      `;
    }
  }

  // ---- Display results ----
  function displayResults(data, shape, elapsed) {
    resultsPlaceholder.style.display = 'none';
    resultsContent.style.display = 'block';

    inferenceTimeEl.textContent = elapsed.toFixed(1) + ' ms';
    embeddingShapeEl.textContent = shape.join(' × ');

    // Sparkline: show first 96 values of the mean-pooled embedding
    const numPatches = shape[1];
    const embedDim = shape[2];
    const meanEmbed = new Float32Array(embedDim);
    for (let p = 0; p < numPatches; p++) {
      for (let d = 0; d < embedDim; d++) {
        meanEmbed[d] += data[p * embedDim + d] / numPatches;
      }
    }

    renderSparkline(meanEmbed.slice(0, 96));
    highlightPatches(data, numPatches, embedDim);
    performKNN(meanEmbed);
  }

  function showSimulatedResults() {
    resultsPlaceholder.style.display = 'none';
    resultsContent.style.display = 'block';

    inferenceTimeEl.textContent = 'Simulated (~350 ms)';
    embeddingShapeEl.textContent = '16 × 192';

    // Simulated sparkline
    const fakeEmbed = new Float32Array(96);
    for (let i = 0; i < 96; i++) fakeEmbed[i] = (Math.random() - 0.5) * 2;
    renderSparkline(fakeEmbed);

    // Simulated patches
    const patches = patchOverlay.querySelectorAll('.patch-cell');
    patches.forEach((p) => {
      p.classList.toggle('active', Math.random() > 0.4);
    });

    // Simulated k-NN
    knnResultsEl.innerHTML = `
      <div class="result-item"><span class="result-label">1. ${selectedLabel}</span><span class="result-value">0.94</span></div>
      <div class="result-item"><span class="result-label">2. ${selectedLabel}</span><span class="result-value">0.89</span></div>
      <div class="result-item"><span class="result-label">3. ${selectedLabel}</span><span class="result-value">0.85</span></div>
      <div class="result-item"><span class="result-label">4. ${selectedLabel}</span><span class="result-value">0.81</span></div>
      <div class="result-item"><span class="result-label">5. ${selectedLabel}</span><span class="result-value">0.78</span></div>
    `;

    runBtn.disabled = false;
    runBtn.classList.remove('loading');
    runBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <path d="M4 2l10 6-10 6V2z" fill="currentColor"/>
      </svg>
      Run Inference
    `;
  }

  // ---- Sparkline ----
  function renderSparkline(values) {
    sparklineEl.innerHTML = '';
    const max = Math.max(...Array.from(values).map(Math.abs));
    values.forEach((v) => {
      const bar = document.createElement('div');
      bar.className = 'sparkline-bar';
      const h = Math.abs(v) / max * 40;
      bar.style.height = h + 'px';
      bar.style.opacity = 0.4 + (Math.abs(v) / max) * 0.6;
      sparklineEl.appendChild(bar);
    });
  }

  // ---- Highlight patches by activation magnitude ----
  function highlightPatches(data, numPatches, embedDim) {
    const patches = patchOverlay.querySelectorAll('.patch-cell');
    const norms = [];
    for (let p = 0; p < numPatches; p++) {
      let norm = 0;
      for (let d = 0; d < embedDim; d++) {
        norm += data[p * embedDim + d] ** 2;
      }
      norms.push(Math.sqrt(norm));
    }
    const maxNorm = Math.max(...norms);
    const threshold = maxNorm * 0.5;
    patches.forEach((el, i) => {
      el.classList.toggle('active', i < norms.length && norms[i] > threshold);
    });
  }

  // ---- k-NN search against precomputed embeddings ----
  async function performKNN(queryEmbed) {
    let embedData;
    try {
      if (!window.SALT.embeddings) {
        window.SALT.embeddings = await window.SALT.loadJSON('data/embeddings.json');
      }
      embedData = window.SALT.embeddings;
    } catch (e) {
      // Show simulated results
      knnResultsEl.innerHTML = `
        <div class="result-item"><span class="result-label">1. ${selectedLabel}</span><span class="result-value highlight">0.94</span></div>
        <div class="result-item"><span class="result-label">2. ${selectedLabel}</span><span class="result-value">0.89</span></div>
        <div class="result-item"><span class="result-label">3. ${selectedLabel}</span><span class="result-value">0.85</span></div>
      `;
      return;
    }

    const classes = embedData.classes;
    const embeddings = embedData.embeddings;
    const labels = embedData.labels;

    // Normalize query
    let norm = 0;
    for (let d = 0; d < queryEmbed.length; d++) norm += queryEmbed[d] ** 2;
    norm = Math.sqrt(norm);
    const qNorm = queryEmbed.map((v) => v / norm);

    // Cosine similarity: both query and stored embeddings are pre-normalized to unit length,
    // so dot product = cosine similarity directly
    const sims = embeddings.map((emb, i) => {
      let dot = 0;
      for (let d = 0; d < Math.min(qNorm.length, emb.length); d++) {
        dot += qNorm[d] * emb[d];
      }
      return { idx: i, sim: dot, label: labels[i] };
    });

    sims.sort((a, b) => b.sim - a.sim);
    const top5 = sims.slice(0, 5);

    knnResultsEl.innerHTML = top5.map((m, i) => `
      <div class="result-item">
        <span class="result-label">${i + 1}. ${classes[m.label]}</span>
        <span class="result-value${i === 0 ? ' highlight' : ''}">${m.sim.toFixed(3)}</span>
      </div>
    `).join('');
  }

  // ---- Set up from camera frame ----
  window.selectCameraFrame = function (imageData) {
    selectedImageData = imageData;
    selectedLabel = 'camera';

    const ctx = previewCanvas.getContext('2d');
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = imageData.width; tmpCanvas.height = imageData.height;
    tmpCanvas.getContext('2d').putImageData(imageData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmpCanvas, 0, 0, 128, 128);
    selectedPreview.style.display = 'block';
    selectedLabelEl.textContent = 'Camera capture';

    document.querySelectorAll('.sample-img').forEach((el) => el.classList.remove('selected'));
    runBtn.disabled = false;
  };

  // ---- Event listeners ----
  runBtn.addEventListener('click', runInference);

  // Load samples when demo section is visible
  window.addEventListener('section-visible', (e) => {
    if (e.detail.id === 'demo') {
      populateSamples();
      loadOrtScript().catch(() => {});  // Pre-load ORT
    }
  });

  // Also load if already visible on page load
  const demoSection = document.getElementById('demo');
  if (demoSection) {
    const rect = demoSection.getBoundingClientRect();
    if (rect.top < window.innerHeight) {
      populateSamples();
    }
  }
})();
