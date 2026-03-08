/**
 * pyodide-console.js — Python REPL in the browser via Pyodide (WASM)
 *
 * Lazy-loads Pyodide (~15MB) when the console section scrolls into view.
 * Pre-installs numpy and provides SALT helper functions.
 */

(function () {
  'use strict';

  let pyodide = null;
  let loading = false;
  let ready = false;

  const terminalBody = document.getElementById('terminalBody');
  const terminalOutput = document.getElementById('terminalOutput');
  const terminalInput = document.getElementById('terminalInput');
  const commandChips = document.querySelectorAll('.command-chip');

  if (!terminalInput) return;

  // ---- Load Pyodide ----
  async function loadPyodide() {
    if (pyodide || loading) return;
    loading = true;

    appendOutput('Loading Python environment...', 'output');

    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js';

    await new Promise((resolve, reject) => {
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });

    appendOutput('Initializing Pyodide (Python → WebAssembly)...', 'output');

    pyodide = await window.loadPyodide({
      indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/',
    });

    appendOutput('Installing numpy...', 'output');
    await pyodide.loadPackage('numpy');

    // Pre-run setup code
    await pyodide.runPythonAsync(`
import numpy as np

# SALT helper functions
def l1_loss(pred, target):
    """L1 loss (used in SALT Stage 2)."""
    return np.abs(pred - target).mean()

def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss (used in I-JEPA)."""
    diff = np.abs(pred - target)
    return np.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta).mean()

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def knn_predict(query, features, labels, k=5):
    """k-NN prediction using cosine similarity."""
    query_norm = query / np.linalg.norm(query)
    feat_norms = features / np.linalg.norm(features, axis=1, keepdims=True)
    sims = feat_norms @ query_norm
    top_k = np.argsort(sims)[-k:][::-1]
    top_labels = [labels[i] for i in top_k]
    return max(set(top_labels), key=top_labels.count), [(labels[i], float(sims[i])) for i in top_k]

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("\\n✓ Python ready! numpy", np.__version__)
print("  Available: l1_loss(), smooth_l1_loss(), cosine_similarity(), knn_predict()")
print("  Type any Python expression below.\\n")
`);

    ready = true;
    loading = false;
    terminalInput.disabled = false;
    terminalInput.placeholder = 'Type Python here...';
    terminalInput.focus();
  }

  // ---- Append output to terminal ----
  function appendOutput(text, type = 'output') {
    const span = document.createElement('span');
    span.className = type;
    span.textContent = text + '\n';
    terminalOutput.appendChild(span);
    terminalBody.scrollTop = terminalBody.scrollHeight;
  }

  function appendPrompt(text) {
    const span = document.createElement('span');
    span.className = 'prompt';
    span.textContent = '>>> ' + text + '\n';
    terminalOutput.appendChild(span);
  }

  // ---- Execute Python ----
  async function executePython(code) {
    if (!pyodide || !ready) return;

    appendPrompt(code);

    try {
      // Redirect stdout
      pyodide.runPython(`
import io, sys
_stdout_capture = io.StringIO()
sys.stdout = _stdout_capture
`);

      const result = await pyodide.runPythonAsync(code);

      // Get captured stdout
      const stdout = pyodide.runPython('_stdout_capture.getvalue()');
      pyodide.runPython('sys.stdout = sys.__stdout__');

      if (stdout) {
        appendOutput(stdout.trimEnd(), 'output');
      }
      if (result !== undefined && result !== null && !stdout) {
        appendOutput(String(result), 'output');
      }
    } catch (err) {
      pyodide.runPython('sys.stdout = sys.__stdout__');
      appendOutput(err.message, 'error');
    }

    terminalBody.scrollTop = terminalBody.scrollHeight;
  }

  // ---- Input handler ----
  const history = [];
  let historyIdx = -1;

  terminalInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const code = terminalInput.value.trim();
      if (!code) return;

      history.unshift(code);
      historyIdx = -1;
      terminalInput.value = '';

      executePython(code);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (historyIdx < history.length - 1) {
        historyIdx++;
        terminalInput.value = history[historyIdx];
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIdx > 0) {
        historyIdx--;
        terminalInput.value = history[historyIdx];
      } else {
        historyIdx = -1;
        terminalInput.value = '';
      }
    }
  });

  // ---- Command chips ----
  commandChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const cmd = chip.getAttribute('data-cmd');
      if (cmd && ready) {
        executePython(cmd);
      } else if (!ready) {
        loadPyodide();
      }
    });
  });

  // ---- Load when section becomes visible ----
  window.addEventListener('section-visible', (e) => {
    if (e.detail.id === 'console') {
      loadPyodide();
    }
  });

  // Focus terminal on click
  terminalBody.addEventListener('click', () => {
    if (ready) terminalInput.focus();
  });
})();
