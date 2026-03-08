/**
 * camera-demo.js — WebRTC camera capture for live SALT inference
 *
 * Captures a frame from the user's camera, resizes to 32x32,
 * and passes to the ONNX demo for inference.
 */

(function () {
  'use strict';

  const cameraBtn = document.getElementById('cameraBtn');
  const cameraVideo = document.getElementById('cameraVideo');
  let stream = null;
  let cameraActive = false;

  if (!cameraBtn) return;

  cameraBtn.addEventListener('click', async () => {
    if (cameraActive) {
      captureFrame();
      return;
    }

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: 320, height: 320 },
      });
      cameraVideo.srcObject = stream;
      cameraVideo.style.display = 'block';
      cameraActive = true;
      cameraBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style="vertical-align:middle; margin-right:4px">
          <circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="1.5" fill="none"/>
          <circle cx="8" cy="8" r="3" fill="currentColor"/>
        </svg>
        Capture Frame
      `;
    } catch (err) {
      console.warn('Camera not available:', err.message);
      cameraBtn.textContent = 'Camera not available';
      cameraBtn.disabled = true;
    }
  });

  function captureFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext('2d');

    // Crop center square from video
    const vw = cameraVideo.videoWidth;
    const vh = cameraVideo.videoHeight;
    const size = Math.min(vw, vh);
    const sx = (vw - size) / 2;
    const sy = (vh - size) / 2;

    ctx.drawImage(cameraVideo, sx, sy, size, size, 0, 0, 32, 32);
    const imageData = ctx.getImageData(0, 0, 32, 32);

    // Stop camera
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }
    cameraVideo.style.display = 'none';
    cameraActive = false;
    cameraBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style="vertical-align:middle; margin-right:4px">
        <circle cx="8" cy="9" r="3" stroke="currentColor" stroke-width="1.5" fill="none"/>
        <path d="M2 6h2l1.5-2h5L12 6h2v8H2V6z" stroke="currentColor" stroke-width="1.5" fill="none"/>
      </svg>
      Use Camera
    `;

    // Pass to ONNX demo
    if (window.selectCameraFrame) {
      window.selectCameraFrame(imageData);
    }
  }
})();
