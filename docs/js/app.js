/**
 * app.js — Main orchestrator for SALT showcase
 * Theme toggle, scroll animations, stat counters, lazy section loading
 */

(function () {
  'use strict';

  // ---- Theme Toggle ----
  const toggle = document.getElementById('themeToggle');
  const html = document.documentElement;

  // Persist theme preference
  const saved = localStorage.getItem('theme');
  if (saved) {
    html.setAttribute('data-theme', saved);
  } else if (window.matchMedia('(prefers-color-scheme: light)').matches) {
    html.setAttribute('data-theme', 'light');
  }

  toggle.addEventListener('click', () => {
    const current = html.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
  });

  // ---- Animated Stat Counters ----
  function animateCounter(el) {
    const text = el.getAttribute('data-text');
    if (text) {
      el.textContent = text;
      return;
    }

    const target = parseFloat(el.getAttribute('data-target'));
    const suffix = el.getAttribute('data-suffix') || '';
    const decimal = parseInt(el.getAttribute('data-decimal') || '0');
    const duration = 1500;
    const start = performance.now();

    function tick(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const value = eased * target;

      if (decimal > 0) {
        el.textContent = value.toFixed(decimal) + suffix;
      } else {
        el.textContent = Math.round(value) + suffix;
      }

      if (progress < 1) {
        requestAnimationFrame(tick);
      }
    }

    requestAnimationFrame(tick);
  }

  // ---- Intersection Observer for Animations ----
  const fadeElements = document.querySelectorAll('.fade-in');
  const statValues = document.querySelectorAll('.stat-value');
  let statsAnimated = false;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: '0px 0px -50px 0px' }
  );

  fadeElements.forEach((el) => observer.observe(el));

  // Stat counters — animate when hero is visible
  const heroObserver = new IntersectionObserver(
    (entries) => {
      if (entries[0].isIntersecting && !statsAnimated) {
        statsAnimated = true;
        statValues.forEach((el) => animateCounter(el));
        heroObserver.disconnect();
      }
    },
    { threshold: 0.3 }
  );

  const heroSection = document.getElementById('hero');
  if (heroSection) heroObserver.observe(heroSection);

  // ---- Lazy Section Loading ----
  // Emit custom events when sections scroll into view
  const sections = ['demo', 'training', 'console'];
  const sectionObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          window.dispatchEvent(new CustomEvent('section-visible', { detail: { id } }));
          sectionObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: '200px 0px 200px 0px' }
  );

  sections.forEach((id) => {
    const el = document.getElementById(id);
    if (el) sectionObserver.observe(el);
  });

  // ---- Patch Grid Overlay ----
  const patchOverlay = document.getElementById('patchOverlay');
  if (patchOverlay) {
    for (let i = 0; i < 16; i++) {
      const cell = document.createElement('div');
      cell.className = 'patch-cell';
      cell.setAttribute('data-patch', i);
      patchOverlay.appendChild(cell);
    }
  }

  // ---- Smooth Scroll for CTA ----
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener('click', (e) => {
      e.preventDefault();
      const target = document.querySelector(anchor.getAttribute('href'));
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // ---- Nav Menu Toggle ----
  const navToggle = document.getElementById('navToggle');
  const navMenu = document.getElementById('navMenu');
  const navBackdrop = document.getElementById('navBackdrop');

  function closeNav() {
    navToggle.classList.remove('active');
    navMenu.classList.remove('open');
    navBackdrop.classList.remove('active');
  }

  if (navToggle && navMenu) {
    navToggle.addEventListener('click', () => {
      const isOpen = navMenu.classList.contains('open');
      if (isOpen) {
        closeNav();
      } else {
        navToggle.classList.add('active');
        navMenu.classList.add('open');
        navBackdrop.classList.add('active');
      }
    });

    // Close on backdrop click
    if (navBackdrop) {
      navBackdrop.addEventListener('click', closeNav);
    }

    // Close on nav link click (smooth scroll handled above)
    navMenu.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        closeNav();
      });
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && navMenu.classList.contains('open')) {
        closeNav();
      }
    });
  }

  // ---- Contact Form Handler ----
  document.querySelectorAll('.contact-form').forEach((form) => {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();

      const submitBtn = form.querySelector('button[type="submit"]');
      const originalText = submitBtn.textContent;
      submitBtn.disabled = true;
      submitBtn.textContent = 'Sending...';

      // Remove any previous error
      const prevError = form.querySelector('.contact-error');
      if (prevError) prevError.remove();

      try {
        const resp = await fetch(form.action, {
          method: 'POST',
          body: new FormData(form),
          headers: { 'Accept': 'application/json' },
        });

        if (resp.ok) {
          // Replace form with success message
          const wrapper = form.parentElement;
          form.style.display = 'none';

          const success = document.createElement('div');
          success.className = 'contact-success';
          success.innerHTML = `
            <div class="success-icon">
              <svg viewBox="0 0 24 24"><path d="M5 13l4 4L19 7"/></svg>
            </div>
            <h3>Message sent.</h3>
            <p>I'll get back to you within 24 hours.</p>
          `;
          wrapper.appendChild(success);
        } else {
          throw new Error('Server returned ' + resp.status);
        }
      } catch (err) {
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;

        const errorEl = document.createElement('div');
        errorEl.className = 'contact-error';
        errorEl.textContent = 'Something went wrong. Please try again or email maxwell.sikorski@gmail.com directly.';
        form.appendChild(errorEl);
      }
    });
  });

  // ---- Global helpers ----
  window.SALT = {
    // Training data (loaded lazily)
    trainingData: null,
    // CIFAR-10 samples (loaded lazily)
    samples: null,
    // Embeddings (loaded lazily)
    embeddings: null,

    async loadJSON(path) {
      const resp = await fetch(path);
      if (!resp.ok) throw new Error(`Failed to load ${path}: ${resp.status}`);
      return resp.json();
    },
  };
})();
