/*
 * base.js – Shared behaviours for every page:
 *   - Dark-mode toggle
 *   - Mobile nav hamburger
 *   - AI status polling → updates nav badge + dispatches 'aiStatusUpdate' event
 *   - Flash banner auto-dismiss
 */
(function () {
  'use strict';

  /* ── Dark-mode toggle ────────────────────────────────────────────────── */
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('theme', theme); } catch (_) {}
  }

  window.toggleTheme = function () {
    var current = document.documentElement.getAttribute('data-theme') || 'light';
    applyTheme(current === 'dark' ? 'light' : 'dark');
  };

  /* ── Mobile nav hamburger ─────────────────────────────────────────────── */
  document.addEventListener('DOMContentLoaded', function () {
    var hamburger = document.getElementById('nav-hamburger');
    var navLinks  = document.getElementById('nav-links-list');
    if (hamburger && navLinks) {
      hamburger.addEventListener('click', function () {
        var open = navLinks.classList.toggle('open');
        hamburger.setAttribute('aria-expanded', String(open));
      });
      /* Close drawer when a link is clicked */
      navLinks.addEventListener('click', function (e) {
        if (e.target.tagName === 'A') {
          navLinks.classList.remove('open');
          hamburger.setAttribute('aria-expanded', 'false');
        }
      });
    }

    /* ── Flash auto-dismiss ───────────────────────────────────────────── */
    document.querySelectorAll('.alert[data-autohide]').forEach(function (el) {
      var delay = parseInt(el.dataset.autohide, 10) || 8000;
      setTimeout(function () {
        el.style.transition = 'opacity 0.5s';
        el.style.opacity = '0';
        setTimeout(function () { el.remove(); }, 500);
      }, delay);
    });

    /* Manual close buttons inside alerts */
    document.querySelectorAll('.alert-close').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var alert = btn.closest('.alert');
        if (alert) {
          alert.style.transition = 'opacity 0.3s';
          alert.style.opacity = '0';
          setTimeout(function () { alert.remove(); }, 300);
        }
      });
    });
  });

  /* ── AI status polling ───────────────────────────────────────────────── */
  var _currentAiStatus = 'unknown';

  /* Allow page scripts to read the last-known status */
  Object.defineProperty(window, '_currentAiStatus', {
    get: function () { return _currentAiStatus; },
    configurable: true,
  });

  function applyNavAiStatus(data) {
    var dot   = document.getElementById('nav-ai-dot');
    var label = document.getElementById('nav-ai-label');
    if (!dot || !label) return;

    var s = data.ai_status;
    _currentAiStatus = s;

    dot.className = 'ai-dot';
    if (s === 'ready') {
      dot.classList.add('ai-dot-green');
      label.textContent = 'AI Ready';
    } else if (s === 'loading') {
      dot.classList.add('ai-dot-yellow');
      label.textContent = 'AI Loading…';
    } else if (s === 'not_configured') {
      dot.classList.add('ai-dot-grey');
      label.textContent = 'OCR only';
    } else {
      dot.classList.add('ai-dot-red');
      label.textContent = 'AI Unavailable';
    }

    /* Dispatch a custom event so page-specific scripts can react */
    document.dispatchEvent(new CustomEvent('aiStatusUpdate', { detail: data }));
  }

  function pollAiStatus() {
    fetch('/api/status')
      .then(function (r) { return r.json(); })
      .then(applyNavAiStatus)
      .catch(function () {
        var dot = document.getElementById('nav-ai-dot');
        if (dot) {
          dot.className = 'ai-dot ai-dot-red';
        }
      });
  }

  /* Initial poll + every 60 s.  The Diagnostics page uses its own 5-second
     poll against /api/diagnostics/ai-progress for the detailed view. */
  document.addEventListener('DOMContentLoaded', function () {
    pollAiStatus();
    setInterval(pollAiStatus, 60000);
  });

}());
