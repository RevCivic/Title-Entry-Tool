/*
 * theme.js – Load BEFORE any rendering to prevent flash-of-wrong-theme.
 * Reads saved preference from localStorage, falls back to OS preference.
 */
(function () {
  'use strict';
  try {
    var saved = localStorage.getItem('theme');
    var theme = saved
      ? saved
      : (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
          ? 'dark'
          : 'light');
    document.documentElement.setAttribute('data-theme', theme);
  } catch (_) {
    /* localStorage unavailable – leave default light theme */
  }
}());
