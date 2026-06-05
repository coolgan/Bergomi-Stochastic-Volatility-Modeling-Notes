
(function () {
  'use strict';

  // ---- Mobile sidebar toggle ----
  var toggle = document.getElementById('menuToggle');
  var overlay = document.getElementById('overlay');
  function closeNav() { document.body.classList.remove('nav-open'); }
  if (toggle) {
    toggle.addEventListener('click', function () {
      document.body.classList.toggle('nav-open');
    });
  }
  if (overlay) overlay.addEventListener('click', closeNav);

  // ---- Desktop reading layout toggles ----
  var sidebarToggle = document.getElementById('sidebarToggle');
  var tocToggle = document.getElementById('tocToggle');
  var storage = null;
  try { storage = window.localStorage; } catch (err) { storage = null; }

  function readStored(key) {
    if (!storage) return false;
    return storage.getItem(key) === '1';
  }

  function writeStored(key, value) {
    if (storage) storage.setItem(key, value ? '1' : '0');
  }

  function setPressed(button, pressed, shownLabel, hiddenLabel) {
    if (!button) return;
    button.setAttribute('aria-pressed', pressed ? 'true' : 'false');
    button.setAttribute('aria-label', pressed ? shownLabel : hiddenLabel);
  }

  function setSidebarCollapsed(collapsed) {
    document.body.classList.toggle('sidebar-collapsed', collapsed);
    setPressed(sidebarToggle, collapsed, '显示章节导航', '隐藏章节导航');
    writeStored('bergomi.sidebarCollapsed', collapsed);
  }

  function setTocCollapsed(collapsed) {
    document.body.classList.toggle('toc-collapsed', collapsed);
    setPressed(tocToggle, collapsed, '显示本章目录', '隐藏本章目录');
    writeStored('bergomi.tocCollapsed', collapsed);
  }

  setSidebarCollapsed(readStored('bergomi.sidebarCollapsed'));
  setTocCollapsed(readStored('bergomi.tocCollapsed'));

  if (sidebarToggle) {
    sidebarToggle.addEventListener('click', function () {
      setSidebarCollapsed(!document.body.classList.contains('sidebar-collapsed'));
    });
  }

  if (tocToggle) {
    tocToggle.addEventListener('click', function () {
      setTocCollapsed(!document.body.classList.contains('toc-collapsed'));
    });
  }

  // ---- Sidebar filter ----
  var search = document.getElementById('navSearch');
  if (search) {
    search.addEventListener('input', function () {
      var q = search.value.trim().toLowerCase();
      var items = document.querySelectorAll('.sidebar li');
      var sections = document.querySelectorAll('.nav-section');
      items.forEach(function (li) {
        var text = li.textContent.toLowerCase();
        li.style.display = !q || text.indexOf(q) !== -1 ? '' : 'none';
      });
      sections.forEach(function (sec) {
        var any = sec.querySelectorAll('li');
        var visible = false;
        any.forEach(function (li) { if (li.style.display !== 'none') visible = true; });
        sec.style.display = visible ? '' : 'none';
      });
    });
  }

  // ---- Keep active sidebar item in view ----
  var active = document.querySelector('.sidebar li.active');
  if (active && active.scrollIntoView) {
    active.scrollIntoView({ block: 'center' });
  }

  // ---- TOC scroll-spy ----
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll('.toc a'));
  if (tocLinks.length && 'IntersectionObserver' in window) {
    var map = {};
    var headings = [];
    tocLinks.forEach(function (a) {
      var id = decodeURIComponent(a.getAttribute('href').slice(1));
      var el = document.getElementById(id);
      if (el) { map[id] = a; headings.push(el); }
    });
    var current = null;
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          if (current) current.classList.remove('active');
          var link = map[entry.target.id];
          if (link) { link.classList.add('active'); current = link; }
        }
      });
    }, { rootMargin: '0px 0px -75% 0px', threshold: 0 });
    headings.forEach(function (h) { observer.observe(h); });
  }
})();
