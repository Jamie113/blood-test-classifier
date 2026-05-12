// Right-side methodology drawer controller.
//
// Open/close, Esc, click-outside, URL state (?methodology=<tab>), HTMX-swap
// resync. The drawer markup is server-rendered once at page load and lives
// outside #page-body, so tab swaps don't recreate it.

(function () {
  const QS_KEY = "methodology";
  const SECTIONS = ["explorer", "population", "investigate", "pairs"];
  const SECTION_TO_ANCHOR = {
    explorer:    "methodology-gmm",
    population:  "methodology-clusters",
    investigate: "methodology-outliers",
    pairs:       "methodology-correlations",
  };

  const drawer = document.getElementById("methodology-drawer");
  if (!drawer) return;

  const panel = drawer.querySelector(".methodology-drawer__panel");
  const closeBtn = drawer.querySelector(".methodology-drawer__close");
  let lastTrigger = null;

  function readSectionFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const s = params.get(QS_KEY);
    return SECTIONS.includes(s) ? s : null;
  }

  function urlWithSection(section) {
    const url = new URL(window.location.href);
    if (section) url.searchParams.set(QS_KEY, section);
    else url.searchParams.delete(QS_KEY);
    return url.pathname + (url.search || "") + url.hash;
  }

  function scrollToSection(section) {
    const anchorId = SECTION_TO_ANCHOR[section];
    if (!anchorId) return;
    const target = drawer.querySelector("#" + anchorId);
    if (!target) return;
    const body = drawer.querySelector(".methodology-drawer__body");
    if (body && target) {
      body.scrollTop = target.offsetTop - body.offsetTop;
    }
  }

  function open(section, opts = {}) {
    if (!SECTIONS.includes(section)) section = "explorer";
    drawer.classList.add("is-open");
    drawer.setAttribute("aria-hidden", "false");
    document.body.classList.add("methodology-drawer-open");
    scrollToSection(section);
    if (opts.pushUrl !== false) {
      const next = urlWithSection(section);
      if (next !== window.location.pathname + window.location.search + window.location.hash) {
        history.pushState({ methodology: section }, "", next);
      }
    }
    // Defer focus so the slide-in transition doesn't fight the scroll.
    setTimeout(() => closeBtn && closeBtn.focus(), 50);
  }

  function close(opts = {}) {
    if (!drawer.classList.contains("is-open")) return;
    drawer.classList.remove("is-open");
    drawer.setAttribute("aria-hidden", "true");
    document.body.classList.remove("methodology-drawer-open");
    if (opts.pushUrl !== false) {
      const next = urlWithSection(null);
      history.pushState({}, "", next);
    }
    if (lastTrigger && document.body.contains(lastTrigger)) {
      lastTrigger.focus();
    }
    lastTrigger = null;
  }

  // Trigger button (re-rendered with each tab swap, so use delegation).
  document.addEventListener("click", (e) => {
    const trigger = e.target.closest('[data-mdr-action="open"]');
    if (trigger) {
      e.preventDefault();
      lastTrigger = trigger;
      const section = trigger.dataset.mdrSection || "explorer";
      open(section);
      return;
    }
    const closer = e.target.closest('[data-mdr-action="close"]');
    if (closer && drawer.contains(closer)) {
      e.preventDefault();
      close();
    }
  });

  // ToC links inside the drawer scroll within the body, no URL change.
  drawer.querySelectorAll(".methodology-drawer__toc a").forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const section = a.dataset.mdrSection;
      if (section) scrollToSection(section);
    });
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && drawer.classList.contains("is-open")) {
      e.preventDefault();
      close();
    }
  });

  // Back/forward navigation: open or close based on the new URL state.
  window.addEventListener("popstate", () => {
    const section = readSectionFromUrl();
    if (section) open(section, { pushUrl: false });
    else close({ pushUrl: false });
  });

  // After an HTMX swap (tab change), HTMX pushes a URL based on the clicked
  // tab's href, which drops the methodology param. If the drawer is open,
  // restore the param and re-scroll to the new tab's section.
  document.body.addEventListener("htmx:afterSwap", () => {
    if (!drawer.classList.contains("is-open")) return;
    const currentTrigger = document.querySelector('[data-mdr-action="open"][data-mdr-section]');
    const section = currentTrigger?.dataset.mdrSection;
    if (!SECTIONS.includes(section)) return;
    scrollToSection(section);
    const next = urlWithSection(section);
    history.replaceState(history.state, "", next);
  });

  // Initial open from URL.
  const initial = readSectionFromUrl();
  if (initial) open(initial, { pushUrl: false });
})();
