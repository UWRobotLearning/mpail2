// Load sections dynamically
async function loadSections() {
    const sectionMappings = {
        'summary-section-container': 'sections/summary.html',
        'method-section-container': 'sections/method.html',
        'results-overview-section-container': 'sections/experiments-overview.html',
        'baselines-section-container': 'sections/baselines.html',
        'efficiency-section-container': 'sections/efficiency.html',
        'transfer-section-container': 'sections/transfer.html',
        'generalization-section-container': 'sections/generalization.html',
        'your-turn-section-container': 'sections/your-turn.html'
    };

    const fetchSection = async (sectionFile, attempts = 3) => {
        for (let i = 0; i < attempts; i++) {
            try {
                const response = await fetch(sectionFile, { cache: 'no-cache' });
                if (response.ok) return await response.text();
                console.error(`Failed to load section (${response.status}): ${sectionFile}`);
            } catch (error) {
                console.error(`Error loading section ${sectionFile} (attempt ${i + 1}):`, error);
            }
            await new Promise(r => setTimeout(r, 250 * (i + 1)));
        }
        return null;
    };

    const loadPromises = Object.entries(sectionMappings).map(async ([containerId, sectionFile]) => {
        const html = await fetchSection(sectionFile);
        if (html === null) return;
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = html;
        }
    });
    await Promise.all(loadPromises);
    // Initialize any interactive elements that depend on loaded sections.
    // Wrap so a failure here can't prevent the scroll re-fires below.
    try {
        initializeInteractiveSections();
    } catch (e) {
        console.error('initializeInteractiveSections failed:', e);
    }

    // Make results blocks collapsible (efficiency + transfer)
    document.querySelectorAll('.content-block .collapsible-body').forEach((bodyEl) => {
        const block = bodyEl.closest('.content-block');
        if (!block) return;
        if (block.id === 'experiment-settings-block') return;   // always visible — no collapse toggle
        const header = block.querySelector('.subsection-header');
        if (!header) return;

        block.classList.add('collapsible');
        block.classList.add('collapsed'); // begin collapsed
        header.style.cursor = 'pointer';
        header.setAttribute('role', 'button');
        header.setAttribute('tabindex', '0');
        header.setAttribute('aria-expanded', block.classList.contains('collapsed') ? 'false' : 'true');

        const toggle = () => {
            block.classList.toggle('collapsed');
            header.setAttribute('aria-expanded', block.classList.contains('collapsed') ? 'false' : 'true');
        };

        header.addEventListener('click', toggle);
        header.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggle();
            }
        });
    });
    // Trigger scroll recalculation after sections load. Fire several times to
    // account for layout shifts as images/videos finish loading (more noticeable
    // on slower hosts like GitHub Pages), plus once on full window load.
    const fireScroll = () => window.dispatchEvent(new Event('scroll'));
    fireScroll();
    [100, 400, 1000, 2500].forEach(ms => setTimeout(fireScroll, ms));
    window.addEventListener('load', fireScroll);
}

// Load sections when DOM is ready
document.addEventListener('DOMContentLoaded', loadSections);

// Rich term tooltips (bold/italic via data-tooltip-html with <strong>, <em>, etc.)
document.addEventListener('DOMContentLoaded', function() {
    var richTooltip = document.createElement('div');
    richTooltip.className = 'term-rich-tooltip';
    document.body.appendChild(richTooltip);

    var hideTimer = null;
    var HIDE_DELAY_MS = 50;

    var hideTransitionTimer = null;

    function showRichTooltip(termEl) {
        if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; }
        if (hideTransitionTimer) { clearTimeout(hideTransitionTimer); hideTransitionTimer = null; }
        var html = termEl.getAttribute('data-tooltip-html');
        if (!html) return;
        richTooltip.innerHTML = html;
        richTooltip.style.display = 'block';
        richTooltip.style.left = '-9999px';
        richTooltip.classList.remove('term-rich-tooltip--visible');
        var rect = termEl.getBoundingClientRect();
        var tipRect = richTooltip.getBoundingClientRect();
        var left = rect.left + (rect.width / 2) - (tipRect.width / 2);
        var top = rect.top - tipRect.height - 6;
        richTooltip.style.left = Math.max(8, Math.min(left, window.innerWidth - tipRect.width - 8)) + 'px';
        richTooltip.style.top = Math.max(8, top) + 'px';
        requestAnimationFrame(function() {
            requestAnimationFrame(function() {
                richTooltip.classList.add('term-rich-tooltip--visible');
            });
        });
    }

    function hideRichTooltip() {
        richTooltip.classList.remove('term-rich-tooltip--visible');
        if (hideTransitionTimer) clearTimeout(hideTransitionTimer);
        hideTransitionTimer = setTimeout(function() {
            hideTransitionTimer = null;
            richTooltip.style.display = 'none';
        }, 200);
    }

    function scheduleHide() {
        if (hideTimer) clearTimeout(hideTimer);
        hideTimer = setTimeout(function() {
            hideTimer = null;
            hideRichTooltip();
        }, HIDE_DELAY_MS);
    }

    function cancelScheduleHide() {
        if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; }
    }

    document.addEventListener('mouseover', function(e) {
        var el = e.target.closest && e.target.closest('[data-tooltip-html]');
        if (el) {
            cancelScheduleHide();
            showRichTooltip(el);
        } else if (e.target === richTooltip || richTooltip.contains(e.target)) {
            cancelScheduleHide();
        }
    });

    document.addEventListener('mouseout', function(e) {
        var term = e.target.closest && e.target.closest('[data-tooltip-html]');
        var enteringTooltip = e.relatedTarget && (e.relatedTarget === richTooltip || richTooltip.contains(e.relatedTarget));
        if (term && !term.contains(e.relatedTarget) && !enteringTooltip) scheduleHide();
        if (term && !e.relatedTarget) scheduleHide();
    });

    richTooltip.addEventListener('mouseleave', function() {
        cancelScheduleHide();
        hideRichTooltip();
    });
});

// Authors toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    // Authors toggle functionality
    var authorsToggle = document.getElementById('authors-toggle');
    var authorsSection = document.getElementById('authors-section');
    if (authorsToggle && authorsSection) {
        authorsToggle.addEventListener('click', function() {
            this.classList.toggle('expanded');
            authorsSection.classList.toggle('expanded');
        });
    }
});

// Hero "NO <assumption>" list. The full list sits beside the video and scrolls
// vertically (a seamless carousel) past a STATIC "NO"; whichever assumption is
// beside NO is in focus. Words are read from the markup, so adding a
// <li class="rotator__word"> in index.html is all that's needed.
document.addEventListener('DOMContentLoaded', function() {
    var list = document.getElementById('hero-rotator');
    var no = document.getElementById('hero-no');
    if (!list || !no) return;
    var claim = list.closest('.hero-claim') || list.parentNode;
    var orig = Array.prototype.slice.call(list.querySelectorAll('.rotator__word'));
    var N = orig.length;
    if (!N) return;

    // Clone the words above and below (3 copies) so the strip always has content
    // on both sides of NO and can wrap with no visible seam.
    var clone = function(w) { var c = w.cloneNode(true); c.classList.remove('is-active'); return c; };
    var firstChild = list.firstChild;
    orig.forEach(function(w) { list.insertBefore(clone(w), firstChild); }); // copy above
    orig.forEach(function(w) { list.appendChild(clone(w)); });              // copy below
    var all = Array.prototype.slice.call(list.children);                    // 3N, middle copy is the original

    var contC = 0;                                     // continuous offset of the strip, in word units
    var focusY = function() { return no.offsetTop + no.offsetHeight / 2; };   // NO's center: the carousel's focal line
    // set the words' heights + the strip position; the transform is driven by the continuous offset
    function setStrip(animate) {
        if (claim.clientHeight < 1) return;            // not laid out yet (e.g. video unsized) — wait for the observer
        var h = claim.clientHeight / N;                // N words exactly fill the video height
        all.forEach(function(el) { el.style.height = h + 'px'; });
        list.style.transition = animate ? '' : 'none';
        list.style.transform = 'translateY(' + (focusY() - ((N + contC) * h + h / 2)) + 'px)';
        if (!animate) { void list.offsetHeight; list.style.transition = ''; }
    }
    // light up whichever word's center is within an epsilon of NO's center (used live while dragging)
    function highlightAtCenter(ty) {
        var h = claim.clientHeight / N, center = focusY(), eps = h * 0.42;
        all.forEach(function(el, k) {
            el.classList.toggle('is-active', Math.abs((ty + k * h + h / 2) - center) < eps);
        });
    }
    // same epsilon focus, but read from live layout so it works during the CSS-driven cycle slide
    // (each word lights up as it passes NO, instead of defocusing then snapping on at the center).
    var hlRAF = null;
    function highlightLive() {
        var cr = claim.getBoundingClientRect(), center = cr.top + focusY();
        var eps = (claim.clientHeight / N) * 0.42;
        all.forEach(function(el) {
            var r = el.getBoundingClientRect();
            el.classList.toggle('is-active', Math.abs((r.top + r.height / 2) - center) < eps);
        });
    }
    setStrip(false); highlightLive();
    if (document.fonts && document.fonts.ready) { document.fonts.ready.then(function() { setStrip(false); }); }
    window.addEventListener('resize', function() { setStrip(false); });
    setTimeout(function() { setStrip(false); }, 300);  // re-layout once the video sets the height
    setTimeout(function() { setStrip(false); }, 1200);
    // the claim's height is driven by the video (align-self: stretch); recompute the strip whenever it
    // actually changes size so the words never stay compressed when the video sizes late (load race)
    var resync = function() { setStrip(false); highlightLive(); };
    if (window.ResizeObserver) { new ResizeObserver(resync).observe(claim); }
    var heroVid = document.querySelector('.hero-rollout video');
    if (heroVid) { heroVid.addEventListener('loadeddata', resync); heroVid.addEventListener('loadedmetadata', resync); }

    var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ----- continuous auto-cycle: the strip scrolls at a steady speed (no pausing), each word
    //       lighting up as it passes NO; pausable while the user drags -----
    var cycleRAF = null, lastFrame = 0, dragging = false;
    var SPEED = 1 / 1500;                              // nominal words per ms (~1.5s per word)
    var MAX_SPEED = 1 / 60;                            // cap a flick at ~one word / 60ms
    var RETURN_TAU = 650;                              // ms time-constant easing speed back to nominal
    var curSpeed = SPEED;                              // live speed (signed); a flick sets it, then it decays
    // keep contC within [-0.5, N-0.5) so the seam (copy-swap) always lands in the dim gap between words
    function wrapC() { contC = ((contC + 0.5) % N + N) % N - 0.5; }
    function startCycle() {
        if (reduce || N < 2 || cycleRAF) return;
        lastFrame = 0;
        var tick = function(now) {
            if (dragging) { cycleRAF = null; return; }
            if (lastFrame) {
                var dt = now - lastFrame;
                curSpeed = SPEED + (curSpeed - SPEED) * Math.exp(-dt / RETURN_TAU);  // ease back to nominal
                contC += dt * curSpeed;
                wrapC();
                var h = claim.clientHeight / N;
                var ty = focusY() - ((N + contC) * h + h / 2);
                list.style.transition = 'none';
                list.style.transform = 'translateY(' + ty + 'px)';
                highlightAtCenter(ty);                 // analytic — avoids a per-frame layout read
            }
            lastFrame = now;
            cycleRAF = requestAnimationFrame(tick);
        };
        cycleRAF = requestAnimationFrame(tick);
    }
    function stopCycle() { if (cycleRAF) cancelAnimationFrame(cycleRAF); cycleRAF = null; lastFrame = 0; }

    // ----- drag to scrub through the assumptions; cycling resumes after release -----
    if (!reduce && N >= 2) {
        var wordH = function() { return claim.clientHeight / N; };
        var startY = 0, baseTy = 0;
        var lastY = 0, lastT = 0, dragVel = 0;                // px & px/ms, for release momentum
        var getY = function(e) {
            if (e.touches && e.touches[0]) return e.touches[0].clientY;
            if (e.changedTouches && e.changedTouches[0]) return e.changedTouches[0].clientY;
            return e.clientY;
        };
        var onDown = function(e) {
            dragging = true; stopCycle();
            startY = getY(e);
            lastY = startY; lastT = performance.now(); dragVel = 0;
            baseTy = focusY() - ((N + contC) * wordH() + wordH() / 2);
            list.style.transition = 'none';
            claim.classList.add('is-dragging');
            e.preventDefault();
        };
        var onMove = function(e) {
            if (!dragging) return;
            var y = getY(e), t = performance.now(), dtm = t - lastT;
            if (dtm > 0) { dragVel = dragVel * 0.6 + ((y - lastY) / dtm) * 0.4; lastY = y; lastT = t; }  // smoothed px/ms
            var ty = baseTy + (y - startY);
            list.style.transform = 'translateY(' + ty + 'px)';
            highlightAtCenter(ty);             // words light up as they pass NO
            e.preventDefault();
        };
        var onUp = function(e) {
            if (!dragging) return;
            dragging = false; claim.classList.remove('is-dragging');
            contC -= (getY(e) - startY) / wordH();            // drag down -> earlier words (continuous)
            wrapC();                                          // normalize into [-0.5, N-0.5)
            // hand off the release velocity as the cycle speed; it eases back to nominal in startCycle.
            // contC moves opposite to drag-Y (see above), so negate; stale flick (paused before release) -> 0.
            if (performance.now() - lastT > 120) dragVel = 0;
            var v = -dragVel / wordH();                       // words/ms in contC space
            curSpeed = Math.max(-MAX_SPEED, Math.min(MAX_SPEED, v));
            highlightLive();
            startCycle();                                     // resume the continuous scroll from here
        };
        claim.addEventListener('mousedown', onDown);
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
        claim.addEventListener('touchstart', onDown, { passive: false });
        window.addEventListener('touchmove', onMove, { passive: false });
        window.addEventListener('touchend', onUp);
    }
    startCycle();
});

// Main functionality
document.addEventListener('DOMContentLoaded', function() {
    // One-time hint for task switching in Experiments
    const leftSidebarHint = document.createElement('div');
    leftSidebarHint.className = 'left-sidebar-hint';
    leftSidebarHint.innerHTML = `
        <button class="hint-close" aria-label="Dismiss hint" title="Dismiss">×</button>
        <strong>Tip:</strong> Switch between experiment tasks here.
    `;
    document.body.appendChild(leftSidebarHint);
    let dismissedLeftHint = false;
    const dismissLeftHint = () => {
        leftSidebarHint.classList.remove('visible');
        dismissedLeftHint = true; // resets on refresh
    };
    leftSidebarHint.querySelector('.hint-close')?.addEventListener('click', dismissLeftHint);
    // Expose for other handlers (e.g., task changes)
    window.dismissLeftSidebarHint = dismissLeftHint;

    const positionLeftSidebarHint = () => {
        const sidebar = document.getElementById('task-buttons-left');
        if (!sidebar) return;
        const sidebarRect = sidebar.getBoundingClientRect();
        if (sidebarRect.width === 0 || sidebarRect.height === 0) return;

        // Ensure we can measure the tooltip
        const wasHidden = !leftSidebarHint.classList.contains('visible');
        if (wasHidden) {
            leftSidebarHint.style.visibility = 'hidden';
            leftSidebarHint.style.display = 'block';
        }

        const hintRect = leftSidebarHint.getBoundingClientRect();

        // the selector is now a horizontal rail at top-center, so the hint sits centered just below it
        const gap = 10;
        const left = sidebarRect.left + (sidebarRect.width / 2) - (hintRect.width / 2);
        const top = sidebarRect.bottom + gap;
        const clampedLeft = Math.max(8, Math.min(left, window.innerWidth - hintRect.width - 8));
        const clampedTop = Math.max(8, Math.min(top, window.innerHeight - hintRect.height - 8));

        leftSidebarHint.style.left = `${clampedLeft}px`;
        leftSidebarHint.style.top = `${clampedTop}px`;
        leftSidebarHint.style.transform = 'translateY(0)';

        // point the (upward) arrow at the horizontal center of the rail
        const arrowX = (sidebarRect.left + sidebarRect.width / 2) - clampedLeft;
        leftSidebarHint.style.setProperty('--hint-arrow-x', `${Math.max(16, Math.min(hintRect.width - 16, arrowX))}px`);

        if (wasHidden) {
            leftSidebarHint.style.display = 'none';
            leftSidebarHint.style.visibility = '';
        }
    };

    window.addEventListener('resize', () => {
        if (leftSidebarHint.classList.contains('visible')) positionLeftSidebarHint();
    });

    // Initialize elements that may or may not be present yet
    initializeInteractiveSections();

    // Initialize navigation elements
    const taskButtonsLeft = document.getElementById('task-buttons-left');
    const tasksSection = document.querySelector('.tasks-section');
    
    // Timeline navigation
    const timelineNav = document.getElementById('timeline-nav');
    const themeToggle = document.getElementById('theme-toggle');

    // Initialize navigation state
    window.isTimelineNavigating = false;

    if (themeToggle) {
        themeToggle.classList.add('hidden-on-hero');
    }
    
    window.addEventListener('scroll', function() {
        const scrollY = window.scrollY;
        const windowHeight = window.innerHeight;

        // Show/hide timeline navigation based on hero visibility
        const heroSection = document.querySelector('.hero-section');
        if (heroSection && timelineNav) {
            const heroRect = heroSection.getBoundingClientRect();
            const heroBottom = heroRect.bottom + window.scrollY; // static page-space bottom
            if (window.scrollY > heroBottom - windowHeight * 0.25) {
                timelineNav.classList.add('visible');
            } else {
                timelineNav.classList.remove('visible');
            }
        }

        // Get section references (they are loaded dynamically)
        const summarySection = document.getElementById('summary-section');
        const methodSection = document.getElementById('method-section');
        const resultsOverviewSection = document.getElementById('results-overview-section');
        const baselinesSection = document.getElementById('baselines-section');
        const efficiencySection = document.getElementById('efficiency-section');
        const transferSection = document.getElementById('transfer-section');
        const generalizationSection = document.getElementById('generalization-section');
        const yourTurnSection = document.getElementById('your-turn-section');

        // Only bail if nothing has loaded yet. Otherwise run with whatever
        // sections are present so a single slow/failed fetch (e.g. on GitHub
        // Pages) can't freeze the whole scroll/highlight/task-button logic.
        if (!summarySection && !methodSection && !resultsOverviewSection &&
            !efficiencySection && !transferSection && !generalizationSection &&
            !yourTurnSection) {
            return;
        }

        // Safe rect: missing sections are treated as far off-screen (top = Infinity)
        // so they never satisfy "top < threshold" checks below.
        const OFF = { top: Infinity, bottom: Infinity };
        const rectOf = (el) => (el ? el.getBoundingClientRect() : OFF);

        // Get section positions
        const summaryRect = rectOf(summarySection);
        const methodRect = rectOf(methodSection);
        const resultsOverviewRect = rectOf(resultsOverviewSection);
        const baselinesRect = rectOf(baselinesSection);
        const efficiencyRect = rectOf(efficiencySection);
        const transferRect = rectOf(transferSection);
        const generalizationRect = rectOf(generalizationSection);
        const yourTurnRect = rectOf(yourTurnSection);
        
        // Show/hide left task buttons
        if (taskButtonsLeft) {
            const showLeftButtons = efficiencyRect.top < windowHeight * 0.75 && generalizationRect.top > windowHeight * 0.5;
            if (showLeftButtons) {
                taskButtonsLeft.classList.add('visible');
            } else {
                taskButtonsLeft.classList.remove('visible');
            }
        }
        // Bottom bar (research questions + claims + training scrubber): show while the results section is
        // meaningfully in view; surface the scrubber here only when the stage's own scrubber is off-screen.
        const resultsSelector = document.getElementById('results-selector');
        if (resultsSelector) {
            const show = resultsOverviewRect.top < windowHeight * 0.55 && resultsOverviewRect.bottom > 130;
            resultsSelector.classList.toggle('visible', show);
            const sc = document.querySelector('#results-media .results-scrub');
            let scOff = false;
            if (sc) { const r = sc.getBoundingClientRect(); scOff = r.bottom < 72 || r.top > windowHeight - 40; }
            resultsSelector.classList.toggle('has-scrub', show && !!sc && scOff);
        }

        if (themeToggle) {
            if (scrollY > windowHeight * 0.9) {
                themeToggle.classList.remove('hidden-on-hero');
            } else {
                themeToggle.classList.add('hidden-on-hero');
            }
        }
        
        // Skip active section updates during navigation
        if (window.isTimelineNavigating) {
            return;
        }
        
        // Determine active section (matches data-section values in the timeline)
        let activeSection = 'summary';
        if (yourTurnRect.top < windowHeight * 0.5) {
            activeSection = 'your-turn';
        } else if (generalizationRect.top < windowHeight * 0.5) {
            activeSection = 'generalization';
        } else if (document.body.classList.contains('results-undirected') &&
                   (transferRect.top < windowHeight * 0.5 || efficiencyRect.top < windowHeight * 0.5)) {
            activeSection = 'results-overview';   // Training + Transfer = the Undirected view of Results
        } else if (resultsOverviewRect.top < windowHeight * 0.5) {
            activeSection = 'results-overview';
        } else if (baselinesRect.top < windowHeight * 0.5) {
            activeSection = 'baselines';
        } else if (methodRect.top < windowHeight * 0.5) {
            activeSection = 'method';
        }

        // Show a one-time hint to use the left sidebar during Side-by-Side
        const leftButtonsVisible = taskButtonsLeft && taskButtonsLeft.classList.contains('visible');
        if (!dismissedLeftHint && leftButtonsVisible && activeSection === 'side-by-side') {
            leftSidebarHint.classList.add('visible');
            positionLeftSidebarHint();
        } else {
            leftSidebarHint.classList.remove('visible');
        }

        const timelineItems = document.querySelectorAll('.timeline-item');
        timelineItems.forEach(item => {
            if (item.dataset.section === activeSection) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });

        // Section visibility management
        const showGeneralization = generalizationRect.top < windowHeight * 0.75;
        // efficiency (Training) + transfer exist only in Undirected mode; when hidden their rects
        // collapse to 0, so gate on the mode to avoid wrongly hiding Generalization in Focus mode
        const undirectedMode = document.body.classList.contains('results-undirected');
        const showTransfer = undirectedMode && transferRect.top < windowHeight * 0.75;
        const showEfficiency = undirectedMode && efficiencyRect.top < windowHeight * 0.75;

        // Reset section states (null-safe)
        summarySection?.classList.remove('fade-out');
        efficiencySection?.classList.remove('fade-in', 'fade-out', 'hidden-below');
        transferSection?.classList.remove('fade-in', 'fade-out', 'hidden-below');
        generalizationSection?.classList.remove('fade-in', 'fade-out', 'hidden-below');

        if (showGeneralization) {
            summarySection?.classList.add('fade-out');
            efficiencySection?.classList.add('fade-out');
            transferSection?.classList.add('fade-out');
            generalizationSection?.classList.add('fade-in');
        } else if (showTransfer) {
            summarySection?.classList.add('fade-out');
            efficiencySection?.classList.add('fade-out');
            transferSection?.classList.add('fade-in');
            generalizationSection?.classList.add('hidden-below');
        } else if (showEfficiency) {
            summarySection?.classList.add('fade-out');
            efficiencySection?.classList.add('fade-in');
            transferSection?.classList.add('hidden-below');
            generalizationSection?.classList.add('hidden-below');
        } else {
            efficiencySection?.classList.add('hidden-below');
            transferSection?.classList.add('hidden-below');
            generalizationSection?.classList.add('hidden-below');
        }

    });
});

// Results table — "tunnel vision" onto the rows/columns that evidence a selected finding
function initResultsTunnel() {
    const table = document.getElementById('tv-table');
    const chipsWrap = document.getElementById('results-claims');
    const caption = document.getElementById('tv-caption');
    if (!table || !chipsWrap) return;

    // body column order (cellIndex 0 is the method label); header row 2 is the same minus the method col
    const COLS = ['method', 'sup-reward', 'sup-actions', 'comp-plan', 'comp-model', 'comp-offpolicy',
        'res-bp', 'res-pnp', 'res-mop', 'tr-bp', 'tr-pnp', 'sc-bp', 'sc-pnp', 'vid-bp'];
    const bodyRows = Array.prototype.slice.call(table.querySelectorAll('tbody tr[data-method]'));
    bodyRows.forEach(tr => Array.prototype.forEach.call(tr.children, (td, i) => { if (COLS[i]) td.dataset.col = COLS[i]; }));
    const headRows = table.querySelectorAll('thead tr');
    const subHead = headRows[1];
    if (subHead) Array.prototype.forEach.call(subHead.children, (th, i) => { th.dataset.col = COLS[i + 1]; });

    // ----- per-method mini "branding" diagram (the CoRL Q1 reference): a gold Planner enclosure (when
    // the method plans) wrapping the components it has — Dynamics(red) Reward(green/gold) Value(gold)
    // Policy(purple). Shown in the table's method cells and the stage's method labels. -----
    const METHOD_PARTS = {   // table key -> {planner, parts:[ [letter, color], ... ]}
        mpail2: { planner: 1, parts: [['D', '#E0705A'], ['R', '#7FB069'], ['Q', '#E0A53B'], ['P', '#A99BF5']] },
        mairl:  { planner: 0, parts: [['D', '#E0705A'], ['R', '#7FB069'], ['Q', '#E0A53B'], ['P', '#A99BF5']] },
        dac:    { planner: 0, parts: [['R', '#7FB069'], ['Q', '#E0A53B'], ['P', '#A99BF5']] },
        rlpd:   { planner: 0, parts: [['R', '#E0A53B', true], ['Q', '#E0A53B'], ['P', '#A99BF5']] },   // hand reward = gold + hazard
        bc:     { planner: 0, parts: [['P', '#A99BF5', true]] }                                       // severed/offline policy = hazard
    };
    let mHaz = 0;                                       // unique-id counter for hazard-stripe patterns
    function miniDiagram(methodKey) {
        const m = METHOD_PARTS[methodKey];
        if (!m) return '';
        const bw = 14, bh = 15, gap = 3, pad = m.planner ? 4 : 0;
        const W = m.parts.length * bw + (m.parts.length - 1) * gap + pad * 2, H = bh + pad * 2;
        let defs = '', body = '';
        m.parts.forEach((p, i) => {
            const x = pad + i * (bw + gap);
            body += '<rect x="' + x + '" y="' + pad + '" width="' + bw + '" height="' + bh + '" rx="3.5" fill="' + p[1] + '26" stroke="' + p[1] + '" stroke-width="1.3"/>';
            if (p[2]) {                                // diagonal "hazard" stripes behind the letter
                const pid = 'mhaz' + (mHaz++);
                defs += '<pattern id="' + pid + '" patternUnits="userSpaceOnUse" width="3" height="3" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="3" stroke="' + p[1] + '" stroke-width="1.4"/></pattern>';
                body += '<rect x="' + x + '" y="' + pad + '" width="' + bw + '" height="' + bh + '" rx="3.5" fill="url(#' + pid + ')" opacity="0.6"/>';
            }
            body += '<text x="' + (x + bw / 2) + '" y="' + (pad + bh / 2) + '" text-anchor="middle" dominant-baseline="central" font-size="8.5" font-weight="700" fill="' + p[1] + '">' + p[0] + '</text>';
        });
        let s = '<svg class="mini-diagram" width="' + (W + 2) + '" height="' + (H + 2) + '" viewBox="-1 -1 ' + (W + 2) + ' ' + (H + 2) + '" aria-hidden="true">';   // 1-unit margin so box strokes aren't clipped
        if (defs) s += '<defs>' + defs + '</defs>';
        if (m.planner) s += '<rect x="0.7" y="0.7" width="' + (W - 1.4).toFixed(1) + '" height="' + (H - 1.4).toFixed(1) + '" rx="5" fill="none" stroke="#E0A53B" stroke-width="1.3"/>';
        return s + body + '</svg>';
    }
    // drop the diagram into each table method cell (below the name)
    bodyRows.forEach(tr => {
        const cell = tr.children[0], svg = miniDiagram(tr.dataset.method);
        if (cell && svg) cell.insertAdjacentHTML('beforeend', '<span class="mini-diagram-wrap">' + svg + '</span>');
    });

    // each claim = the methods (rows) and metrics (columns) that constitute its evidence, dimming the rest
    const SCRATCH = ['res-bp', 'res-pnp', 'res-mop'];
    const CLAIMS = {
        world: {
            rows: ['mpail2', 'mairl', 'dac'], cols: ['comp-model'].concat(SCRATCH),
            text: 'World modeling is critical. The only IRL methods with a learned dynamics model, MPAIL2 and [−P] (MAIRL), are the only ones with any real-world success. [−PM] (DAC), which drops the model, scores 0% everywhere.'
        },
        planning: {
            rows: ['mpail2', 'mairl'], cols: ['comp-plan'].concat(SCRATCH),
            text: 'Planning mitigates IRL instability. With the MPPI planner, MPAIL2 consistently improves over training, far outperforming the otherwise-identical [−P] (MAIRL) that learns from the policy alone (e.g. 82% vs 16% on Pick-and-Place).'
        },
        supervision: {
            rows: ['mpail2', 'mairl', 'dac', 'bc', 'rlpd'], cols: ['sup-reward', 'sup-actions'].concat(SCRATCH),
            text: 'Less supervision does not mean less performance. MPAIL2 and its IRL ablations use no hand-designed reward and no action supervision, yet match or beat BC and RLPD, which are given both.'
        },
        transfer: {
            rows: ['mpail2', 'mairl', 'bc'], cols: ['tr-bp', 'tr-pnp', 'sc-bp', 'sc-pnp'],
            text: 'Positive online transfer. Initialized from a first task and continued on a related one, MPAIL2 retains high success (90% / 94%) where BC degrades — the first real-world demonstration of positive online transfer in IRLfO.'
        },
        video: {
            rows: ['mpail2', 'mairl', 'dac', 'bc', 'rlpd'], cols: ['vid-bp'],
            text: 'Video-only demonstration. Given only a single fixed, table-mounted camera (no wrist camera, no proprioception), MPAIL2 still learns the push at 63% success.'
        }
    };

    const rail = document.getElementById('results-selector');
    const rqTabs = rail ? Array.prototype.slice.call(rail.querySelectorAll('.results-rq')) : [];
    const chips = Array.prototype.slice.call(chipsWrap.querySelectorAll('.tv-chip'));   // compact-bar chips (RQ-filtered)
    // every clickable claim: the text claims up top + the compact-bar chips
    const claimBtns = Array.prototype.slice.call(document.querySelectorAll('.tv-chip[data-claim], .rq-claim[data-claim]'));
    const RQ_FIRST = { q1: 'world', q2: 'supervision', q3: 'transfer' };   // first claim shown per RQ
    let active = null;

    const cells = Array.prototype.slice.call(table.querySelectorAll('tbody td, thead th'));

    // show only the given RQ's claim chips and mark its tab active
    function showRQ(rq) {
        rqTabs.forEach(t => t.classList.toggle('is-on', t.dataset.rq === rq));
        chips.forEach(c => c.classList.toggle('rq-on', c.dataset.rq === rq));
    }

    function clear() {                                  // un-focus the table (used by the Undirected view)
        table.classList.remove('tv-active');
        cells.forEach(c => c.classList.remove('tv-dim', 'tv-focus'));
        caption.classList.remove('is-on');
        claimBtns.forEach(b => b.classList.remove('is-on'));
        active = null;
    }

    // focus a given set of rows+cols on the table (shared by single-claim and RQ-union views)
    function focusTable(rowSet, colSet) {
        table.classList.add('tv-active');
        bodyRows.forEach(tr => {
            const rowOn = rowSet.has(tr.dataset.method);
            Array.prototype.forEach.call(tr.children, td => {
                const isMethod = td.dataset.col === 'method';
                const focus = isMethod ? rowOn : (rowOn && colSet.has(td.dataset.col));
                td.classList.toggle('tv-focus', focus);
                td.classList.toggle('tv-dim', !focus);
            });
        });
        // sub-header: highlight the relevant metric columns, dim the rest (the method header has no data-col)
        if (subHead) Array.prototype.forEach.call(subHead.children, th => {
            const on = colSet.has(th.dataset.col);
            th.classList.toggle('tv-focus', on); th.classList.toggle('tv-dim', !on);
        });
        // group-header row: dim it as a whole (its colspans don't map cleanly to single columns)
        if (headRows[0]) Array.prototype.forEach.call(headRows[0].children, th => {
            if (th.classList.contains('col-model')) { th.classList.remove('tv-dim'); th.classList.add('tv-focus'); }
            else { th.classList.add('tv-dim'); th.classList.remove('tv-focus'); }
        });
        // if the table is collapsed (scrollable), slide it so the highlighted columns come into view
        const scroller = table.closest('.results-table-scroll');
        if (scroller && subHead && scroller.scrollWidth > scroller.clientWidth + 4) {
            const cols = Array.prototype.slice.call(subHead.children).filter(th => colSet.has(th.dataset.col));
            if (cols.length) {
                const sRect = scroller.getBoundingClientRect();
                const a = cols[0].getBoundingClientRect(), b = cols[cols.length - 1].getBoundingClientRect();
                const center = (a.left + b.right) / 2 - sRect.left + scroller.scrollLeft;
                scroller.scrollTo({ left: Math.max(0, center - scroller.clientWidth / 2), behavior: 'smooth' });
            }
        }
    }

    function apply(key) {
        const claim = CLAIMS[key];
        if (!claim) return;
        selected.clear();                              // an explicit claim resets any manual selection
        bodyRows.forEach(tr => Array.prototype.forEach.call(tr.children, td => td.classList.remove('tv-selected')));
        active = key;
        window.resultsActiveClaim = key;               // so selectTask can refresh the focus media
        claimBtns.forEach(b => b.classList.toggle('is-on', b.dataset.claim === key));
        const chip = claimBtns.filter(c => c.dataset.claim === key)[0];
        if (chip) showRQ(chip.dataset.rq);             // keep the compact bar's RQ + visible chips in sync
        focusTable(new Set(claim.rows), new Set(claim.cols));
        caption.textContent = claim.text;
        caption.classList.add('is-on');
        if (typeof window.renderClaimMedia === 'function') window.renderClaimMedia(key);   // videos
        if (typeof window.renderClaimPlot === 'function') window.renderClaimPlot(key);     // efficiency plots
    }

    // RQ tab / question header -> stage all the RQ's evidence; any claim (text or chip) -> narrow to it
    // (clicking either while in Undirected switches back to Focus so the focusing is visible)
    rqTabs.forEach(tab => tab.addEventListener('click', () => {
        if (document.body.classList.contains('results-undirected')) setResultsMode('focus');
        applyRQ(tab.dataset.rq);
    }));
    claimBtns.forEach(btn => btn.addEventListener('click', () => {
        if (document.body.classList.contains('results-undirected')) setResultsMode('focus');
        apply(btn.dataset.claim);
    }));
    // accordion: clicking a question only expands/collapses its claims — it does NOT change what's
    // active, so opening another question never undoes the currently focused claim. (Stage the whole
    // RQ's evidence via its claim chips / the compact-bar Q1–Q3 tabs.)
    document.querySelectorAll('.rq-group__q').forEach(q => q.addEventListener('click', () => {
        const group = q.closest('.rq-group');
        const open = group.classList.toggle('is-open');
        q.setAttribute('aria-expanded', open ? 'true' : 'false');
    }));
    // reset: clear claims, selections, and the stage back to a neutral state
    const resetBtn = document.getElementById('results-reset');
    if (resetBtn) resetBtn.addEventListener('click', () => {
        selected.clear();
        clear();                                       // un-focus table + active=null + caption off
        bodyRows.forEach(tr => Array.prototype.forEach.call(tr.children, td => td.classList.remove('tv-selected')));
        document.querySelectorAll('.rq-group.is-open').forEach(g => {
            g.classList.remove('is-open');
            const q = g.querySelector('.rq-group__q'); if (q) q.setAttribute('aria-expanded', 'false');
        });
        rqTabs.forEach(t => t.classList.remove('is-on'));
        renderStage([], '');
        if (typeof window.renderClaimPlot === 'function') window.renderClaimPlot('__none');
    });

    // ---- results stage: a grid of method×task training clips, fed by a claim, a research question
    //      (union of its claims), or a hand-picked set of table cells. A shared slider scrubs every
    //      swappable clip through training (as a % of its own schedule, since tasks differ in length). ----
    const mediaPanel = document.getElementById('results-media');
    const METHOD_NAME = { mpail2: 'MPAIL2', mairl: '[−P] (MAIRL)', dac: '[−PM] (DAC)', rlpd: 'RLPD', bc: 'BC (Diffusion)' };
    const FOLDER = { mairl: 'mpail2_p', dac: 'mpail2_pm', rlpd: 'rlpd' };   // baseline comparison clip folders
    const START_PCT = 100;                             // default to the trained (final) iteration

    // (method, result-column) -> a stageable clip, or null if none exists for that pairing.
    //   { prefix, max, task } = iteration-swappable training clip (prefix + "<iter>.mp4", iters 0..max step 10)
    //   { src, task, badges } = a single fixed clip
    function cellVideo(methodKey, col) {
        if (col === 'res-bp' || col === 'res-pnp' || col === 'res-mop') {
            const t = col === 'res-bp'  ? { dir: 'Push', task: 'Block Push',     max: 100 }
                    : col === 'res-pnp' ? { dir: 'Pick', task: 'Pick-and-Place', max: 150 }
                                        : { dir: 'MoP',  task: 'Mug-on-Plate',   max: 200, baselineMax: 140 };
            if (methodKey === 'mpail2') return { prefix: 'Media/Video/' + t.dir + '/iter_', max: t.max, task: t.task };
            const f = FOLDER[methodKey];
            if (!f) return null;                       // BC has no training clip
            return { prefix: 'Media/Video/Comparison/' + f + '/' + t.dir + '/iter_', max: t.baselineMax ?? t.max, task: t.task };
        }
        if (methodKey !== 'mpail2') return null;       // the remaining columns are MPAIL2-only
        switch (col) {
            case 'tr-bp':  return { prefix: 'Media/Video/Transfer/Push/iter_', max: 100, task: 'Block Push · transferred' };
            case 'tr-pnp': return { prefix: 'Media/Video/Transfer/Pick/iter_', max: 150, task: 'Pick-and-Place · transferred' };
            case 'sc-bp':  return { prefix: 'Media/Video/From%20Scratch/Push/iter_', max: 100, task: 'Block Push · from scratch' };
            case 'sc-pnp': return { prefix: 'Media/Video/From%20Scratch/Pick/iter_', max: 150, task: 'Pick-and-Place · from scratch' };
            case 'vid-bp': return { src: 'Media/Video/Push/Video_only/push_vid_only_h264.mp4', task: 'Block Push · video-only', badges: ['Wrist Camera', 'Proprioception'] };
        }
        return null;
    }

    // each claim stages a small set of (method, column) cells, plus an explanatory note
    const CLAIM_CELLS = {
        world:       [['mpail2', 'res-bp'], ['dac', 'res-bp'], ['mpail2', 'res-pnp'], ['dac', 'res-pnp'],
                      ['mpail2', 'res-mop'], ['dac', 'res-mop']],
        planning:    [['mpail2', 'res-bp'], ['mairl', 'res-bp'], ['mpail2', 'res-pnp'], ['mairl', 'res-pnp'],
                      ['mpail2', 'res-mop'], ['mairl', 'res-mop']],
        supervision: [['mpail2', 'res-bp'], ['mairl', 'res-bp'], ['dac', 'res-bp'], ['rlpd', 'res-bp'],
                      ['mpail2', 'res-pnp'], ['mairl', 'res-pnp'], ['dac', 'res-pnp'], ['rlpd', 'res-pnp'],
                      ['mpail2', 'res-mop'], ['mairl', 'res-mop'], ['dac', 'res-mop'], ['rlpd', 'res-mop']],
        transfer:    [['mpail2', 'sc-bp'], ['mpail2', 'tr-bp'], ['mpail2', 'sc-pnp'], ['mpail2', 'tr-pnp']],
        video:       [['mpail2', 'vid-bp']]
    };
    const CLAIM_NOTE = {
        world:       'Only the model-based learner gains traction in the real world; dropping the world model (DAC) collapses to 0%.',
        planning:    'The planner stabilizes the adversarial objective, yielding markedly more robust behavior than the policy alone.',
        supervision: 'With no hand-designed reward and no action labels, the observation-only IRL methods (MPAIL2 and its ablations) match or beat the fully-supervised RLPD.',
        transfer:    'Transfer reaches higher success in roughly half the real-world training time.',
        video:       'Learned from a single fixed, table-mounted camera — no wrist camera and no proprioception.'
    };
    const RQ_CLAIMS = { q1: ['world', 'planning'], q2: ['supervision'], q3: ['transfer', 'video'] };

    // column -> task label, independent of method (used when grouping the stage by task)
    const COL_LABEL = {
        'res-bp': 'Block Push', 'res-pnp': 'Pick-and-Place', 'res-mop': 'Mug-on-Plate',
        'tr-bp': 'Block Push · transfer', 'tr-pnp': 'Pick-and-Place · transfer',
        'sc-bp': 'Block Push · scratch', 'sc-pnp': 'Pick-and-Place · scratch', 'vid-bp': 'Block Push · video-only'
    };
    const METHOD_ORDER = ['mpail2', 'mairl', 'dac', 'bc', 'rlpd'];
    // abbreviated task name shown as the in-video tag (top-right)
    const COL_ABBR = {
        'res-bp': 'BP', 'res-pnp': 'PnP', 'res-mop': 'MoP',
        'tr-bp': 'BP·tr', 'tr-pnp': 'PnP·tr', 'sc-bp': 'BP·sc', 'sc-pnp': 'PnP·sc', 'vid-bp': 'BP·vid'
    };
    const iterFor = (pct, max) => Math.round(pct / 100 * max / 10) * 10;   // % of schedule -> nearest available iter

    // shared training-scrubber state, driven by the stage slider, the bottom-bar slider, and arrow keys.
    // Retargets every staged scrub video to the matching iteration and keeps all scrubber UIs in sync.
    let trainingPct = START_PCT;
    function applyTrainingPct(pct, fromEl) {
        pct = Math.max(0, Math.min(100, Math.round(pct / 10) * 10));
        trainingPct = pct;
        const media = document.getElementById('results-media');
        if (media) media.querySelectorAll('video[data-prefix]').forEach(function (vd) {
            const it = iterFor(pct, +vd.dataset.max || 100);
            const src = vd.dataset.prefix + it + '.mp4', s = vd.querySelector('source');
            if (s && s.getAttribute('src') !== src) { s.setAttribute('src', src); vd.load(); vd.play().catch(function () {}); }
            try { vd.playbackRate = 3; } catch (e) {}
        });
        document.querySelectorAll('.results-scrub__slider, #results-railscrub-slider').forEach(function (s) { if (s !== fromEl && +s.value !== pct) s.value = pct; });
        document.querySelectorAll('.results-scrub__val, #results-railscrub-val').forEach(function (v) { v.textContent = pct + '%'; });
    }
    window.applyTrainingPct = applyTrainingPct;

    const diagramWrap = (methodKey) => {                 // mini branding diagram, centered above a group label
        const svg = miniDiagram(methodKey);
        return svg ? '<span class="mini-diagram-wrap mini-diagram-wrap--label">' + svg + '</span>' : '';
    };
    // in-video overlay tags: branding diagram top-left (method name on hover), abbreviated task top-right
    function overlays(methodKey, col) {
        const svg = miniDiagram(methodKey);
        const brand = '<div class="results-tag results-tag--brand" tabindex="0" title="' + (METHOD_NAME[methodKey] || methodKey) + '">' +
            (svg ? '<span class="results-tag__diagram">' + svg + '</span>' : '') +
            '<span class="results-tag__name">' + (METHOD_NAME[methodKey] || methodKey) + '</span></div>';
        const task = col ? '<div class="results-tag results-tag--task">' + (COL_ABBR[col] || col) + '</div>' : '';
        return brand + task;
    }
    function mediaHTML(methodKey, col, v) {
        const video = v.src
            ? '<video autoplay muted loop playsinline preload="metadata"><source src="' + v.src + '" type="video/mp4"></video>'
            : '<video autoplay muted loop playsinline preload="metadata" data-prefix="' + v.prefix + '" data-max="' + v.max + '"><source src="' + v.prefix + iterFor(START_PCT, v.max) + '.mp4" type="video/mp4"></video>';
        const badges = (v.badges || []).map(b => '<div class="video-badge video-badge--disabled"><span class="video-badge__cross">No</span>' + b + '</div>').join('');
        return '<div class="video-display">' + video + overlays(methodKey, col) +
            (badges ? '<div class="video-badge-row">' + badges + '</div>' : '') + '</div>';
    }
    const tileHTML = (methodKey, col, v) => '<div class="results-tile">' + mediaHTML(methodKey, col, v) + '</div>';
    const emptyTile = '<div class="results-tile results-tile--empty"></div>';

    var stageGroup = 'method';                            // 'none' | 'method' | 'task' (set by the toggle)
    var lastCells = [], lastNote = '';                   // remembered so the toggle can re-render

    // render the stage from a list of [method, col] cells (+ optional note), grouped per stageGroup
    function renderStage(cellList, note) {
        if (!mediaPanel) return;
        lastCells = cellList || []; lastNote = note || '';
        const items = lastCells.map(c => ({ m: c[0], col: c[1], v: cellVideo(c[0], c[1]) })).filter(x => x.v);
        if (!items.length) {
            mediaPanel.innerHTML = '<p class="results-stage-empty">Pick a research question or finding above — or hover the table and click cells, row, or column headers — to stage training videos here.</p>';
            return;
        }
        const hasScrub = items.some(x => x.v.prefix);
        const colsPresent = COLS.filter(c => c !== 'method' && items.some(x => x.col === c));
        const methodsPresent = METHOD_ORDER.filter(m => items.some(x => x.m === m));
        const find = (m, col) => items.filter(x => x.m === m && x.col === col)[0];

        let body;
        if (stageGroup === 'method' && (methodsPresent.length > 1 || colsPresent.length > 1)) {
            // one bordered group per method; columns are the tasks, aligned across groups
            body = '<div class="results-groups">' + methodsPresent.map(m => {
                const grid = colsPresent.map(col => { const it = find(m, col); return it ? tileHTML(m, col, it.v) : emptyTile; }).join('');
                return '<div class="results-group"><div class="results-group__label">' + diagramWrap(m) +
                    '<span>' + (METHOD_NAME[m] || m) + '</span></div>' +
                    '<div class="results-group__grid" style="grid-template-columns:repeat(auto-fit,var(--stage-tile-w, 190px));max-width:calc(' + colsPresent.length + '*var(--stage-tile-w, 190px) + ' + (colsPresent.length - 1) + '*12px)">' + grid + '</div></div>';
            }).join('') + '</div>';
        } else if (stageGroup === 'task' && (methodsPresent.length > 1 || colsPresent.length > 1)) {
            // one bordered group per task; columns are the methods, aligned across groups
            body = '<div class="results-groups">' + colsPresent.map(col => {
                const grid = methodsPresent.map(m => { const it = find(m, col); return it ? tileHTML(m, col, it.v) : emptyTile; }).join('');
                return '<div class="results-group"><div class="results-group__label"><span>' + (COL_LABEL[col] || col) + '</span></div>' +
                    '<div class="results-group__grid" style="grid-template-columns:repeat(auto-fit,var(--stage-tile-w, 190px));max-width:calc(' + methodsPresent.length + '*var(--stage-tile-w, 190px) + ' + (methodsPresent.length - 1) + '*12px)">' + grid + '</div></div>';
            }).join('') + '</div>';
        } else {
            // ungrouped: a flat responsive grid; each video carries its own in-frame tags
            body = '<div class="results-stage-grid' + (items.length === 1 ? ' results-stage-grid--single' : '') + '">' +
                items.map(x => tileHTML(x.m, x.col, x.v)).join('') + '</div>';
        }

        const scrub = hasScrub ? '<div class="results-scrub">' +
            '<span class="results-scrub__label">Training <b class="results-scrub__val">' + trainingPct + '%</b></span>' +
            '<input type="range" class="results-scrub__slider" min="0" max="100" step="10" value="' + trainingPct + '" aria-label="Training progress">' +
            '<span class="results-keyhint" title="Use the left/right arrow keys to scrub"><kbd>&larr;</kbd><kbd>&rarr;</kbd></span>' +
            '<span class="results-scrub__hint">drag or use arrow keys</span></div>' : '';
        mediaPanel.innerHTML = scrub + body + (note ? '<p class="results-media__note">' + note + '</p>' : '');
        const slider = mediaPanel.querySelector('.results-scrub__slider');
        if (slider) slider.addEventListener('input', function () { applyTrainingPct(+this.value, this); });
        mediaPanel.querySelectorAll('video').forEach(v => { try { v.playbackRate = 3; v.play().catch(function () {}); } catch (e) {} });
        if (hasScrub) applyTrainingPct(trainingPct);          // sync freshly-rendered videos + every scrubber UI
    }
    window.renderStage = renderStage;
    // claim media still routes through the claim cells (used by apply())
    window.renderClaimMedia = function (key) { renderStage(CLAIM_CELLS[key], CLAIM_NOTE[key]); };

    // stage grouping toggle: None / Method / Task — re-render the current stage with the new grouping
    document.querySelectorAll('#results-stage-opts .stage-group-btn').forEach(btn => btn.addEventListener('click', () => {
        stageGroup = btn.dataset.group;
        document.querySelectorAll('#results-stage-opts .stage-group-btn').forEach(b => b.classList.toggle('is-on', b === btn));
        renderStage(lastCells, lastNote);
    }));
    // size slider: scale the staged videos by driving --stage-tile-w on the media panel
    const sizeSlider = document.getElementById('stage-size');
    if (sizeSlider && mediaPanel) {
        const applySize = () => mediaPanel.style.setProperty('--stage-tile-w', sizeSlider.value + 'px');
        sizeSlider.addEventListener('input', applySize);
        applySize();
    }
    // full-width toggle: break the explorer + stage out to the viewport width
    const fwBtn = document.getElementById('results-fullwidth');
    if (fwBtn) fwBtn.addEventListener('click', () => {
        const on = document.body.classList.toggle('results-fullwidth');
        fwBtn.classList.toggle('is-on', on);
        fwBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
        if (typeof refreshCharts === 'function') refreshCharts();   // re-fit any visible plots to the new width
    });

    // ---- table as a control surface: clicking selectable cells / row & column headers picks the videos ----
    const selected = new Set();                        // entries: "<method>|<col>"
    // mark every cell that maps to a clip, so it reads as clickable
    bodyRows.forEach(tr => Array.prototype.forEach.call(tr.children, td => {
        if (td.dataset.col && td.dataset.col !== 'method' && cellVideo(tr.dataset.method, td.dataset.col)) td.classList.add('tv-pick');
    }));
    function selectableCols(methodKey) {
        return COLS.filter(c => c !== 'method' && cellVideo(methodKey, c));
    }
    function syncSelection() {
        bodyRows.forEach(tr => Array.prototype.forEach.call(tr.children, td => {
            if (!td.dataset.col || td.dataset.col === 'method') return;
            td.classList.toggle('tv-selected', selected.has(tr.dataset.method + '|' + td.dataset.col));
        }));
        if (selected.size) {                           // a manual selection overrides any active claim
            table.classList.remove('tv-active');
            cells.forEach(c => c.classList.remove('tv-dim', 'tv-focus'));
            caption.classList.remove('is-on');
            claimBtns.forEach(b => b.classList.remove('is-on'));
            const list = COLS.filter(c => c !== 'method').reduce((acc, col) => {   // stable order: by column, then table row order
                bodyRows.forEach(tr => { if (selected.has(tr.dataset.method + '|' + col)) acc.push([tr.dataset.method, col]); });
                return acc;
            }, []);
            renderStage(list, 'Custom selection — ' + list.length + (list.length === 1 ? ' video' : ' videos') + ' chosen from the table.');
            if (typeof window.renderSelectionPlot === 'function') window.renderSelectionPlot(list);   // plot the picked cells
        } else if (active) {
            apply(active);                             // fall back to the active claim
        } else {
            renderStage([], '');                       // neutral
            if (typeof window.renderClaimPlot === 'function') window.renderClaimPlot('__none');
        }
    }
    function toggleCell(methodKey, col) {
        const k = methodKey + '|' + col;
        if (selected.has(k)) selected.delete(k); else selected.add(k);
        syncSelection();
    }
    function toggleBulk(keys) {                         // select all if any missing, else clear them
        const allOn = keys.every(k => selected.has(k));
        keys.forEach(k => { if (allOn) selected.delete(k); else selected.add(k); });
        syncSelection();
    }
    table.addEventListener('click', e => {
        const td = e.target.closest('td, th');
        if (!td || !table.contains(td)) return;
        const inBody = td.closest('tbody');
        if (inBody) {
            const tr = td.closest('tr[data-method]');
            if (!tr) return;
            const method = tr.dataset.method, col = td.dataset.col;
            if (col === 'method') {                     // row header -> all that method's selectable cells
                const keys = selectableCols(method).map(c => method + '|' + c);
                if (keys.length) toggleBulk(keys);
            } else if (cellVideo(method, col)) {
                toggleCell(method, col);
            }
        } else if (td.dataset.col) {                    // column (sub-)header -> all selectable cells in it
            const col = td.dataset.col;
            const keys = bodyRows.filter(tr => cellVideo(tr.dataset.method, col)).map(tr => tr.dataset.method + '|' + col);
            if (keys.length) toggleBulk(keys);
        }
    });

    // research question -> stage the union of its claims' cells, focus their union on the table (plot
    // stays a hint, since RQ-level shows everything; click a claim to narrow + plot)
    function applyRQ(rq) {
        const claims = RQ_CLAIMS[rq];
        if (!claims) return;
        selected.clear();
        active = null; window.resultsActiveClaim = null;
        const seen = new Set(), unionCells = [], rowSet = new Set(), colSet = new Set();
        claims.forEach(k => {
            (CLAIM_CELLS[k] || []).forEach(c => { const id = c[0] + '|' + c[1]; if (!seen.has(id)) { seen.add(id); unionCells.push(c); } });
            (CLAIMS[k].rows || []).forEach(r => rowSet.add(r));
            (CLAIMS[k].cols || []).forEach(c => colSet.add(c));
        });
        showRQ(rq);
        focusTable(rowSet, colSet);
        caption.textContent = 'Research question — all supporting evidence. Click a finding to narrow the view.';
        caption.classList.add('is-on');
        bodyRows.forEach(tr => Array.prototype.forEach.call(tr.children, td => td.classList.remove('tv-selected')));
        renderStage(unionCells, '');
        if (typeof window.renderClaimPlot === 'function') window.renderClaimPlot('__none');
    }
    window.applyResultsRQ = applyRQ;

    // ---- Focus / Undirected mode: Undirected reveals the full Training section as "all the evidence".
    //      There are two mode toggles (the selector + the training bar), so wire them by class. ----
    function setResultsMode(mode) {
        const undirected = (mode === 'undirected');
        document.body.classList.toggle('results-undirected', undirected);
        document.querySelectorAll('.results-mode-btn').forEach(b => b.classList.toggle('is-on', b.dataset.mode === mode));
        if (undirected) {
            clear();                                                       // show the full, un-dimmed table
            if (typeof refreshCharts === 'function') refreshCharts();      // size the now-visible plots
            const sl = document.getElementById('iteration-slider');        // nudge the just-revealed scrubber videos
            if (sl && typeof updateIteration === 'function') updateIteration(sl.value);
            document.querySelectorAll('#efficiency-section video').forEach(v => { try { v.play().catch(function () {}); } catch (e) {} });
        } else if (window.resultsActiveClaim) {
            apply(window.resultsActiveClaim);                              // re-focus the active claim
        }
    }
    document.querySelectorAll('.results-mode-btn').forEach(btn => btn.addEventListener('click', () => setResultsMode(btn.dataset.mode)));

    // bottom-bar training scrubber (mirrors the stage scrubber; surfaced when that one is off-screen)
    const railSlider = document.getElementById('results-railscrub-slider');
    if (railSlider) railSlider.addEventListener('input', function () { applyTrainingPct(+this.value, this); });

    // arrow-key control of the training scrubber while the results section is in view
    const resultsSec = document.getElementById('results-overview-section');
    window.addEventListener('keydown', function (e) {
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        const tag = (e.target && e.target.tagName || '').toLowerCase();
        if ((tag === 'input' && !e.target.classList.contains('results-scrub__slider')) || tag === 'textarea' || tag === 'select') return;
        if (!resultsSec) return;
        const r = resultsSec.getBoundingClientRect();
        if (!(r.top < window.innerHeight * 0.6 && r.bottom > 130)) return;        // only while in the section
        if (!document.querySelector('#results-media .results-scrub')) return;     // nothing to scrub
        e.preventDefault();
        applyTrainingPct(trainingPct + (e.key === 'ArrowRight' ? 10 : -10));
    });

    window.resultsTunnelClear = clear;
    apply('world');                                    // default: Q1, world-modeling claim, Focus mode
}

// Method section — the architecture figure is an interactive selector that focuses the matching
// step(s) of the training algorithm (Alg. 1) beside it, with a plain-language explanation.
function initMethodExplorer() {
    const root = document.getElementById('method-explorer');
    if (!root || root.dataset.init === '1') return;
    const svgHost = document.getElementById('method-figure-svg');
    const hot = document.getElementById('method-figure-hot');
    const algoEl = document.getElementById('method-algo');
    const explainEl = document.getElementById('method-explain');
    if (!svgHost || !hot || !algoEl || !explainEl) return;
    if (!window.katex) { return void setTimeout(initMethodExplorer, 200); }   // wait for KaTeX (deferred)
    root.dataset.init = '1';

    const ktx = (tex) => { try { return katex.renderToString(tex, { throwOnError: false }); } catch (e) { return tex; } };
    const renderInline = (s) => s.replace(/\\\((.+?)\\\)/g, (_, m) => ktx(m));

    // component palette (matches the site branding)
    const COL = { task: '#3F9E78', encoder: '#5BB1E8', dynamics: '#E0705A', reward: '#7FB069',
                  value: '#E0A53B', policy: '#A99BF5', planner: '#E0A53B', replay: '#7C8597' };

    // each component: figure region(s) (in the 1017×598 viewBox) for the hover hotspots + explanation.
    // (the actual lifted cells are resolved per-shape by colour + position in computeCells.)
    const COMP = {
        task:     { label: 'Task Observations', regions: [[2, 2, 996, 141], [730, 172, 90, 86]],
                    desc: 'Expert demonstrations &mdash; <strong>observation-only</strong>, with no rewards or action labels.' },
        encoder:  { label: 'Encoder', regions: [[244, 338, 206, 134]], ref: ['4a', 'Architecture'],
                    desc: 'Maps a raw observation into a compact latent state \\(z = e(o)\\). All planning and learning happen in this latent space.' },
        dynamics: { label: 'Dynamics', regions: [[468, 228, 78, 254]], ref: ['4a', 'Architecture'],
                    desc: 'The learned world model \\(f(z,a)\\) predicts the next latent state, letting the agent imagine rollouts without touching the real world.' },
        value:    { label: 'Value', regions: [[578, 222, 72, 300]], ref: ['5', 'Architecture'],
                    desc: 'The off-policy value \\(Q(z,a)\\) bootstraps long-horizon return beyond the planning horizon, learned from replayed experience.' },
        reward:   { label: 'Inferred Reward', regions: [[512, 336, 100, 176]], ref: ['4b', 'Architecture'],
                    desc: 'An adversarial (IRL) reward \\(r(z,z\')\\) scores transitions by how expert-like they look &mdash; inferred, never hand-designed.' },
        policy:   { label: 'Policy', regions: [[348, 472, 128, 90]], ref: ['5', 'Architecture'],
                    desc: 'A multi-step policy \\(\\pi(a\\mid z)\\) proposes action sequences; it warm-starts the planner and is optimized against the value.' },
        // planner = the initial observation / robot (panel ①), the imagined MPPI rollouts, the camera,
        // and where the robot actually executes the plan (right scene)
        planner:  { label: 'Planner (MPPI)', regions: [[360, 240, 130, 296], [490, 360, 160, 172], [700, 292, 290, 270]], video: true, ref: ['mppi', 'Algorithm'],
                    desc: 'At act-time, MPPI rolls randomly sampled- and policy-proposal plans through the dynamics, scores them with reward + value, and executes a robust plan \\(\\widehat{\\Pi}\\) on the robot.' },
        replay:   { label: 'Replay Buffer', regions: [[684, 158, 298, 142]],
                    desc: 'Real interactions \\((o,a,o\')\\) are stored and replayed, so every component learns <strong>off-policy</strong> and sample-efficiently.' }
    };

    // ---- algorithm (Alg. 1, Training). Each line is tagged with the component(s) it touches. ----
    const REQ = [
        ['task',     'r', '\\mathcal{D}\\subset\\mathcal{O}\\times\\mathcal{O}', 'Task observations'],
        ['encoder',  'r', 'e:\\mathcal{O}\\to\\mathcal{Z}', 'Encoder'],
        ['dynamics', 'r', 'f:\\mathcal{Z}\\times\\mathcal{A}\\to\\mathcal{Z}', 'Dynamics'],
        ['reward',   'r', 'r:\\mathcal{Z}\\times\\mathcal{Z}\\to\\mathbb{R}', 'Inferred reward'],
        ['value',    'r', 'Q:\\mathcal{Z}\\times\\mathcal{A}\\to\\mathbb{R}', 'Value'],
        ['policy',   'r', 'a\\sim\\pi(\\cdot\\mid z)', 'Policy'],
        ['planner',  'r', 'a\\sim\\widehat{\\Pi}(\\cdot\\mid a,z\\,;f,r,Q,\\pi)', 'Planner'],
        ['replay',   'r', '\\mathcal{B}\\coloneqq\\{\\}', 'Replay buffer']
    ];
    // [components, indent, text, optional equation tex]
    const BODY = [
        ['planner',         0, 'Interact using the <b>planner</b>', null],
        ['replay',          0, 'Store experience', '\\mathcal{B}\\gets\\mathcal{B}\\cup\\{(o_t,a_t,o_{t+1})\\}_{t=1}^{T}'],
        ['__for',           0, '<span class="algo-kw">for</span> updates per episode <span class="algo-kw">do</span>', null],
        ['replay task',     1, 'Sample trajectories &amp; task observations', '\\{(o_t,a_t,o_{t+1})\\}\\sim\\mathcal{B}\\,,\\ (o,o\')\\in\\mathcal{D}'],
        ['encoder dynamics',1, 'Update <b>Encoder</b> &amp; <b>Dynamics</b>', '\\mathcal{L}_{e,f}=\\mathbb{E}_{\\tau}\\!\\big[\\textstyle\\sum_{t\'}\\rho^{\\,t\'-t}\\lVert\\hat z_{t\'}-\\mathrm{sg}(z_{t\'})\\rVert_2^2\\big]'],
        ['reward',          1, 'Update <b>Inferred Reward</b>', '\\mathcal{L}_{r}=\\mathbb{E}_{(z,z\')\\sim\\tau}[r]-\\mathbb{E}_{d}[r]+\\beta\\,\\mathrm{GP}(r,\\tau,d)'],
        ['value',           1, 'Update <b>Value</b>', '\\mathcal{L}_{Q}=\\mathbb{E}\\big[(q_t-\\bar G^{\\lambda}_t)^2\\big]'],
        ['policy',          1, 'Update <b>Policy</b>', '\\mathcal{L}_{\\pi}=-\\mathbb{E}_{\\hat\\tau}\\big[G^{\\lambda}_t\\big]']
    ];

    // ---- build the algorithm panel ----
    let n = 0, h = '<div class="algo__title"><b>MPAIL2</b> (Training)</div><div class="algo__req">';
    h += '<div class="algo-line algo-kwline"><span class="algo-kw">require</span></div>';
    REQ.forEach(([comp, , tex, name]) => {
        h += '<div class="algo-line algo-step algo-i1" data-comp="' + comp + '" tabindex="0"><span class="algo-eqi">' + ktx(tex) + '</span><span class="algo-cmt">' + name + '</span></div>';
    });
    h += '</div><div class="algo-line algo-kwline"><span class="algo-kw">while</span> learning <span class="algo-kw">do</span></div>';
    BODY.forEach(([comp, indent, text, tex]) => {
        if (comp === '__for') { h += '<div class="algo-line algo-kwline algo-i1">' + text + '</div>'; return; }
        n += 1;
        h += '<div class="algo-line algo-step algo-i' + indent + '" data-comp="' + comp + '" tabindex="0">' +
            '<span class="algo-num">' + n + '</span><span class="algo-do">' + text + '</span>' +
            (tex ? '<span class="algo-eq">' + ktx(tex) + '</span>' : '') + '</div>';
    });
    h += '<div class="algo-line algo-kwline algo-i1"><span class="algo-kw">end for</span></div>' +
         '<div class="algo-line algo-kwline"><span class="algo-kw">end while</span></div>';
    algoEl.innerHTML = h;

    // ---- build the figure hotspots (one transparent rect per region, for hover/click) ----
    let sh = '';
    Object.keys(COMP).forEach(id => {
        COMP[id].regions.forEach(([x, y, w, h2]) => {
            sh += '<rect class="method-hot" data-comp="' + id + '" x="' + x + '" y="' + y + '" width="' + w + '" height="' + h2 + '" rx="9"></rect>';
        });
    });
    hot.innerHTML = sh;

    // ---- selection state + visuals ----
    const SVGNS = 'http://www.w3.org/2000/svg';
    const figureEl = document.getElementById('method-figure');
    const methodVideo = document.querySelector('#method-explorer .method-video');
    const algoLines = Array.prototype.slice.call(algoEl.querySelectorAll('.algo-line[data-comp]'));
    const hotRects = Array.prototype.slice.call(hot.querySelectorAll('.method-hot'));
    const defaultExplain = explainEl.innerHTML;
    let figInner = null, veil = null, liftLayer = null, hiddenOriginals = [];
    const cellEls = {};                                  // per component: the actual drawio shape cells inside it
    const flatCells = new Set();                          // cells lifted without the scale pop (wide strip border / caption)
    // the drawio component captions (Encoder, Dynamics, …) are exported as full-canvas label groups that
    // can't be located by bbox — route each to its component by exact text (lifted flat, no scale pop)
    const LABEL_TEXT = { Encoder: 'encoder', Dynamics: 'dynamics', Value: 'value', Reward: 'reward',
                         Policy: 'policy', Plan: 'planner', Experience: 'replay' };
    const labelOwner = (t) => (/Task\s*Observ/i.test(t) ? 'task' : (LABEL_TEXT[t] || null));

    // Resolve which component a shape cell belongs to, by its colour + position. Models claim their own
    // coloured trapezoid + connector(s) (+ z' for dynamics); the planner gets the plan curves, the
    // initial-state robot, the camera and the acting scene — NOT the model connectors or z'.
    function byRegion(cx, cy) {
        if (cx >= 402 && cx <= 452 && cy >= 380 && cy <= 452) return null;     // the small triangle under z_0 — don't lift
        if (cy < 150) return 'task';                                 // observation strip (both photo rows, border, label)
        if (cx >= 728 && cx <= 818 && cy >= 150 && cy <= 224) return 'task';   // (o,o') cylinder only (not the black connector below it)
        if (cx >= 578 && cx <= 652 && cy >= 222 && cy <= 312) return 'value';
        if (cx >= 512 && cx <= 612 && cy >= 338 && cy <= 424) return 'reward';
        if (cx >= 470 && cx <= 544 && cy >= 230 && cy <= 316) return 'dynamics';
        if (cx >= 484 && cx <= 528 && cy >= 386 && cy <= 432) return 'dynamics';   // z' label
        if (cx >= 348 && cx <= 416 && cy >= 472 && cy <= 558) return 'policy';
        if (cx >= 300 && cx <= 446 && cy >= 338 && cy <= 472) return 'encoder';    // e(o), z_0 labels
        if (cx >= 700 && cy >= 290) return 'planner';                              // acting scene + camera (right)
        if (cx >= 358 && cx <= 492 && cy >= 240 && cy <= 384) return 'planner';    // initial-state robot (panel ①)
        if (cx >= 398 && cx <= 662 && cy >= 376 && cy <= 534) return 'planner';    // imagined plan curves
        return null;
    }
    function ownerOf(x, y, w, h, color) {
        const cx = x + w / 2, cy = y + h / 2;
        // the (o,a,o') interaction circles belong to the replay buffer (Store/Sample experience) —
        // o in the left panel ①, a and o' in the right acting panel ③. The action a is ALSO part of the
        // planner (the robot acts → produces a), so it lifts for both.
        if (cx >= 262 && cx <= 300 && cy >= 400 && cy <= 440) return 'replay';            // o  (panel ①)
        if (cx >= 898 && cx <= 950 && cy >= 326 && cy <= 352) return 'planner';           // robot → a connector stub
        if (cx >= 935 && cx <= 978 && cy >= 315 && cy <= 358) return ['replay', 'planner']; // a  (panel ③)
        if (cx >= 945 && cx <= 988 && cy >= 438 && cy <= 478) return 'replay';            // o' (panel ③)
        // Experience replay buffer: the (o,a,o') cylinder + its label text, and the wires carrying a/o'
        // into it (NOT the dashed panel box — that stays dimmed)
        if (cx >= 855 && cx <= 1000 && cy >= 195 && cy <= 318) return 'replay';           // (o,a,o') cylinder + text
        if (cx >= 920 && cx <= 1005 && cy >= 330 && cy <= 470 && color === '#000000') return 'replay';  // a/o' → cylinder wires
        // far-left input scene (the "first panel"): the real robot arm, the manipulated cube and the
        // camera are not part of any model — never lift them (the raw observation o is replay, above)
        if (cx < 300 && cy >= 300) return null;
        // acting scene (third panel): the planner highlights only the arm + its movement (the "Plan"
        // arrow) — NOT the base vertical link, the cube, the camera, the table, or the o'/a circles
        if (cx >= 700 && cy >= 290) {
            if (cx >= 870 && cx <= 905 && cy >= 320 && cy <= 358) return null;  // the round elbow joint — don't lift
            if (cx >= 905) return null;                            // o', a and their wires (far right)
            if (color === '#6c8ebf') return null;                 // the manipulated cube
            if (color === '#000000') return (cy >= 440 && cx <= 885) ? 'planner' : null;   // the "Plan" movement arrow
            if (color === '#666666') return (w < 50 && h > 110) ? null : 'planner';         // arm links (drop the base post)
            return null;                                          // table / camera / misc — not lifted
        }
        switch (color) {
            // each model also owns its mini-box in the Experience panel (by colour)
            case '#6c8ebf': return 'encoder';                        // blue — encoder (+ its mini-box)
            case '#d79b00': return 'value';                          // gold — value
            case '#82b366':                                          // green — the Task strip border + (o,o') cylinder, else the reward
                return (cy < 150 || (cx >= 660 && cy < 255)) ? 'task' : 'reward';
            case '#9673a6':                                          // purple — policy box / connector / mini-box stay policy-only;
                return (cx >= 440 && cx <= 520 && cy >= 430 && cy <= 490) ? ['policy', 'planner'] : 'policy';  // only the rollout proposal dots also lift for the planner
            case '#b85450': return 'dynamics';                       // red — dynamics (trapezoid, z', rollout arrows, mini-box)
        }
        return byRegion(cx, cy);                                     // black/grey/colourless → by position
    }
    function computeCells() {
        if (cellEls.__done || !figInner) return;
        const svgRect = figInner.getBoundingClientRect();
        if (!svgRect.width) return;                      // not laid out yet — retry on next focus
        const sx = 1017 / svgRect.width, sy = 598 / svgRect.height;
        Object.keys(COMP).forEach(id => { cellEls[id] = []; });
        figInner.querySelectorAll('g[data-cell-id]').forEach(g => {
            if (g.querySelector('g[data-cell-id]')) return;          // leaf cells only (individual shapes)
            let bb; try { bb = g.getBoundingClientRect(); } catch (e) { return; }
            const w = bb.width * sx, h = bb.height * sy;
            if (!w || !h) return;
            const x = (bb.left - svgRect.left) * sx, y = (bb.top - svgRect.top) * sy;
            // component caption labels (full-canvas groups) — route by text and lift flat (no scale)
            const lab = labelOwner((g.textContent || '').trim());
            if (lab) { cellEls[lab].push(g); flatCells.add(g); return; }
            const inStrip = (y + h / 2) < 145;                       // the obs strip (two photo rows + green border) is large — keep it
            if ((w > 330 || h > 330) && !inStrip) return;            // otherwise skip oversized background cells
            const sc = g.querySelector('[stroke]'); const color = sc ? (sc.getAttribute('stroke') || '') : '';
            const owner = ownerOf(x, y, w, h, color.toLowerCase());  // a string, an array (shared cell), or null
            (Array.isArray(owner) ? owner : [owner]).forEach(o => {
                if (o && cellEls[o]) { cellEls[o].push(g); if (w > 300) flatCells.add(g); }  // wide cells (strip border) lift flat
            });
        });
        cellEls.__done = true;
    }
    // Encoder & Dynamics are trained jointly (Alg. step 4) — focusing either lifts both as one.
    const MERGE = { encoder: 'encdyn', dynamics: 'encdyn' };
    const GROUP = { encdyn: ['encoder', 'dynamics'] };
    const GROUP_INFO = { encdyn: { label: 'Encoder &amp; Dynamics', col: COL.encoder, ref: ['4a', 'Architecture'],
        desc: 'Trained jointly: the encoder \\(e\\) maps an observation to a latent state and the dynamics \\(f\\) predicts the next latent (loss \\(\\mathcal{L}_{e,f}\\)).' } };
    const expand = (ids) => { const out = []; ids.forEach(id => (MERGE[id] ? GROUP[MERGE[id]] : [id]).forEach(g => { if (out.indexOf(g) < 0) out.push(g); })); return out; };

    // clone the given components' real cells onto a top layer (inside the same drawio layer, so their
    // absolute coordinates still place them correctly) above a dim veil, then raise them
    function lift(ids) {
        if (!liftLayer || !veil) return;
        liftLayer.textContent = '';
        hiddenOriginals.forEach(c => { c.style.opacity = ''; });   // restore any previously-hidden originals
        hiddenOriginals = [];
        const cells = [];
        ids.forEach(id => (cellEls[id] || []).forEach(c => { if (cells.indexOf(c) < 0) cells.push(c); }));
        if (!cells.length) { veil.style.opacity = '0'; return; }
        veil.style.opacity = '1';
        cells.forEach(cell => {
            const flat = flatCells.has(cell);
            const lifter = document.createElementNS(SVGNS, 'g');
            lifter.setAttribute('class', 'method-liftcell' + (flat ? ' method-liftcell--flat' : ''));
            lifter.appendChild(cell.cloneNode(true));
            liftLayer.appendChild(lifter);
            // flat cells (labels / strip border) are raised the SAME size as the original, so the dim
            // original showing through the veil would ghost behind the bright clone — hide it.
            if (flat) { cell.style.opacity = '0'; hiddenOriginals.push(cell); }
        });
        // the planner's "robot → a" connector isn't its own drawio cell — draw it onto the lift layer so
        // it lifts together with the acting arm and the action node a
        if (ids.indexOf('planner') >= 0) {
            const g = document.createElementNS(SVGNS, 'g');
            g.setAttribute('class', 'method-liftcell method-liftcell--flat');
            const ln = document.createElementNS(SVGNS, 'line');
            ln.setAttribute('x1', '906'); ln.setAttribute('y1', '338'); ln.setAttribute('x2', '939'); ln.setAttribute('y2', '338');
            ln.setAttribute('stroke', '#5a5f6a'); ln.setAttribute('stroke-width', '2'); ln.setAttribute('stroke-linecap', 'round');
            g.appendChild(ln); liftLayer.appendChild(g);
        }
        requestAnimationFrame(() => liftLayer.querySelectorAll('.method-liftcell').forEach(l => l.classList.add('is-up')));
    }
    function showIds(ids, primary) {
        ids = expand(ids);
        if (!ids.length || !COMP[primary]) return clearShow();
        const grp = (ids.length === 2 && MERGE[primary]) ? GROUP_INFO[MERGE[primary]] : null;
        const col = grp ? grp.col : COL[primary];
        root.classList.add('is-active');
        computeCells();
        lift(ids);
        hotRects.forEach(r => r.classList.toggle('is-on', ids.indexOf(r.dataset.comp) >= 0));
        if (methodVideo) methodVideo.classList.toggle('is-focus', ids.some(i => COMP[i] && COMP[i].video));
        algoLines.forEach(l => {
            const comps = l.dataset.comp.split(' ');
            const on = comps.some(cc => ids.indexOf(cc) >= 0);
            l.classList.toggle('is-on', on);
            if (on) {
                l.style.setProperty('--c', COL[comps[0]] || col);
                // only an explicit merge group (Encoder & Dynamics) gets the half-half tint
                if (comps.length === 2 && MERGE[comps[0]] && COL[comps[1]]) { l.style.setProperty('--c2', COL[comps[1]]); l.classList.add('is-split'); }
                else { l.style.removeProperty('--c2'); l.classList.remove('is-split'); }
            } else { l.style.removeProperty('--c'); l.style.removeProperty('--c2'); l.classList.remove('is-split'); }
        });
        const info = grp || COMP[primary];
        const refHtml = info.ref ? ' <button type="button" class="method-figref" data-fig="' + info.ref[0] + '" title="Open the training-objective figure">' + info.ref[1] + '&nbsp;&rsaquo;</button>' : '';
        explainEl.innerHTML = '<p class="method-explain__title" style="color:' + col + '">' + info.label + '</p>' +
            '<p class="method-explain__body">' + renderInline(info.desc) + refHtml + '</p>';
        cueFigtab(info.ref ? info.ref[0] : null, col);
    }
    // ring the file tab that documents the focused component, tinted to its colour
    function cueFigtab(fig, col) {
        document.querySelectorAll('#method-figtabs .figtab').forEach(t => {
            const on = !!fig && t.dataset.fig === fig;
            t.classList.toggle('is-cued', on);
            if (on && col) t.style.setProperty('--cue', col); else t.style.removeProperty('--cue');
        });
    }
    const show = (id) => showIds([id], id);
    function clearShow() {
        root.classList.remove('is-active');
        if (veil) veil.style.opacity = '0';
        if (liftLayer) liftLayer.textContent = '';
        hiddenOriginals.forEach(c => { c.style.opacity = ''; }); hiddenOriginals = [];
        hotRects.forEach(r => r.classList.remove('is-on'));
        if (methodVideo) methodVideo.classList.remove('is-focus');
        algoLines.forEach(l => { l.classList.remove('is-on', 'is-split'); l.style.removeProperty('--c'); l.style.removeProperty('--c2'); });
        explainEl.innerHTML = defaultExplain;
        cueFigtab(null);
    }
    let pinnedIds = null, pinnedPrimary = null;
    const leave = () => { if (pinnedIds) showIds(pinnedIds, pinnedPrimary); else clearShow(); };
    function togglePin(ids, primary) {
        if (pinnedPrimary === primary) { pinnedIds = pinnedPrimary = null; clearShow(); }
        else { pinnedIds = ids; pinnedPrimary = primary; showIds(ids, primary); }
    }
    // explicit setter so the appendix "file tabs" can pin/clear a focus (e.g. the MPPI tab → step 1, planner)
    window.methodSetFocus = (ids, primary) => {
        if (ids && ids.length) { pinnedIds = ids; pinnedPrimary = primary || ids[0]; showIds(pinnedIds, pinnedPrimary); }
        else { pinnedIds = pinnedPrimary = null; clearShow(); }
    };

    hotRects.forEach(r => {
        r.addEventListener('mouseenter', () => showIds([r.dataset.comp], r.dataset.comp));
        r.addEventListener('click', e => { e.stopPropagation(); togglePin([r.dataset.comp], r.dataset.comp); });
    });
    algoLines.forEach(l => {
        const comps = l.dataset.comp.split(' ');
        l.addEventListener('mouseenter', () => showIds(comps, comps[0]));
        l.addEventListener('mouseleave', leave);
        l.addEventListener('click', () => togglePin(comps, comps[0]));
        l.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); togglePin(comps, comps[0]); } });
    });
    hot.addEventListener('mouseleave', leave);
    hot.addEventListener('click', e => { if (e.target === hot) { pinnedIds = pinnedPrimary = null; clearShow(); } });

    // a focus description's figure reference opens the matching appendix file tab
    explainEl.addEventListener('click', e => {
        const ref = e.target.closest('.method-figref');
        if (!ref) return;
        const tab = document.querySelector('.figtab[data-fig="' + ref.dataset.fig + '"]');
        if (tab) { tab.click(); tab.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
    });

    // ---- inject the architecture SVG (inline so its light-dark() follows the page theme), then append a
    //      dim veil + a top layer (inside the same SVG) where focused cells are cloned and raised ----
    fetch(root.dataset.svg).then(r => r.text()).then(txt => {
        svgHost.innerHTML = txt;
        const inner = svgHost.querySelector('svg');
        if (!inner) return;
        figInner = inner;
        inner.removeAttribute('width'); inner.removeAttribute('height');
        inner.style.width = '100%'; inner.style.height = 'auto'; inner.style.display = 'block';
        // veil dims everything painted before it; the lift layer (after it) stays bright
        veil = document.createElementNS(SVGNS, 'rect');
        veil.setAttribute('class', 'method-veil');
        veil.setAttribute('x', '0'); veil.setAttribute('y', '0'); veil.setAttribute('width', '1017'); veil.setAttribute('height', '598');
        veil.style.opacity = '0';
        liftLayer = document.createElementNS(SVGNS, 'g');
        liftLayer.setAttribute('class', 'method-lift-layer');
        // place the veil + lift layer INSIDE the drawio content layer so clones share the cells'
        // coordinate space (their absolute path coords then render in place)
        const layer = inner.querySelector('g[data-cell-id="1"]') || inner.querySelector('g[data-cell-id="0"]') || inner;
        layer.appendChild(veil); layer.appendChild(liftLayer);
    }).catch(() => {});
}

// Baseline Models — interactive component diagram
function initBaselineDiagram() {
    const picker = document.getElementById('baseline-picker');
    const svg = document.getElementById('baseline-diagram');
    if (!picker || !svg) return;
    if (picker.dataset.initialized === '1') return;
    picker.dataset.initialized = '1';

    // Per baseline:
    //   planner/dynamics/offpolicy : MPAIL2's three pillars [P]/[M]/[O]
    //   reward  : learns an *inferred* reward (IRL).  value : has a critic.
    //   online  : interacts with the environment.  sup/actions/handR : human supervision.
    const BASELINES = {
        mpail2: { name: 'MPAIL2', planner: true,  dynamics: true,  offpolicy: true,  reward: true, value: true, online: true,
            info: 'The full method: latent planning, a learned dynamics model, and off-policy value learning &mdash; learning visual manipulation from observation alone.' },
        o:      { name: '[&minus;O] MPAIL', planner: true,  dynamics: true,  offpolicy: false, reward: true, value: true, online: true,
            info: 'Drops off-policy learning. It still plans with a dynamics model, but learns an <em>on-policy</em> state value from fresh rollouts (no replay).' },
        p:      { name: '[&minus;P] MAIRL', planner: false, dynamics: true,  offpolicy: true,  reward: true, value: true, online: true,
            info: 'Removes the planner. The dynamics model and off-policy value remain, but actions come straight from the policy with no online planning.' },
        pm:     { name: '[&minus;PM] DAC', planner: false, dynamics: false, offpolicy: true,  reward: true, value: true, online: true,
            info: 'No planner and no dynamics model: a model-free, off-policy actor&ndash;critic with an adversarial (inferred) reward only.' },
        pmo:    { name: '[&minus;PMO] AIRL', planner: false, dynamics: false, offpolicy: false, reward: true, value: true, online: true,
            info: 'Strips all three pillars: model-free, on-policy, no planning &mdash; the most basic adversarial IRL baseline.' },
        rlpd:   { name: 'RLPD', planner: false, dynamics: false, offpolicy: true,  reward: false, value: true, online: true,
            sup: true, actions: true, handR: true, intValue: true,
            info: 'RL from prior data. Given <em>hand-designed rewards</em> and <em>demonstrated actions</em> (human supervision), it learns an off-policy Q&#8209;value and policy, replaying real interaction. No reward inference, dynamics model, or planner.' },
        bc:     { name: 'BC (Diffusion)', planner: false, dynamics: false, offpolicy: false, reward: false, value: false, online: false,
            sup: true, actions: true, handR: false, acts: true, deadInt: true, noIntEnc: true,
            info: 'Behavior cloning: supervised learning from <em>demonstrated actions</em>. The policy can act in the environment, but that interaction is never learned from &mdash; the loop is severed (offline).' }
    };

    const infoTitle = document.getElementById('bl-info-title');
    const infoBody = document.getElementById('bl-info-body');
    const pillars = document.getElementById('baseline-pillars');

    // mini branding diagram from a baseline's component flags (gold Planner enclosure when it plans,
    // wrapping Dynamics(red) Reward(green inferred / gold hand) Value(gold) Policy(purple))
    let blHaz = 0;                                      // unique-id counter for hazard-stripe patterns
    function baselineDiagram(b) {
        const parts = [];
        if (b.dynamics) parts.push(['D', '#E0705A']);
        // hand-designed reward (RLPD) gets hazard stripes — it's given, not learned
        if (b.reward) parts.push(['R', '#7FB069']); else if (b.handR) parts.push(['R', '#E0A53B', true]);
        if (b.value) parts.push(b.offpolicy ? ['Q', '#E0A53B'] : ['V', '#5BB1E8']);   // off-policy Q (gold) vs on-policy V (blue)
        parts.push(['P', '#A99BF5', !b.online]);       // every baseline has a policy; offline/severed (BC) gets hazard stripes
        const bw = 14, bh = 15, gap = 3, pad = b.planner ? 4 : 0, SC = 1.5;   // render scale (boxes stay uniform)
        const W = parts.length * bw + (parts.length - 1) * gap + pad * 2, H = bh + pad * 2;
        let defs = '', body = '';
        parts.forEach((p, i) => {
            const x = pad + i * (bw + gap);
            body += '<rect x="' + x + '" y="' + pad + '" width="' + bw + '" height="' + bh + '" rx="3.5" fill="' + p[1] + '26" stroke="' + p[1] + '" stroke-width="1.3"/>';
            if (p[2]) {                                // diagonal "hazard" stripes behind the letter
                const pid = 'bhaz' + (blHaz++);
                defs += '<pattern id="' + pid + '" patternUnits="userSpaceOnUse" width="3" height="3" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="3" stroke="' + p[1] + '" stroke-width="1.4"/></pattern>';
                body += '<rect x="' + x + '" y="' + pad + '" width="' + bw + '" height="' + bh + '" rx="3.5" fill="url(#' + pid + ')" opacity="0.6"/>';
            }
            body += '<text x="' + (x + bw / 2) + '" y="' + (pad + bh / 2) + '" text-anchor="middle" dominant-baseline="central" font-size="8.5" font-weight="700" fill="' + p[1] + '">' + p[0] + '</text>';
        });
        let s = '<svg class="mini-diagram" width="' + ((W + 2) * SC).toFixed(1) + '" height="' + ((H + 2) * SC).toFixed(1) + '" viewBox="-1 -1 ' + (W + 2) + ' ' + (H + 2) + '" aria-hidden="true">';   // 1-unit margin so box strokes aren't clipped
        if (defs) s += '<defs>' + defs + '</defs>';
        if (b.planner) s += '<rect x="0.7" y="0.7" width="' + (W - 1.4).toFixed(1) + '" height="' + (H - 1.4).toFixed(1) + '" rx="5" fill="none" stroke="#E0A53B" stroke-width="1.3"/>';
        return s + body + '</svg>';
    }
    picker.querySelectorAll('.baseline-btn[data-baseline]').forEach(btn => {
        const b = BASELINES[btn.dataset.baseline];
        if (b) btn.insertAdjacentHTML('afterbegin', '<span class="baseline-btn__diagram">' + baselineDiagram(b) + '</span>');
    });

    const setComp = (name, on) => {
        svg.querySelectorAll('.bl-comp[data-comp="' + name + '"]').forEach(el => {
            el.classList.toggle('is-off', !on);
        });
    };
    const setPillar = (p, on) => {
        const chip = pillars && pillars.querySelector('.pillar-chip[data-pillar="' + p + '"]');
        if (chip) chip.classList.toggle('on', on);
    };

    // "act in environment" connector: its origin animates between the Planner
    // (planning methods plan + execute actions) and the Policy (model-free methods).
    const fbPath = svg.querySelector('#bl-feedback');
    const FB_PLANNER = { x: 504, y: 230 };   // planner bottom-center
    const FB_POLICY  = { x: 664, y: 207 };   // policy bottom-center
    let fbX = FB_PLANNER.x, fbY = FB_PLANNER.y, fbRAF = null;
    const fbD = (x, y) => roundOrtho('M' + x.toFixed(1) + ',' + y.toFixed(1) + ' V350 H44 V320', 8);
    function setFeedback(planner, animate) {
        const t = planner ? FB_PLANNER : FB_POLICY;
        if (fbRAF) { cancelAnimationFrame(fbRAF); fbRAF = null; }
        if (!fbPath) return;
        if (!animate) { fbX = t.x; fbY = t.y; fbPath.setAttribute('d', fbD(fbX, fbY)); return; }
        const sx = fbX, sy = fbY, dur = 520, t0 = performance.now();
        const step = (now) => {
            let p = Math.min(1, (now - t0) / dur);
            const e = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2; // easeInOutCubic
            fbX = sx + (t.x - sx) * e;
            fbY = sy + (t.y - sy) * e;
            fbPath.setAttribute('d', fbD(fbX, fbY));
            if (p < 1) fbRAF = requestAnimationFrame(step);
        };
        fbRAF = requestAnimationFrame(step);
    }

    const setDim = (name, dim) => {
        svg.querySelectorAll('[data-comp="' + name + '"]').forEach(el => el.classList.toggle('bl-dim', dim));
    };

    function apply(key, animate) {
        const c = BASELINES[key];
        if (!c) return;
        const irl = !!c.reward;                     // inferred-reward (IRL) methods vs RLPD/BC
        setComp('planner', c.planner);
        setComp('dynamics', c.dynamics);
        setComp('dynamics-wire', c.dynamics && c.online);
        setComp('replay', c.offpolicy && c.dynamics);
        // inferred reward + its IRL wiring
        setComp('reward', irl);
        setComp('obs-reward', irl);
        setComp('int-reward', irl && c.online);
        setComp('reward-value', irl && c.value);
        // value head: Q-Value (off-policy) vs state Value (on-policy); absent entirely for BC
        setComp('value-q', c.value && c.offpolicy);
        setComp('value-v', c.value && !c.offpolicy);
        setComp('ac-bidir', c.value && c.offpolicy);
        setComp('ac-single', c.value && !c.offpolicy);
        // RLPD / BC: observation routed straight to value / policy
        setComp('obs-value', !irl && c.value);
        setComp('obs-policy', !irl);
        // human supervision: the Actions + Reward boxes are PERSISTENT (greyed when the current method
        // doesn't use them), so you can always click to add/remove supervision; the dashed enclosure
        // shows for supervision methods.
        setComp('supervision', true);
        svg.querySelectorAll('[data-comp="supervision"]').forEach(el => el.classList.toggle('bl-sup-off', !c.sup));
        setComp('actions', true);
        setComp('hand-reward', true);
        svg.querySelectorAll('[data-comp="actions"]').forEach(el => el.classList.toggle('bl-sup-off', !c.actions));
        svg.querySelectorAll('[data-comp="hand-reward"]').forEach(el => el.classList.toggle('bl-sup-off', !c.handR));
        setComp('actions-policy', !!c.actions);
        setComp('hand-reward-value', !!c.handR && c.value);
        // interaction encoder + its input arrow (the obs encoder is always kept)
        setComp('int-encoder', !c.noIntEnc);
        setComp('int-enc', c.online && !c.noIntEnc);
        setComp('dynamics-wire', c.dynamics && c.online);
        setComp('int-value', !!c.intValue);          // RLPD: interaction replays into the value
        setComp('int-dead', !!c.deadInt);            // BC: severed interaction path
        // "act in environment": online methods, plus BC (it acts but doesn't learn)
        const acts = c.online || c.acts;
        if (fbPath) fbPath.classList.toggle('bl-hidden', !acts);
        if (acts) setFeedback(c.planner, animate);
        setPillar('P', c.planner);
        setPillar('M', c.dynamics);
        setPillar('O', c.offpolicy);
        if (infoTitle) infoTitle.innerHTML = c.name || '';
        if (infoBody) infoBody.innerHTML = c.info || '';
    }

    // ===== interactive: click the model boxes to toggle MPAIL2's pillars and discover baselines =====
    // P (planner) / M (dynamics) / O (off-policy); planner needs a model (P => M). Each {P,M,O}
    // maps to the nearest named IRL baseline; the one unnamed combo (!P,M,!O) snaps to MAIRL.
    const NS = 'http://www.w3.org/2000/svg';
    const PILLAR_MAP = { '111': 'mpail2', '110': 'o', '011': 'p', '001': 'pm', '000': 'pmo', '010': 'p' };
    let PS = { P: true, M: true, O: true };
    let curKey = 'mpail2', lastIRL = 'mpail2';
    const psKey = (p) => (p.P ? 1 : 0) + '' + (p.M ? 1 : 0) + '' + (p.O ? 1 : 0);
    const mkEl = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };

    // dashed, clickable "ghost" placeholders shown where an off dynamics / planner used to be
    const dynGhost = mkEl('g', { class: 'bl-ghost bl-hidden', 'data-ghost': 'M' });
    dynGhost.appendChild(mkEl('rect', { x: 300, y: 143, width: 88, height: 64, rx: 12 }));
    const dgl = mkEl('text', { x: 344, y: 178, class: 'bl-ghost-label' }); dgl.textContent = '+ Dynamics'; dynGhost.appendChild(dgl);
    const planGhost = mkEl('g', { class: 'bl-ghost bl-ghost--pill bl-hidden', 'data-ghost': 'P' });
    const planGhostRect = mkEl('rect', { x: 439, y: 104, width: 130, height: 22, rx: 11 });
    planGhost.appendChild(planGhostRect);
    const pgl = mkEl('text', { x: 504, y: 119, class: 'bl-ghost-label' }); pgl.textContent = '+ Planner'; planGhost.appendChild(pgl);
    svg.appendChild(dynGhost); svg.appendChild(planGhost);

    function syncAfterApply(key) {
        const c = BASELINES[key]; if (!c) return;
        curKey = key;
        if (c.reward) lastIRL = key;            // remember the last inferred-reward (IRL) baseline
        PS = { P: !!c.planner, M: !!c.dynamics, O: !!c.offpolicy };
        const isIRL = !!c.reward;
        dynGhost.classList.toggle('bl-hidden', !(isIRL && !PS.M));
        // planner ghost always shows when planning is off; greyed with "(needs dynamics)" until a model exists
        const planOff = isIRL && !PS.P;
        planGhost.classList.toggle('bl-hidden', !planOff);
        if (planOff) {
            const needsDyn = !PS.M;
            planGhost.classList.toggle('bl-ghost--disabled', needsDyn);
            pgl.textContent = needsDyn ? '+ Planner (needs dynamics)' : '+ Planner';
            const w = needsDyn ? 190 : 130;
            planGhostRect.setAttribute('width', w);
            planGhostRect.setAttribute('x', 504 - w / 2);
        }
        picker.querySelectorAll('.baseline-btn').forEach(b => {
            const on = b.dataset.baseline === key;
            b.classList.toggle('active', on); b.setAttribute('aria-selected', on ? 'true' : 'false');
        });
    }
    function select(key, animate) { apply(key, animate); syncAfterApply(key); }
    function togglePillar(which) {
        const p = Object.assign({}, PS);
        p[which] = !p[which];
        if (which === 'M' && !p.M) p.P = false;   // dropping the model drops planning
        if (which === 'P' && p.P) p.M = true;     // planning needs a model
        select(PILLAR_MAP[psKey(p)], true);
    }
    const clickToggle = (sel, which) => svg.querySelectorAll(sel).forEach(el => {
        el.style.cursor = 'pointer';
        el.addEventListener('click', (e) => { e.stopPropagation(); togglePillar(which); });
    });
    clickToggle('[data-comp="value-q"],[data-comp="value-v"]', 'O');
    clickToggle('[data-comp="dynamics"]', 'M');
    clickToggle('[data-comp="planner"]', 'P');
    dynGhost.addEventListener('click', () => togglePillar('M'));
    planGhost.addEventListener('click', () => { if (PS.M) togglePillar('P'); }); // disabled until a model exists

    // persistent supervision boxes are clickable to add/remove human supervision:
    //   Reward: RLPD <-> BC (and adds the full supervision from an IRL method)
    //   Actions: IRL -> BC (add action demos); from a supervision method, drop back to the last IRL baseline
    svg.querySelectorAll('[data-comp="hand-reward"]').forEach(el => {
        el.style.cursor = 'pointer';
        el.addEventListener('click', (e) => { e.stopPropagation(); select(curKey === 'rlpd' ? 'bc' : 'rlpd', true); });
    });
    svg.querySelectorAll('[data-comp="actions"]').forEach(el => {
        el.style.cursor = 'pointer';
        el.addEventListener('click', (e) => {
            e.stopPropagation();
            select(BASELINES[curKey] && BASELINES[curKey].sup ? lastIRL : 'bc', true);
        });
    });

    picker.querySelectorAll('.baseline-btn').forEach(btn => {
        btn.addEventListener('click', () => select(btn.dataset.baseline, true));
    });

    // ----- hover explainer for each block -----
    const TIPS = {
        obs:      ['Task Observation', 'Demonstration frames of the task being performed. Used only to learn the reward &mdash; the actions are never observed.'],
        int:      ['Direct Interaction', "The robot's own experience from acting in the environment."],
        encoder:  ['Encoder', 'Compresses raw observations into a compact latent state that the world model reasons over.'],
        dynamics: ['Dynamics Model', 'The learned world model: predicts the next latent state from the current state and action &mdash; what makes planning and off-policy value learning possible.'],
        reward:   ['Reward', 'An <em>inferred</em> reward (inverse RL): scores how expert-like a state is, learned adversarially from observation alone.'],
        value:    ['Critic', 'Estimates expected long-term return. Off-policy variants learn a Q&#8209;value from replayed data; on-policy variants learn a state value from fresh rollouts.'],
        policy:   ['Policy', 'The actor: maps the latent state to an action distribution.'],
        planner:  ['Planner', 'Model-predictive planner: imagines candidate action sequences through the dynamics model, scores them with the reward and value, and executes the best plan.'],
        electron: ['Off-policy replay', 'This line indicates that the method also uses the dynamics model to optimize the policy, like Dreamer or MBPO.'],
        supervision: ['Human supervision', 'Signals a person must provide by hand &mdash; demonstrated actions and/or a designed reward. MPAIL2 needs none of these.'],
        actions:  ['Action supervision', "The demonstrator's actions are provided directly (not just observations) &mdash; used by BC and RLPD."],
        'hand-reward': ['Hand-designed reward', 'A reward function written by a human, rather than inferred from demonstrations.']
    };
    // hover explainer -> the lower side box updates dynamically
    const tipTitle = document.getElementById('bl-tip-title');
    const tipBody = document.getElementById('bl-tip-body');
    const TIP_DEFAULT = ['Hover a block', 'Mouse over any block in the diagram to read what it does.'];
    const setTip = (info, el) => {
        if (tipTitle) tipTitle.innerHTML = info[0];
        if (tipBody) tipBody.innerHTML = info[1];
        svg.querySelectorAll('.bl-focus').forEach(x => x.classList.remove('bl-focus'));
        if (el) el.classList.add('bl-focus');
    };
    if (tipTitle && tipBody) {
        svg.addEventListener('mouseover', (e) => {
            const el = e.target.closest && e.target.closest('[data-tip]');
            if (el && TIPS[el.getAttribute('data-tip')]) setTip(TIPS[el.getAttribute('data-tip')], el);
        });
        svg.addEventListener('mouseout', (e) => {
            const el = e.target.closest && e.target.closest('[data-tip]');
            if (el && !el.contains(e.relatedTarget)) setTip(TIP_DEFAULT, null);
        });
        svg.querySelectorAll('[data-tip]').forEach((el) => {
            el.setAttribute('tabindex', '0');
            el.addEventListener('focus', () => setTip(TIPS[el.getAttribute('data-tip')] || TIP_DEFAULT, el));
            el.addEventListener('blur', () => setTip(TIP_DEFAULT, null));
        });
    }

    // ---- round every orthogonal connector's corners ----
    function roundOrtho(d, r) {
        // parse an absolute M/H/V/L path into points
        const pts = []; let cx = 0, cy = 0;
        const re = /([MHVL])\s*(-?[\d.]+)?(?:[ ,]+(-?[\d.]+))?/gi; let m;
        while ((m = re.exec(d))) {
            const c = m[1].toUpperCase();
            if (c === 'M' || c === 'L') { cx = parseFloat(m[2]); cy = parseFloat(m[3]); }
            else if (c === 'H') { cx = parseFloat(m[2]); }
            else if (c === 'V') { cy = parseFloat(m[2]); }
            pts.push([cx, cy]);
        }
        if (pts.length < 3) return d;
        let out = 'M' + pts[0][0] + ',' + pts[0][1];
        for (let i = 1; i < pts.length - 1; i++) {
            const p0 = pts[i - 1], p1 = pts[i], p2 = pts[i + 1];
            const d1 = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
            const d2 = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
            const r1 = Math.min(r, d1 / 2), r2 = Math.min(r, d2 / 2);
            const a = [p1[0] + (p0[0] - p1[0]) / d1 * r1, p1[1] + (p0[1] - p1[1]) / d1 * r1];
            const b = [p1[0] + (p2[0] - p1[0]) / d2 * r2, p1[1] + (p2[1] - p1[1]) / d2 * r2];
            out += ' L' + a[0].toFixed(1) + ',' + a[1].toFixed(1) +
                   ' Q' + p1[0] + ',' + p1[1] + ' ' + b[0].toFixed(1) + ',' + b[1].toFixed(1);
        }
        const last = pts[pts.length - 1];
        out += ' L' + last[0] + ',' + last[1];
        return out;
    }
    svg.querySelectorAll('.bl-wire').forEach(p => {
        if (p.id === 'bl-feedback') return;            // animated separately
        const d = p.getAttribute('d');
        if (d) p.setAttribute('d', roundOrtho(d, 8));
    });

    select('mpail2', false);

    // dev/verification: ?blbaseline=p pre-selects a baseline
    const blPre = new URLSearchParams(location.search).get('blbaseline');
    if (blPre && BASELINES[blPre]) select(blPre, false);
}

// Overview: scroll-driven native diagram that assembles the MPAIL2 architecture
// (the same diagram built in sections/baselines.html), beat by beat, as the script captions advance.
function initStoryScrub() {
    const story = document.getElementById('summary-section');
    if (!story || !story.classList.contains('story')) return;
    if (story.dataset.init === '1') return;
    story.dataset.init = '1';
    const svg = document.getElementById('story-diagram');
    if (!svg) return;
    const caps = Array.prototype.slice.call(story.querySelectorAll('.story__cap'));
    const chips = Array.prototype.slice.call(story.querySelectorAll('.story__chip:not(.story__skip)'));
    const bar = document.getElementById('story-bar');
    const duration = parseFloat(story.dataset.duration) || 120;

    // round orthogonal wire corners (same look as the baselines diagram)
    function roundOrtho(d, r) {
        const pts = []; let cx = 0, cy = 0;
        const re = /([MHVL])\s*(-?[\d.]+)?(?:[ ,]+(-?[\d.]+))?/gi; let m;
        while ((m = re.exec(d))) {
            const c = m[1].toUpperCase();
            if (c === 'M' || c === 'L') { cx = parseFloat(m[2]); cy = parseFloat(m[3]); }
            else if (c === 'H') { cx = parseFloat(m[2]); }
            else if (c === 'V') { cy = parseFloat(m[2]); }
            pts.push([cx, cy]);
        }
        if (pts.length < 3) return d;
        let out = 'M' + pts[0][0] + ',' + pts[0][1];
        for (let i = 1; i < pts.length - 1; i++) {
            const p0 = pts[i - 1], p1 = pts[i], p2 = pts[i + 1];
            const d1 = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
            const d2 = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
            const r1 = Math.min(r, d1 / 2), r2 = Math.min(r, d2 / 2);
            const a = [p1[0] + (p0[0] - p1[0]) / d1 * r1, p1[1] + (p0[1] - p1[1]) / d1 * r1];
            const b = [p1[0] + (p2[0] - p1[0]) / d2 * r2, p1[1] + (p2[1] - p1[1]) / d2 * r2];
            out += ' L' + a[0].toFixed(1) + ',' + a[1].toFixed(1) +
                   ' Q' + p1[0] + ',' + p1[1] + ' ' + b[0].toFixed(1) + ',' + b[1].toFixed(1);
        }
        const last = pts[pts.length - 1];
        out += ' L' + last[0] + ',' + last[1];
        return out;
    }
    svg.querySelectorAll('.bl-wire').forEach(p => {
        const d = p.getAttribute('d');
        if (d && /[HV]/.test(d)) p.setAttribute('d', roundOrtho(d, 8));
    });

    // ----- beat definitions (16). Each `on` lists which data-comp groups are present;
    // label/flag fields drive relabels and the few genuine morphs. The `t` values align
    // with the caption/chapter data-t scale so captions + chapter rail stay in lockstep. -----
    const valLabel = document.getElementById('st-val-label');
    const valSub = document.getElementById('st-val-sub');
    const policyLabel = document.getElementById('st-policy-label');
    const obsPanel = document.getElementById('st-obs');
    const dynBox = svg.querySelector('[data-comp="dynamics"]');

    // ----- per-method paper citation (bottom-right; swaps with the active chapter) -----
    // Indexed to the chapter rail order: [Observation/LfO, IRL, AIL, MPAIL, MPAIL2].
    // Fill label/cite/href per method; an entry with no href keeps the chip hidden.
    const citeEl = document.getElementById('story-cite');
    const CITES = [
        { label: 'LfO',    cite: 'Torabi et al. 2019',  href: 'https://arxiv.org/abs/1905.13566' },
        { label: 'IRL',    cite: 'Abbeel &amp; Ng 2004', href: 'https://doi.org/10.1145/1015330.1015430' },
        { label: 'AIL',    cite: 'Ho &amp; Ermon 2016',  href: 'https://proceedings.neurips.cc/paper_files/paper/2016/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html' },
        { label: 'MPAIL',  cite: 'Han et al. 2026',     href: 'https://arxiv.org/abs/2507.21533' },
        { label: 'MPAIL2', cite: 'This work',           href: '' }
    ];
    let curCite = -2;
    const setCite = (hi) => {
        if (hi === curCite) return; curCite = hi;
        const c = hi >= 0 ? CITES[hi] : null;
        if (!citeEl) return;
        if (!c || (!c.cite && !c.href)) { citeEl.classList.remove('is-on'); return; }
        const arrow = c.href ? '<span class="story__cite-arr">&#8599;</span>' : '';
        citeEl.innerHTML = '<span class="story__cite-m">' + c.label + '</span><span>' + c.cite + '</span>' + arrow;
        if (c.href) { citeEl.href = c.href; citeEl.removeAttribute('aria-disabled'); citeEl.classList.remove('story__cite--plain'); }
        else { citeEl.removeAttribute('href'); citeEl.setAttribute('aria-disabled', 'true'); citeEl.classList.add('story__cite--plain'); }
        citeEl.classList.add('is-on');
    };

    // imgPhase: 'center' = single obs centered (LfO); 'side' = obs+interaction side-by-side (IRL start);
    // 'column' (default) = top/bottom column, positioned by the layout engine.
    const B = (t, on, x) => Object.assign({ t: t, on: on }, x || {});
    const BEATS = [
        B(0,   ['obsPanel'], { imgPhase: 'center' }),
        B(5,   ['obsPanel'], { imgPhase: 'center' }),
        B(12,  ['obsPanel', 'obsDeck'], { imgPhase: 'center' }),               // stack with the LfO con
        B(19,  ['obsPanel', 'intPanel', 'obsDeck'], { imgPhase: 'side', deckCollapse: true }), // IRL begins: side-by-side; data stack collapses
        B(25,  ['obsPanel', 'intPanel', 'reward', 'wObsReward', 'wIntReward'], { deckCollapse: true }), // reward -> column; keep the data stack collapsed as it fades
        B(32,  ['obsPanel', 'intPanel', 'reward', 'wObsReward', 'wIntReward', 'valbox', 'wRewardVal', 'return'], { val: 'RL', sub: '' }),
        B(38,  ['obsPanel', 'intPanel', 'reward', 'wObsReward', 'wIntReward', 'valbox', 'wRewardVal', 'recycle', 'return', 'intDeck'], { val: 'RL', sub: '' }), // IRL con: cycle + interaction stack
        B(49,  ['obsPanel', 'intPanel', 'reward', 'wObsReward', 'wIntReward', 'valbox', 'wRewardVal', 'policy', 'wAc', 'return', 'recycleX', 'intDeck'], { val: 'Value', sub: '' }),
        B(58,  ['obsPanel', 'intPanel', 'reward', 'wObsReward', 'wIntReward', 'valbox', 'wRewardVal', 'policy', 'wAc', 'return', 'intDeck'], { val: 'Value', sub: '', sketchy: true }),
        // MPAIL: dynamics is introduced together with the planner (with "(prior)"); policy is absorbed
        B(66,  ['obsPanel', 'intPanel', 'reward', 'wObsReward', 'wIntReward', 'valbox', 'wRewardVal', 'return', 'planner', 'dynamics', 'intDeck'], { val: 'Value', sub: '' }),
        // MPAIL con: the interaction stack becomes simulation/terminal-style (needs a prior model + sim)
        B(72,  ['obsPanel', 'intPanel', 'reward', 'wObsReward', 'wIntReward', 'valbox', 'wRewardVal', 'return', 'planner', 'dynamics', 'intDeck'], { val: 'Value', sub: '', prior: 'show', sim: true }),
        B(81,  ['obsPanel', 'intPanel', 'reward', 'wObsReward', 'wIntReward', 'valbox', 'wRewardVal', 'return', 'planner', 'dynamics', 'intDeck'], { val: 'Value', sub: '', prior: 'show', sim: true }),
        // online dynamics: encoders appear, "(prior)" is crossed out, interaction -> dynamics connector appears
        B(90,  ['obsPanel', 'intPanel', 'obsEnc', 'intEnc', 'reward', 'wObsReward', 'wIntReward', 'valbox', 'wRewardVal', 'return', 'planner', 'dynamics', 'wIntDyn', 'intDeck'], { val: 'Value', sub: '', prior: 'gone', sim: true }),
        // off-policy: the policy box reappears (added for value optimization); the interaction stack
        // collapses into the panel as it goes away (sample efficiency solved)
        B(98,  ['obsPanel', 'intPanel', 'obsEnc', 'intEnc', 'reward', 'wObsReward', 'wIntReward', 'valbox', 'wRewardVal', 'policy', 'wAcBidir', 'return', 'planner', 'dynamics', 'wIntDyn', 'intDeck'], { val: 'Q-Value', sub: 'off-policy', prior: 'done', sim: true, intDeckCollapse: true }),
        // multi-step policy: the off-policy "electron" replay line appears here
        B(105, ['obsPanel', 'intPanel', 'obsEnc', 'intEnc', 'reward', 'wObsReward', 'wIntReward', 'valbox', 'wRewardVal', 'policy', 'wAcBidir', 'return', 'planner', 'dynamics', 'wIntDyn', 'replay'], { val: 'Q-Value', sub: 'off-policy', prior: 'done', policyName: 'Multi-Step Policy' }),
        B(115, ['obsPanel', 'intPanel', 'obsEnc', 'intEnc', 'reward', 'wObsReward', 'wIntReward', 'valbox', 'wRewardVal', 'policy', 'wAcBidir', 'return', 'planner', 'dynamics', 'wIntDyn', 'replay'], { val: 'Q-Value', sub: 'off-policy', prior: 'done', policyName: 'Multi-Step Policy' })
    ];
    const ALL_COMPS = ['obsPanel', 'intPanel', 'obsEnc', 'intEnc', 'obsDeck', 'intDeck',
        'reward', 'wObsReward', 'wIntReward', 'wRewardVal', 'valbox', 'policy', 'wAc', 'wAcBidir',
        'recycle', 'recycleX', 'return', 'planner', 'dynamics', 'wIntDyn', 'replay'];

    const setComp = (name, on) => {
        svg.querySelectorAll('[data-comp="' + name + '"]').forEach(el => el.classList.toggle('is-off', !on));
    };

    // ---- layout engine ----
    // The whole diagram (input images + encoders + core) is laid out left-to-right with fixed internal
    // gaps and centered as a unit, so when few boxes are present the images move inward (short
    // connectors) and slide back out as the core grows. Reward/Value/Policy ride in #st-rvp; Dynamics
    // is placed independently. In the 'center'/'side' phases the images take bespoke transforms instead.
    const CORE = ['dynamics', 'reward', 'value', 'policy'];
    const CW = { dynamics: 88, reward: 84, value: 88, policy: 88 };  // box widths
    const HOME = { reward: 436, value: 548, policy: 662 };           // home left edge of rvp boxes
    const DYN_HOME = 320, GAP = 28;
    const OBS_C = { x: 110, y: 80 }, INT_C = { x: 110, y: 271 };     // panel centers (for center-scaling)
    const stRvp = document.getElementById('st-rvp');
    const dynGroup = document.getElementById('st-dyn-pos');   // wrapper that carries the layout translate
    const intPanel = svg.querySelector('[data-comp="intPanel"]');
    const obsEncG = svg.querySelector('[data-comp="obsEnc"]');
    const intEncG = svg.querySelector('[data-comp="intEnc"]');
    const obsDeckG = svg.querySelector('[data-comp="obsDeck"]');
    const intDeckG = svg.querySelector('[data-comp="intDeck"]');
    const plRect = document.getElementById('st-planner-rect');
    const plLabel = document.getElementById('st-planner-label');
    const wObs = svg.querySelector('[data-comp="wObsReward"] .bl-wire');
    const wInt = svg.querySelector('[data-comp="wIntReward"] .bl-wire');
    const wObsHead = svg.querySelector('[data-comp="wObsReward"] .st-arrowhead');
    const wIntHead = svg.querySelector('[data-comp="wIntReward"] .st-arrowhead');
    const rewardWireGs = [svg.querySelector('[data-comp="wObsReward"]'), svg.querySelector('[data-comp="wIntReward"]')];
    const wIntDynP = svg.querySelector('[data-comp="wIntDyn"] path');
    const replayP = svg.querySelector('[data-comp="replay"] path');
    const retP = document.getElementById('st-return');
    const counterEl = document.getElementById('story-counter');
    let LS = { encOn: false, plannerOn: false, policyOn: false };
    const KEYS = ['rvpDx', 'dynDx', 'plx', 'plw', 'ply', 'plh', 'plrx', 'obsTx', 'obsTy', 'obsS', 'intTx', 'intTy', 'intS', 'imgDx'];

    function planeFor(beat, onSet) {
        const phase = beat.imgPhase || 'column';
        const S = { rvpDx: 0, dynDx: 0, plx: 295, plw: 480, ply: 120, plh: 110, plrx: 15, obsTx: 0, obsTy: 0, obsS: 1, intTx: 0, intTy: 0, intS: 1, imgDx: 0 };
        if (phase === 'center') {                          // single obs image, centered + enlarged
            S.obsTx = 286; S.obsTy = 121; S.obsS = 1.55; S.intTx = 286; S.intTy = 121;
        } else if (phase === 'side') {                     // obs + interaction side-by-side, centered
            S.obsTx = 181; S.obsTy = 95; S.obsS = 1.08; S.intTx = 391; S.intTy = -96; S.intS = 1.08;
        } else {                                           // column: center the whole diagram
            const present = CORE.filter(c => onSet.has(c === 'value' ? 'valbox' : c));
            let W = GAP * Math.max(0, present.length - 1); present.forEach(c => W += CW[c]);
            const imgDx = 226 - W / 2;                      // centers image(180)+gap(120)+core on viewBox center
            S.imgDx = imgDx; S.obsTx = imgDx; S.intTx = imgDx;
            // dynamics is always the leftmost core box, so its slot is at imgDx whether or not it is
            // present yet. Pre-positioning it there means it EXPANDS in place when the planner introduces
            // it (a pure scale, no slide-in) instead of dropping in from below.
            S.dynDx = imgDx;
            if (present.length) {
                const firstLeft = DYN_HOME + imgDx;        // leftmost core box sits a fixed gap from the images
                let cur = firstLeft; const left = {};
                present.forEach(c => { left[c] = cur; cur += CW[c] + GAP; });
                for (const c of ['reward', 'value', 'policy']) if (c in left) { S.rvpDx = left[c] - HOME[c]; break; }
                const last = present[present.length - 1];
                // planner enclosure wraps every present core box. Before the planner exists it sits
                // exactly on the policy box (same geom + color) so MPAIL's planner can MORPH out of it:
                // the box grows outward and recolors purple->gold instead of cross-fading.
                if (onSet.has('planner')) {
                    S.plx = firstLeft - 25; S.plw = (left[last] + CW[last] + 25) - (firstLeft - 25);
                    S.ply = 120; S.plh = 110; S.plrx = 15;
                } else if (onSet.has('policy')) {
                    S.plx = 662 + S.rvpDx; S.plw = CW.policy; S.ply = 143; S.plh = 64; S.plrx = 12;
                }
            }
        }
        return S;
    }
    const tfStr = (tx, ty, s, c) => (Math.abs(s - 1) < 1e-3)
        ? 'translate(' + tx.toFixed(2) + ',' + ty.toFixed(2) + ')'
        : 'translate(' + tx.toFixed(2) + ',' + ty.toFixed(2) + ') translate(' + c.x + ',' + c.y + ') scale(' + s.toFixed(3) + ') translate(' + (-c.x) + ',' + (-c.y) + ')';
    function renderLayout(S) {
        if (obsPanel) obsPanel.setAttribute('transform', tfStr(S.obsTx, S.obsTy, S.obsS, OBS_C));
        if (obsDeckG) obsDeckG.setAttribute('transform', tfStr(S.obsTx, S.obsTy, S.obsS, OBS_C));
        if (intPanel) intPanel.setAttribute('transform', tfStr(S.intTx, S.intTy, S.intS, INT_C));
        if (intDeckG) intDeckG.setAttribute('transform', tfStr(S.intTx, S.intTy, S.intS, INT_C));
        const encT = 'translate(' + S.imgDx.toFixed(2) + ',0)';
        if (obsEncG) obsEncG.setAttribute('transform', encT);
        if (intEncG) intEncG.setAttribute('transform', encT);
        if (stRvp) stRvp.setAttribute('transform', 'translate(' + S.rvpDx.toFixed(2) + ',0)');
        if (dynGroup) dynGroup.setAttribute('transform', 'translate(' + S.dynDx.toFixed(2) + ',0)');
        const sx = (200 + S.imgDx).toFixed(1);   // feeder runs the full width; the opaque-backed encoder occludes its middle
        const ex = (456 + S.rvpDx).toFixed(1);
        // chevron arrowhead baked into the path tail so it draws (grows) with the connector
        const exn = +ex;
        if (wObs) wObs.setAttribute('d', roundOrtho('M' + sx + ',80 H' + ex + ' V141', 8));
        if (wInt) wInt.setAttribute('d', roundOrtho('M' + sx + ',271 H' + ex + ' V209', 8));
        // filled-triangle arrowheads sized to match the shared st-arr marker (markerWidth 7 x strokeWidth 1.6)
        if (wObsHead) wObsHead.setAttribute('d', 'M' + (exn - 5.6).toFixed(1) + ',132 L' + (exn + 5.6).toFixed(1) + ',132 L' + ex + ',143.2 Z');
        if (wIntHead) wIntHead.setAttribute('d', 'M' + (exn - 5.6).toFixed(1) + ',218 L' + (exn + 5.6).toFixed(1) + ',218 L' + ex + ',206.8 Z');
        if (wIntDynP) wIntDynP.setAttribute('d', roundOrtho('M' + (274 + S.imgDx).toFixed(1) + ',271 H' + (364 + S.dynDx).toFixed(1) + ' V207', 8));
        if (replayP) replayP.setAttribute('d', 'M' + (408 + S.dynDx).toFixed(1) + ',165 H' + (636 + S.rvpDx).toFixed(1));
        if (plRect) {
            plRect.setAttribute('x', S.plx.toFixed(1)); plRect.setAttribute('width', S.plw.toFixed(1));
            plRect.setAttribute('y', S.ply.toFixed(1)); plRect.setAttribute('height', S.plh.toFixed(1));
            plRect.setAttribute('rx', S.plrx.toFixed(1));
        }
        if (plLabel) { plLabel.setAttribute('x', (S.plx + S.plw / 2).toFixed(1)); plLabel.setAttribute('y', (S.ply - 7).toFixed(1)); }
        // act-in-environment return origin: planner > policy > value/RL box; endpoint tracks the interaction
        let ox, oy;
        if (LS.plannerOn) { ox = S.plx + S.plw / 2; oy = 230; }
        else if (LS.policyOn) { ox = 706 + S.rvpDx; oy = 207; }
        else { ox = 592 + S.rvpDx; oy = 207; }
        const ix = (44 + S.imgDx).toFixed(1);             // re-enter the interaction panel wherever it sits
        if (retP) retP.setAttribute('d', roundOrtho('M' + ox.toFixed(1) + ',' + oy + ' V350 H' + ix + ' V320', 8));
    }
    let curS = null, layoutRAF = null;
    const lerp = (a, b, e) => a + (b - a) * e;
    function animateLayout(target, animate) {
        if (!animate || !curS) {
            if (layoutRAF) cancelAnimationFrame(layoutRAF);
            layoutRAF = null; curS = Object.assign({}, target); renderLayout(curS); return;
        }
        const f = Object.assign({}, curS), t0 = performance.now(), dur = 560;
        if (layoutRAF) cancelAnimationFrame(layoutRAF);
        const step = (now) => {
            const p = Math.min(1, (now - t0) / dur);
            const e = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
            const S = {}; KEYS.forEach(k => S[k] = lerp(f[k], target[k], e));
            renderLayout(S); curS = S;
            if (p < 1) layoutRAF = requestAnimationFrame(step); else layoutRAF = null;
        };
        layoutRAF = requestAnimationFrame(step);
    }

    // obs/interaction stacks: fan OUT (animated) when newly appearing, fan IN (collapse, animated) on
    // exit, else stay fanned. A pending fan-out rAF is cancelled first so it can't undo a later collapse.
    let obsDeckWasOn = false, intDeckWasOn = false;
    function setDeckState(group, on, wasOn, collapseFlag, instant) {
        if (group._deckRAF) { cancelAnimationFrame(group._deckRAF); group._deckRAF = 0; group.classList.remove('st-deck-instant'); }
        if (collapseFlag) { group.classList.add('st-collapse'); return; }   // fan in (animates via the .6s transition)
        if (on && !wasOn && !instant) {              // newly appearing -> animate the fan-out from collapsed
            group.classList.add('st-deck-instant', 'st-collapse');
            void group.offsetWidth;                  // commit the collapsed start with transitions off
            group.classList.remove('st-deck-instant');
            group._deckRAF = requestAnimationFrame(() => { group.classList.remove('st-collapse'); group._deckRAF = 0; });
            return;
        }
        group.classList.remove('st-collapse');
    }

    let curBeat = -1;
    function applyBeat(i, instant) {
        if (i === curBeat && !instant) return;
        const first = curBeat === -1;
        curBeat = i;
        const b = BEATS[i];
        const onSet = new Set(b.on);
        ALL_COMPS.forEach(c => setComp(c, onSet.has(c)));
        // relabels
        if (valLabel && b.val) valLabel.textContent = b.val;
        if (valSub) valSub.textContent = b.sub || '';
        if (policyLabel) {
            const pn = b.policyName || 'Policy';
            // "Multi-Step Policy" overflows the box, so wrap it onto two lines
            if (pn === 'Multi-Step Policy') {
                policyLabel.innerHTML = '<tspan x="706" dy="-6.5">Multi-Step</tspan><tspan x="706" dy="15">Policy</tspan>';
            } else {
                policyLabel.textContent = pn;
            }
        }
        // "act in environment" return line turns sketchy/brittle (AIL con), solid otherwise
        svg.querySelectorAll('.st-return').forEach(el => el.classList.toggle('st-sketchy', !!b.sketchy));
        // dynamics "(prior)" crossout + fade once it is learned online (animated via CSS)
        if (dynBox) {
            dynBox.classList.toggle('st-prior-show', b.prior === 'show');
            dynBox.classList.toggle('st-dyn-online', b.prior === 'gone');   // the crossout moment (animates)
            dynBox.classList.toggle('st-dyn-done', b.prior === 'done');     // already online — no strike, statically gone
        }
        // policy -> planner morph: the planner rect carries the policy's purple while planning is absent,
        // then drops st-as-policy when the planner appears so it recolors to gold as it grows out of the
        // policy box. The policy box fades in place (st-morph-out) instead of dropping away beneath it.
        const morphIntoPlanner = onSet.has('planner') && !onSet.has('policy');
        if (plRect) plRect.classList.toggle('st-as-policy', !onSet.has('planner'));
        const policyG = svg.querySelector('[data-comp="policy"]');
        if (policyG) policyG.classList.toggle('st-morph-out', morphIntoPlanner);
        // once the reward (and the architecture) is introduced, fade the input photos into the background
        svg.classList.toggle('st-dim-imgs', onSet.has('reward'));
        // obs/interaction stacks: fan OUT when newly appearing, fan IN (collapse) on exit; sim-style at the MPAIL con
        const obsDeckOn = onSet.has('obsDeck'), intDeckOn = onSet.has('intDeck');
        if (obsDeckG) setDeckState(obsDeckG, obsDeckOn, obsDeckWasOn, !!b.deckCollapse, instant);
        if (intDeckG) {
            intDeckG.classList.toggle('is-sim', !!b.sim);
            setDeckState(intDeckG, intDeckOn, intDeckWasOn, !!b.intDeckCollapse, instant);
        }
        obsDeckWasOn = obsDeckOn; intDeckWasOn = intDeckOn;
        // layout state for renderLayout, then reflow (instant on first paint / forced capture / loop reset)
        LS = { encOn: onSet.has('obsEnc'), plannerOn: onSet.has('planner'), policyOn: onSet.has('policy') };
        const doAnim = !first && !story.dataset.forceP && !instant;
        // play the reward feeder-wire "grow" ONLY on a real arrival at frame 5 (index 4); base state
        // (any other frame) is fully drawn. Toggling the class off elsewhere lets it restart on re-entry.
        rewardWireGs.forEach(g => g && g.classList.toggle('st-grow-anim', i === 4 && doAnim));
        animateLayout(planeFor(b, onSet), doAnim);
        if (doAnim) enterDoneAt = performance.now() + ENTER_MS;   // when this arrival transition will finish
        if (counterEl && !instant) counterEl.textContent = 'frame ' + (i + 1) + ' / ' + BEATS.length;
    }

    // ----- animation loop: while parked on a beat, replay its enter transition on a loop
    // (jump to the start instantly, then play to the end), with a progress bar above the caption. -----
    const animBar = document.getElementById('story-animbar');
    const animBarFill = animBar ? animBar.querySelector('span') : null;
    // a short hold at the start, the animation, then a short hold at the end, before replaying
    const ANIM_MS = 680, START_PAUSE = 340, END_PAUSE = 750, LOOP_MS = START_PAUSE + ANIM_MS + END_PAUSE;
    const ENTER_MS = 620;   // ~duration of a beat's arrival transition (layout tween + deck/style fades)
    let loopRAF = null, loopT0 = 0, loopGen = 0, enterDoneAt = 0, lastScrollP = -1, lastScrollAt = 0;
    // Loop ONLY a beat's specific "style change" animation, in isolation (reset just that property to its
    // "before" instantly, then play it to its "after") — NOT block introductions, relabels, collapses,
    // or layout/phase intros. Returns {reset, apply} or null if the beat shouldn't loop.
    const has = (b, c) => b.on.indexOf(c) >= 0;
    function getReplay(idx) {
        const b = BEATS[idx], p = BEATS[idx - 1] || { on: [] };
        if ((b.imgPhase || 'col') !== (p.imgPhase || 'col')) return null;   // layout/phase intro, not a style loop
        const R = [], A = [];                                               // reset / apply actions for this beat
        // a small element fading in (NOT a model box) — encoders, the interaction->dynamics wire, the ✕
        const appear = (c) => { if (has(b, c) && !has(p, c)) { R.push(() => setComp(c, false)); A.push(() => setComp(c, true)); } };
        const deck = (g) => { R.push(() => { if (g._deckRAF) { cancelAnimationFrame(g._deckRAF); g._deckRAF = 0; } g.classList.add('st-collapse'); }); A.push(() => g.classList.remove('st-collapse')); };
        if (has(b, 'obsDeck') && !has(p, 'obsDeck')) deck(obsDeckG);         // data-stack expansion (fan out)
        if (has(b, 'intDeck') && !has(p, 'intDeck')) deck(intDeckG);         // interaction-stack expansion
        if (b.intDeckCollapse && !p.intDeckCollapse) {                        // interaction-stack collapse (fan in)
            R.push(() => { if (intDeckG._deckRAF) { cancelAnimationFrame(intDeckG._deckRAF); intDeckG._deckRAF = 0; } intDeckG.classList.remove('st-collapse'); });
            A.push(() => intDeckG.classList.add('st-collapse'));
        }
        if (!!b.sim && !p.sim) { R.push(() => intDeckG.classList.remove('is-sim')); A.push(() => intDeckG.classList.add('is-sim')); } // sim/terminal-style
        if (b.prior === 'show' && p.prior !== 'show') { R.push(() => dynBox.classList.remove('st-prior-show')); A.push(() => dynBox.classList.add('st-prior-show')); } // "(prior)" appears
        if (b.prior === 'gone' && p.prior !== 'gone') {                      // "(prior)" crossout
            R.push(() => { dynBox.classList.remove('st-dyn-online'); dynBox.classList.add('st-prior-show'); });
            A.push(() => { dynBox.classList.remove('st-prior-show'); dynBox.classList.add('st-dyn-online'); });
        }
        if (!!b.sketchy && !p.sketchy) {                                     // return connector turns brittle
            R.push(() => svg.querySelectorAll('.st-return').forEach(e => e.classList.remove('st-sketchy')));
            A.push(() => svg.querySelectorAll('.st-return').forEach(e => e.classList.add('st-sketchy')));
        }
        appear('obsEnc'); appear('intEnc'); appear('wIntDyn');
        if (!R.length) return null;                                          // block intro / no style change
        return { reset: () => R.forEach(f => f()), apply: () => A.forEach(f => f()) };
    }
    function stopLoop() {
        loopGen++;                                          // invalidate any in-flight cycle frames
        if (loopRAF) { cancelAnimationFrame(loopRAF); loopRAF = null; }
        svg.classList.remove('st-no-anim');                 // never leave transitions globally disabled
        if (animBar) animBar.classList.remove('is-on');
    }
    function startLoop(beatIdx) {
        stopLoop();
        if ((story.dataset.forceP && !story.dataset.testLoop) || beatIdx <= 0) return;
        const rep = getReplay(beatIdx);
        if (!rep) return;
        const gen = ++loopGen;
        // The arrival ("enter") transition is the animation's FIRST play. So the loop opens in an
        // 'enter' phase: it lets that arrival transition finish and then holds a full END_PAUSE before
        // the first reset — otherwise the idle interval (which fires ~240ms after scrolling stops, mid
        // arrival) would snap the half-played animation back to its start, which reads as a stutter.
        // After that the normal cycle runs: jump to start (transitions OFF, so the start state paints
        // and becomes the transition baseline), hold START_PAUSE, play to the end, hold END_PAUSE.
        let phase = 'enter', applied = false;
        const reset = () => { svg.classList.add('st-no-anim'); rep.reset(); applied = false; };
        if (animBar) animBar.classList.add('is-on');
        loopT0 = performance.now();
        const firstHoldUntil = Math.max(loopT0, enterDoneAt) + END_PAUSE;   // arrival finishes, then a full pause
        const step = (now) => {
            if (gen !== loopGen) return;
            if (phase === 'enter') {
                // fill the bar as the arrival animation completes, hold it full through the pause
                const denom = Math.max(1, enterDoneAt - loopT0);
                if (animBarFill) animBarFill.style.width = (Math.max(0, Math.min(1, (now - loopT0) / denom)) * 100) + '%';
                if (now >= firstHoldUntil) { phase = 'loop'; loopT0 = now; reset(); }
            } else {
                const t = now - loopT0;
                if (!applied && t >= START_PAUSE) {        // start hold over: enable transitions + play to end
                    svg.classList.remove('st-no-anim'); rep.apply(); applied = true;
                }
                if (animBarFill) animBarFill.style.width = (Math.max(0, Math.min(1, (t - START_PAUSE) / ANIM_MS)) * 100) + '%';
                if (t >= LOOP_MS) { loopT0 = now; reset(); }
            }
            loopRAF = requestAnimationFrame(step);
        };
        loopRAF = requestAnimationFrame(step);
    }

    const videoEl = document.getElementById('story-video');

    // ----- narrated mini-player: while the narration plays, float it into a corner as the overview
    // scrolls out of view (the video element is *moved*, not recreated, so playback never stops). -----
    let pipBox = null, pipW = 360;                                  // remembered mini-player width
    function ensurePipBox() {
        if (pipBox) return pipBox;
        pipBox = document.createElement('div');
        pipBox.className = 'story__pip';
        pipBox.innerHTML = '<span class="story__pip-grip" aria-hidden="true" title="Drag to resize"></span>' +
            '<span class="story__pip-label">Narrated overview</span>' +
            '<button type="button" class="story__pip-close" aria-label="Close mini player" title="Close">&times;</button>';
        pipBox.querySelector('.story__pip-close').addEventListener('click', closePip);
        // top-left grip resizes the (bottom-right-anchored) box
        const grip = pipBox.querySelector('.story__pip-grip');
        let sx = 0, sw = 0, dragging = false;
        const mv = (e) => { if (!dragging) return; const x = (e.touches ? e.touches[0].clientX : e.clientX); pipW = Math.max(220, Math.min(620, sw + (sx - x))); pipBox.style.width = pipW + 'px'; e.preventDefault(); };
        const up = () => { dragging = false; document.removeEventListener('pointermove', mv); document.removeEventListener('pointerup', up); };
        grip.addEventListener('pointerdown', (e) => { dragging = true; sx = e.clientX; sw = pipBox.getBoundingClientRect().width; document.addEventListener('pointermove', mv); document.addEventListener('pointerup', up); e.preventDefault(); });
        document.body.appendChild(pipBox);
        return pipBox;
    }
    // FLIP: animate `el` from a previously-measured rect `first` to its current position
    function flip(el, first) {
        const last = el.getBoundingClientRect();
        if (!last.width) return;
        const dx = first.left - last.left, dy = first.top - last.top;
        const sx = first.width / last.width, sy = first.height / last.height;
        el.style.transition = 'none';
        el.style.transformOrigin = 'top left';
        el.style.transform = `translate(${dx}px, ${dy}px) scale(${sx.toFixed(4)}, ${sy.toFixed(4)})`;
        void el.offsetWidth;                                        // commit the inverted start
        el.style.transition = 'transform .42s cubic-bezier(.4,0,.2,1)';
        el.style.transform = 'none';
        const done = () => { el.style.transition = ''; el.style.transformOrigin = ''; el.style.transform = ''; el.removeEventListener('transitionend', done); };
        el.addEventListener('transitionend', done);
    }
    function enterPip() {
        if (!videoEl || document.body.classList.contains('story-pip-on')) return;
        const first = videoEl.getBoundingClientRect();             // the video's spot in the stage
        ensurePipBox();
        pipBox.style.width = pipW + 'px';
        pipBox.insertBefore(videoEl, pipBox.firstChild);
        document.body.classList.add('story-pip-on');
        flip(pipBox, first);                                       // fly the corner box out from the stage spot
    }
    function exitPip() {
        if (!videoEl || !document.body.classList.contains('story-pip-on')) return;
        const first = pipBox ? pipBox.getBoundingClientRect() : null;   // the corner box
        const stage = story.querySelector('.story__stage'), diagram = document.getElementById('story-diagram');
        if (stage) stage.insertBefore(videoEl, diagram || null);   // put it back before the diagram
        document.body.classList.remove('story-pip-on');
        if (first) flip(videoEl, first);                           // fly the stage video back from the corner
    }
    function closePip() { try { videoEl.pause(); } catch (e) {} exitPip(); }
    story._exitPip = exitPip;                                       // let the mode toggle dock it on exit

    function onScroll() {
        const rect = story.getBoundingClientRect();
        // narrated mode: no scrubbing. If the narration is playing, float it into a corner mini-player
        // as the overview scrolls out of view; dock it back when the overview returns.
        if (story.classList.contains('story--narrated')) {
            const outOfView = rect.bottom < 40 || rect.top > window.innerHeight - 40;
            if (outOfView) { if (videoEl && !videoEl.paused) enterPip(); }
            else exitPip();
            return;
        }
        const range = story.offsetHeight - window.innerHeight;
        let p = range > 0 ? (-rect.top) / range : 0;
        p = Math.max(0, Math.min(1, p));
        if (story.dataset.forceP) p = parseFloat(story.dataset.forceP);
        const t = p * duration;
        // active beat / caption / chapter = last whose threshold <= current time
        let bi = 0; for (let i = 0; i < BEATS.length; i++) if (BEATS[i].t <= t + 0.01) bi = i;
        const scrolled = Math.abs(p - lastScrollP) > 1e-4;   // ignore stray onScroll (resize, etc.) that don't move
        lastScrollP = p;
        if (scrolled) stopLoop();           // an actual scroll cancels any running animation loop
        applyBeat(bi);
        let ci = -1; for (let i = 0; i < caps.length; i++) if (parseFloat(caps[i].dataset.t) <= t + 0.01) ci = i;
        caps.forEach((c, i) => c.classList.toggle('is-on', i === ci));
        let hi = -1; for (let i = 0; i < chips.length; i++) if (parseFloat(chips[i].dataset.t) <= t + 0.01) hi = i;
        chips.forEach((c, i) => c.classList.toggle('is-on', i === hi));
        setCite(hi);
        if (bar) bar.style.width = (p * 100) + '%';
        story.classList.toggle('is-scrolling', p > 0.008);
        if (scrolled) lastScrollAt = performance.now();   // the idle detector below restarts the loop
    }
    // Robustly run the current beat's animation loop whenever scrolling has been idle a moment.
    setInterval(() => {
        if (loopRAF || story.classList.contains('story--narrated')) return;        // already looping / narrated
        if (story.dataset.forceP && !story.dataset.testLoop) return;                // capture mode
        if (performance.now() - lastScrollAt < 240) return;                         // still settling
        if (!story.dataset.forceP) {                                                // only when the stage fills the view
            const r = story.getBoundingClientRect();
            if (r.bottom < window.innerHeight * 0.5 || r.top > window.innerHeight * 0.5) return;
        }
        startLoop(curBeat);
    }, 200);

    // dev: ?loopdebug shows a live HUD of the loop state so we can see why it does/doesn't fire.
    if (new URLSearchParams(location.search).has('loopdebug')) {
        const hud = document.createElement('div');
        hud.style.cssText = 'position:fixed;top:8px;left:8px;z-index:2147483647;background:rgba(0,0,0,.88);' +
            'color:#9effa0;font:12px/1.5 monospace;padding:8px 10px;border-radius:6px;white-space:pre;pointer-events:none;';
        document.body.appendChild(hud);
        setInterval(() => {
            const r = story.getBoundingClientRect();
            const gate = !(r.bottom < window.innerHeight * 0.5 || r.top > window.innerHeight * 0.5);
            const rep = curBeat > 0 ? getReplay(curBeat) : null;
            hud.textContent =
                'frame      ' + (curBeat + 1) + '  (beat ' + curBeat + ')\n' +
                'loops?     ' + (rep ? 'YES' : 'no (block intro / frame<=1)') + '\n' +
                'looping    ' + (loopRAF ? 'RUNNING' : 'stopped') + '\n' +
                'idle ms    ' + Math.round(performance.now() - lastScrollAt) + '  (need >240)\n' +
                'in view    ' + (gate ? 'yes' : 'NO  rect.top=' + Math.round(r.top)) + '\n' +
                'forceP     ' + (story.dataset.forceP || '-') + '   narrated ' + story.classList.contains('story--narrated');
        }, 120);
    }

    // dev verification hook: ?storyp=0.42 forces a fixed progress and pins the
    // sticky stage into view so a headless screenshot captures that beat.
    const sp = new URLSearchParams(location.search).get('storyp');
    if (sp !== null) {
        story.dataset.forceP = sp;
        story.classList.add('story--forced');   // pins the stage to the viewport for capture
    }
    // dev: ?testloop=N pins the stage on beat N and runs its animation loop (for verifying the loop)
    const tl = new URLSearchParams(location.search).get('testloop');
    if (tl !== null) {
        const n = parseInt(tl, 10);
        story.dataset.forceP = ((BEATS[n].t + 0.5) / duration).toString();   // pin onScroll to beat N
        story.dataset.testLoop = '1';
        story.classList.add('story--forced');
        // NOTE: no direct startLoop() — the idle interval must trigger it (verifies the real path)
    }
    // dev: ?autoloop=N scrolls to beat N (no forcing) so the real idle-interval loop must trigger it
    const al = new URLSearchParams(location.search).get('autoloop');
    if (al !== null) {
        const n = parseInt(al, 10);
        setTimeout(() => {
            const range = story.offsetHeight - window.innerHeight;
            window.scrollTo(0, story.offsetTop + ((BEATS[n].t + 0.5) / duration) * range);
        }, 200);
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);

    // ----- arrow-key navigation between beats -----
    // While the sticky stage is pinned, Up/Down (and Left/Right, PageUp/Down) snap to the
    // adjacent beat. Stepping past either end falls through to normal scrolling so the keys
    // still carry you out of the story.
    const beatScrollTarget = (i) => {
        const range = story.offsetHeight - window.innerHeight;
        const tNext = i < BEATS.length - 1 ? BEATS[i + 1].t : duration;
        const tMid = (BEATS[i].t + tNext) / 2;            // park mid-beat so it reads as active
        return story.offsetTop + Math.min(1, tMid / duration) * range;
    };
    const currentBeat = () => {
        const range = story.offsetHeight - window.innerHeight;
        let p = range > 0 ? (-story.getBoundingClientRect().top) / range : 0;
        const t = Math.max(0, Math.min(1, p)) * duration;
        let bi = 0; for (let i = 0; i < BEATS.length; i++) if (BEATS[i].t <= t + 0.01) bi = i;
        return bi;
    };
    const storyPinned = () => {
        const r = story.getBoundingClientRect();
        return r.top <= 1 && r.bottom >= window.innerHeight - 1;
    };
    window.addEventListener('keydown', (e) => {
        if (story.dataset.forceP || e.metaKey || e.ctrlKey || e.altKey) return;
        if (story.classList.contains('story--narrated')) return;
        const next = e.key === 'ArrowDown' || e.key === 'ArrowRight' || e.key === 'PageDown';
        const prev = e.key === 'ArrowUp' || e.key === 'ArrowLeft' || e.key === 'PageUp';
        if ((!next && !prev) || !storyPinned()) return;
        const tgt = currentBeat() + (next ? 1 : -1);
        if (tgt < 0 || tgt >= BEATS.length) return;        // let the page scroll out of the story
        e.preventDefault();
        window.scrollTo({ top: beatScrollTarget(tgt), behavior: 'smooth' });
    });

    // ----- click the chapter rail to jump to a method; Skip jumps to the end -----
    const scrollToT = (t) => {
        const range = story.offsetHeight - window.innerHeight;
        window.scrollTo({ top: story.offsetTop + Math.min(1, (t + 0.4) / duration) * range, behavior: 'smooth' });
    };
    chips.forEach(chip => chip.addEventListener('click', () => scrollToT(parseFloat(chip.dataset.t) || 0)));
    const skip = document.getElementById('story-skip');
    if (skip) skip.addEventListener('click', () => {
        const range = story.offsetHeight - window.innerHeight;
        window.scrollTo({ top: story.offsetTop + range, behavior: 'smooth' });
    });

    // ----- mode toggle: scroll-driven diagram vs. narrated overview video -----
    const modeWrap = document.getElementById('story-mode');
    const setMode = (mode) => {
        const narrated = mode === 'narrated';
        if (modeWrap) modeWrap.querySelectorAll('.story__mode-btn').forEach(btn => {
            const on = btn.dataset.mode === mode;
            btn.classList.toggle('is-active', on);
            btn.setAttribute('aria-selected', on ? 'true' : 'false');
        });
        if (narrated) {
            window.scrollTo(0, story.offsetTop);          // park at the section top before it shrinks
            story.classList.add('story--narrated');
            if (videoEl) { try { videoEl.muted = false; videoEl.currentTime = 0; const pr = videoEl.play(); if (pr && pr.catch) pr.catch(() => {}); } catch (e) {} }
        } else {
            story.classList.remove('story--narrated');
            if (story._exitPip) story._exitPip();          // dock the mini-player back before pausing
            if (videoEl) { try { videoEl.pause(); } catch (e) {} }
        }
        onScroll();
    };
    if (modeWrap) modeWrap.querySelectorAll('.story__mode-btn').forEach(btn =>
        btn.addEventListener('click', () => setMode(btn.dataset.mode)));
    // deep-link straight to the narrated overview: ?storymode=narrated
    if (new URLSearchParams(location.search).get('storymode') === 'narrated') setMode('narrated');

    onScroll();
}

// Initialize sliders, dots, checkmarks, and video speed badges
// Method section — the appendix architecture figures (4a/4b/5) as collapsible "file tabs":
// each tab slides its figure + caption open; clicking the open tab (or its tab again) closes it.
function initMethodFigTabs() {
    const root = document.getElementById('method-figtabs');
    if (!root || root.dataset.init === '1') return;
    root.dataset.init = '1';
    const stage = document.getElementById('method-figstage');
    const tabs = Array.prototype.slice.call(root.querySelectorAll('.figtab'));
    const panels = Array.prototype.slice.call(root.querySelectorAll('.figpanel'));
    if (!stage || !tabs.length) return;
    // typeset the MPPI algorithm's inline \( ... \) math (once KaTeX is ready)
    (function renderAlgoMath() {
        if (!window.katex) return void setTimeout(renderAlgoMath, 200);
        const algo = root.querySelector('#figpanel-mppi');
        if (algo && !algo.dataset.tex) {
            algo.dataset.tex = '1';
            algo.innerHTML = algo.innerHTML.replace(/\\\((.+?)\\\)/g, (_, m) => {
                try { return katex.renderToString(m, { throwOnError: false }); } catch (e) { return m; }
            });
        }
    })();
    // each file tab focuses the component(s) it documents in the diagram/algorithm:
    // planner (step 1), encoder+dynamics (step 4), reward (step 5), value+policy (λ-return, steps 6-7)
    const TAB_FOCUS = {
        mppi: [['planner'], 'planner'],
        '4a': [['encoder'], 'encoder'],          // expands to the Encoder & Dynamics group
        '4b': [['reward'], 'reward'],
        '5':  [['value', 'policy'], 'value']      // λ-return drives the actor-critic update
    };
    let openFig = null;
    function setOpen(fig) {
        const prev = openFig;
        openFig = fig;
        tabs.forEach(t => { const on = t.dataset.fig === fig; t.classList.toggle('is-open', on); t.setAttribute('aria-expanded', on ? 'true' : 'false'); });
        panels.forEach(p => p.classList.toggle('is-active', p.dataset.fig === fig));
        stage.classList.toggle('is-open', !!fig);
        // focus the open tab's component(s) in the figure/algorithm; clear when a focusing tab closes
        if (typeof window.methodSetFocus === 'function') {
            const f = fig && TAB_FOCUS[fig];
            if (f) window.methodSetFocus(f[0], f[1]);
            else if (prev && TAB_FOCUS[prev]) window.methodSetFocus(null);
        }
    }
    tabs.forEach(t => t.addEventListener('click', () => setOpen(openFig === t.dataset.fig ? null : t.dataset.fig)));
}

function initializeInteractiveSections() {
    try { initBaselineDiagram(); } catch (e) { console.error('initBaselineDiagram failed:', e); }
    try { initStoryScrub(); } catch (e) { console.error('initStoryScrub failed:', e); }
    try { initResultsTunnel(); } catch (e) { console.error('initResultsTunnel failed:', e); }
    try { initMethodExplorer(); } catch (e) { console.error('initMethodExplorer failed:', e); }
    try { initMethodFigTabs(); } catch (e) { console.error('initMethodFigTabs failed:', e); }
    const dotsContainer = document.getElementById('slider-dots');
    const iterationSlider = document.getElementById('iteration-slider');
    const sliderHint = document.querySelector('#time-lapses-block .slider-hint');
    
    // Initialize for efficiency section (only if elements are present)
    if (dotsContainer && iterationSlider) {
        // Start at the first success checkmark (iteration 60)
        const initialIteration = 60;
        iterationSlider.max = 100;
        iterationSlider.value = initialIteration;
        
        // Create dots for efficiency slider
        if (!dotsContainer.hasChildNodes()) {
            for (let i = 0; i <= 100; i += 10) {
                const dot = document.createElement('div');
                dot.className = 'slider-dot' + (i <= initialIteration ? ' active' : '');
                dot.dataset.value = i;
                dotsContainer.appendChild(dot);
            }
        }
        const checkmarksContainer = document.getElementById('slider-checkmarks');
        if (checkmarksContainer && !checkmarksContainer.hasChildNodes()) {
            const pushCheckValues = [60, 70, 80, 90, 100];
            for (let i = 0; i <= 100; i += 10) {
                const span = document.createElement('span');
                span.textContent = pushCheckValues.includes(i) ? '✓' : '';
                checkmarksContainer.appendChild(span);
            }
        }
        // Ensure all dependent UI (labels, videos, time) match initial position
        updateIteration(initialIteration);

        // Show a one-time hint above the slider until the user interacts
        if (sliderHint) {
            sliderHint.classList.add('visible');
            const dismissHint = () => {
                sliderHint.classList.remove('visible');
            };
            iterationSlider.addEventListener('input', dismissHint, { once: true });
            iterationSlider.addEventListener('mousedown', dismissHint, { once: true });
            iterationSlider.addEventListener('touchstart', dismissHint, { once: true });
        }
    }

    // Initialize for transfer section
    const transferDotsContainer = document.getElementById('transfer-slider-dots');
    const transferSlider = document.getElementById('transfer-iteration-slider');
    const transferSliderHint = document.querySelector('#transfer-time-lapses-block .slider-hint');
    
    if (transferDotsContainer && transferSlider) {
        // Start at the first success checkmark (iteration 30)
        const initialTransferIteration = 30;
        transferSlider.max = 100;
        transferSlider.value = initialTransferIteration;
        
        // Create dots for transfer slider
        if (!transferDotsContainer.hasChildNodes()) {
            for (let i = 0; i <= 100; i += 10) {
                const dot = document.createElement('div');
                dot.className = 'slider-dot' + (i <= initialTransferIteration ? ' active' : '');
                dot.dataset.value = i;
                transferDotsContainer.appendChild(dot);
            }
        }
        const transferCheckmarksContainer = document.getElementById('transfer-slider-checkmarks');
        if (transferCheckmarksContainer && !transferCheckmarksContainer.hasChildNodes()) {
            const transferPushCheckValues = [30, 40, 50, 60, 80, 100];
            for (let i = 0; i <= 100; i += 10) {
                const span = document.createElement('span');
                span.textContent = transferPushCheckValues.includes(i) ? '✓' : '';
                transferCheckmarksContainer.appendChild(span);
            }
        }
        // Ensure all dependent UI (labels, videos, time) match initial position
        updateTransferIteration(initialTransferIteration);

        // Show a one-time hint above the transfer slider until the user interacts
        if (transferSliderHint) {
            transferSliderHint.classList.add('visible');
            const dismissTransferHint = () => {
                transferSliderHint.classList.remove('visible');
            };
            transferSlider.addEventListener('input', dismissTransferHint, { once: true });
            transferSlider.addEventListener('mousedown', dismissTransferHint, { once: true });
            transferSlider.addEventListener('touchstart', dismissTransferHint, { once: true });
        }
    }

    // Ensure comparison / transfer videos have speed badges and playback rate set
    document.querySelectorAll('.video-display').forEach(display => {
        const isDemo = display.closest('.demo-video-wrapper');
        const isTeaser = display.classList.contains('teaser-video');
        const isMethodCard = display.closest('.method-mppi-video') || display.closest('.method-card-media') || display.closest('.method-video');
        if (isMethodCard) return;
        if (!isDemo && !isTeaser && !display.querySelector('.video-speed-badge')) {
            const badge = document.createElement('span');
            badge.className = 'video-speed-badge';
            badge.textContent = '3X';
            display.appendChild(badge);
        }
        display.querySelectorAll('video').forEach(v => {
            // Leave the method MPAIL2 video and the summary teaser at normal speed
            if (v.dataset.allowAudio !== undefined || isTeaser) {
                v.playbackRate = 1;
                return;
            }
            v.playbackRate = 3;
            v.addEventListener('loadeddata', function() { this.playbackRate = 3; });
        });
    });
    ['demonstrations-grid', 'initial-demonstrations-grid', 'transfer-demonstrations-grid'].forEach(gridId => {
        const grid = document.getElementById(gridId);
        if (grid && !grid.querySelector('.video-speed-badge')) {
            const badge = document.createElement('span');
            badge.className = 'video-speed-badge';
            badge.textContent = '3X';
            grid.appendChild(badge);
        }
    });

    // Keep sample method and transfer demo in a consistent initial state
    updateSampleMethodVideo();
    switchTransferDemo('initial');
    initializeMethodVideos();
}

function initializeMethodVideos() {
    const methodVideoIds = ['method-mppi-video', 'method-card-video-right'];
    methodVideoIds.forEach(id => {
        const v = document.getElementById(id);
        if (!v || v.dataset.initialized === 'true') return;
        v.dataset.initialized = 'true';
        v.playbackRate = 1;
        v.muted = true;
        v.defaultMuted = true;

        const tryPlay = () => { if (v.paused) v.play().catch(() => {}); };
        if (v.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
            tryPlay();
        } else {
            v.addEventListener('loadeddata', tryPlay, { once: true });
            v.addEventListener('canplay', tryPlay, { once: true });
        }
    });

    const methodSection = document.getElementById('method-section');
    if (methodSection && 'IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    methodVideoIds.forEach(id => {
                        const v = document.getElementById(id);
                        if (v && v.paused) v.play().catch(() => {});
                    });
                }
            });
        }, { threshold: 0.25 });
        observer.observe(methodSection);
    }
}

function updateIteration(value) {
    const iteration = parseInt(value);
    document.getElementById('iteration-display').textContent = iteration;
    
    // Update dots
    const dots = document.querySelectorAll('#slider-dots .slider-dot');
    dots.forEach(dot => {
        const dotValue = parseInt(dot.dataset.value);
        if (dotValue <= iteration) {
            dot.classList.add('active');
        } else {
            dot.classList.remove('active');
        }
    });
    
    // Update time spent
    const timeSpent = document.getElementById('time-spent');
    if (iteration === 0) {
        timeSpent.textContent = '0';
    } else {
        const rate = currentTask === 'pick-place' ? (68 / 150) : (currentTask === 'mug-on-plate' ? (100 / 200) : (49 / 100));
        const minutes = Math.round(iteration * rate);
        timeSpent.textContent = `~${minutes} min`;
    }
    
    updateVideo();
    updateSampleMethodVideo();
}

// Task selection functionality
var currentTask = 'push';
window.currentTask = currentTask;

function selectTask(task) {
    currentTask = task;
    window.currentTask = task;

    // If there's an experiments sidebar hint showing, hide it when user changes tasks
    if (typeof window.dismissLeftSidebarHint === 'function') {
        window.dismissLeftSidebarHint();
    }
    
    // Update task buttons
    document.querySelectorAll('.task-btn').forEach(btn => {
        if (btn.dataset.task === task) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    // Update task button wrappers
    document.querySelectorAll('.task-btn-wrapper').forEach(wrapper => {
        if (wrapper.dataset.task === task) {
            wrapper.classList.add('active');
        } else {
            wrapper.classList.remove('active');
        }
    });
    
    if (task === 'pick-place') {
        document.getElementById('efficiency-section').classList.add('task-pick-place');
        document.getElementById('transfer-section').classList.add('task-pick-place');
    } else {
        document.getElementById('efficiency-section').classList.remove('task-pick-place');
        document.getElementById('transfer-section').classList.remove('task-pick-place');
    }

    // Hide transfer section entirely for Mug on Plate (no transfer results)
    const transferSection = document.getElementById('transfer-section');
    if (transferSection) {
        transferSection.style.display = task === 'mug-on-plate' ? 'none' : '';
    }

    updateSliderRange();
    updateVideo();
    updateSampleMethodVideo();
    updateDemonstrationVideos();
    updateResultsImages();
    if (task !== 'mug-on-plate') {
        updateTransferVideo();
        updateTransferDemonstrationVideos();
        updateInitialDemonstrationVideos();
    }
    if (typeof refreshCharts === 'function') refreshCharts();
}

function updateSliderRange() {
    const maxIter = currentTask === 'push' ? 100 : (currentTask === 'pick-place' ? 150 : 200);

    const slider = document.getElementById('iteration-slider');
    if (slider) {
        slider.max = maxIter;
        if (parseInt(slider.value) > maxIter) slider.value = maxIter;
        rebuildDots('slider-dots', maxIter, parseInt(slider.value));
        updateIteration(slider.value);
    }

    if (currentTask !== 'mug-on-plate') {
        const transferSlider = document.getElementById('transfer-iteration-slider');
        if (transferSlider) {
            transferSlider.max = maxIter;
            if (parseInt(transferSlider.value) > maxIter) transferSlider.value = maxIter;
            rebuildDots('transfer-slider-dots', maxIter, parseInt(transferSlider.value));
            updateTransferIteration(transferSlider.value);
        }
    }
}

function rebuildDots(containerId, maxIter, currentValue) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    for (let i = 0; i <= maxIter; i += 10) {
        const dot = document.createElement('div');
        dot.className = 'slider-dot' + (i <= currentValue ? ' active' : '');
        dot.dataset.value = i;
        container.appendChild(dot);
    }
    if (containerId === 'slider-dots') {
        const checkmarksContainer = document.getElementById('slider-checkmarks');
        if (checkmarksContainer) {
            const pushCheckValues = [60, 70, 80, 90, 100];
            const pickPlaceCheckValues = [60, 70, 100, 110, 120, 130, 140];
            const mopCheckValues = [100, 120, 140, 160, 180, 200];
            const checkValues = currentTask === 'push' ? pushCheckValues : (currentTask === 'pick-place' ? pickPlaceCheckValues : mopCheckValues);
            checkmarksContainer.innerHTML = '';
            for (let i = 0; i <= maxIter; i += 10) {
                const span = document.createElement('span');
                span.textContent = checkValues.includes(i) ? '✓' : '';
                checkmarksContainer.appendChild(span);
            }
        }
    }
    if (containerId === 'transfer-slider-dots') {
        const transferCheckmarksContainer = document.getElementById('transfer-slider-checkmarks');
        if (transferCheckmarksContainer) {
            const transferPushCheckValues = [30, 40, 50, 60, 80, 100];
            const transferPickPlaceCheckValues = [30, 40, 70, 80, 90, 100, 110, 130, 140, 150];
            const transferCheckValues = currentTask === 'push' ? transferPushCheckValues : transferPickPlaceCheckValues;
            transferCheckmarksContainer.innerHTML = '';
            for (let i = 0; i <= maxIter; i += 10) {
                const span = document.createElement('span');
                span.textContent = transferCheckValues.includes(i) ? '✓' : '';
                transferCheckmarksContainer.appendChild(span);
            }
        }
    }
}

function updateResultsImages() {
    if (currentTask === 'mug-on-plate') return;
    const folder = currentTask === 'push' ? 'Push' : 'Pick';
    const tableImg = document.getElementById('results-table-img');
    if (tableImg) tableImg.src = `Media/Image/Results/${folder}_table.png`;
    const tTableImg = document.getElementById('transfer-results-table-img');
    if (tTableImg) tTableImg.src = `Media/Image/Results/Transfer_${folder}_table.png`;
}

function updateDemonstrationVideos() {
    const taskFolder = currentTask === 'push' ? 'Push' : (currentTask === 'pick-place' ? 'Pick' : 'MoP');
    const demoVideos = document.querySelectorAll('.demo-video');
    
    demoVideos.forEach((video, index) => {
        const demoNum = index + 1;
        const videoPath = `Media/Video/Demonstrations/${taskFolder}/demo_${demoNum}.mp4`;
        video.src = videoPath;
        video.load();
    });
}

function updateTransferDemonstrationVideos() {
    const taskFolder = currentTask === 'push' ? 'Push' : 'Pick';
    const demoVideos = document.querySelectorAll('.transfer-demo-video');
    
    demoVideos.forEach((video, index) => {
        const demoNum = index + 1;
        const videoPath = `Media/Video/Demonstrations/Transfer/${taskFolder}/demo_${demoNum}.mp4`;
        video.src = videoPath;
        video.load();
    });
}

function updateInitialDemonstrationVideos() {
    const taskFolder = currentTask === 'push' ? 'Push' : 'Pick';
    const demoVideos = document.querySelectorAll('.initial-demo-video');
    
    demoVideos.forEach((video, index) => {
        const demoNum = index + 1;
        const videoPath = `Media/Video/Demonstrations/${taskFolder}/demo_${demoNum}.mp4`;
        video.src = videoPath;
        video.load();
    });
}

function switchTransferDemo(mode) {
    const initialPanel = document.getElementById('transfer-demo-panel-initial');
    const newPanel = document.getElementById('transfer-demo-panel-new');
    const buttons = document.querySelectorAll('.transfer-demo-btn');
    if (!initialPanel || !newPanel) return;

    if (mode === 'initial') {
        initialPanel.classList.remove('hidden');
        newPanel.classList.add('hidden');
    } else {
        initialPanel.classList.add('hidden');
        newPanel.classList.remove('hidden');
    }

    buttons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.transferDemo === mode);
    });
}

let currentSampleMethod = 'mpail2-p';
const sampleMethodFolderMap = {
    'mpail2-p': 'mpail2_p',
    'mpail2-pm': 'mpail2_pm',
    'rlpd': 'rlpd'
};
const sampleMethodLabelMap = {
    'mpail2-p': 'Without Planning',
    'mpail2-pm': 'Model-Free',
    'rlpd': 'RLPD'
};
function selectSampleMethod(method) {
    currentSampleMethod = method;
    document.querySelectorAll('.sample-method-btn').forEach(btn => {
        if (btn.dataset.method === method) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    const label = document.getElementById('sample-method-label');
    if (label) label.textContent = sampleMethodLabelMap[method] || method;
    updateSampleMethodVideo();
}
function updateSampleMethodVideo() {
    const iterationSliderEl = document.getElementById('iteration-slider');
    if (!iterationSliderEl) {
        // Sections (and slider) not yet loaded; nothing to update
        return;
    }
    let iteration = parseInt(iterationSliderEl.value);
    const folder = sampleMethodFolderMap[currentSampleMethod] || 'mpail2_p';
    const taskFolder = currentTask === 'push' ? 'Push' : (currentTask === 'pick-place' ? 'Pick' : 'MoP');
    // MoP baseline videos only go to iter_140; clamp so we don't request a missing file
    if (currentTask === 'mug-on-plate') iteration = Math.min(iteration, 140);
    const videoPath = `Media/Video/Comparison/${folder}/${taskFolder}/iter_${iteration}.mp4`;
    const video = document.getElementById('sample-method-video');
    if (!video) return;
    video.classList.add('loading');
    video.src = videoPath;
    video.load();
    video.oncanplay = function() {
        this.playbackRate = 3;
        this.classList.remove('loading');
        this.play();
        this.oncanplay = null;
    };
}

function updateVideo() {
    const iterationSliderEl = document.getElementById('iteration-slider');
    const videoLeft = document.getElementById('task-video');
    if (!iterationSliderEl || !videoLeft) {
        return;
    }
    const iteration = parseInt(iterationSliderEl.value);

    const taskFolder = currentTask === 'push' ? 'Push' : (currentTask === 'pick-place' ? 'Pick' : 'MoP');
    const videoPath = `Media/Video/${taskFolder}/iter_${iteration}.mp4`;

    videoLeft.classList.add('loading');

    setTimeout(() => {
        videoLeft.src = videoPath;
        videoLeft.load();

        videoLeft.oncanplay = function() {
            this.playbackRate = 3;
            videoLeft.classList.remove('loading');
            videoLeft.play();
            videoLeft.oncanplay = null;
        };
    }, 150);
}

// Transfer section functionality
function updateTransferIteration(value) {
    const iteration = parseInt(value);
    document.getElementById('transfer-iteration-display').textContent = iteration;
    
    // Update dots
    const dots = document.querySelectorAll('#transfer-slider-dots .slider-dot');
    dots.forEach(dot => {
        const dotValue = parseInt(dot.dataset.value);
        if (dotValue <= iteration) {
            dot.classList.add('active');
        } else {
            dot.classList.remove('active');
        }
    });
    
    // Update time spent
    const timeSpent = document.getElementById('transfer-time-spent');
    if (iteration === 0) {
        timeSpent.textContent = '0';
    } else {
        const rate = currentTask === 'pick-place' ? (72 / 150) : (47 / 100);
        const minutes = Math.round(iteration * rate);
        timeSpent.textContent = `~${minutes} min`;
    }
    
    updateTransferVideo();
}

function updateTransferVideo() {
    const transferSliderEl = document.getElementById('transfer-iteration-slider');
    const videoLeft = document.getElementById('transfer-video-left');
    const videoRight = document.getElementById('transfer-video-right');
    if (!transferSliderEl || !videoLeft || !videoRight) {
        return;
    }
    const iteration = parseInt(transferSliderEl.value);
    
    // Determine task folder
    const taskFolder = currentTask === 'push' ? 'Push' : 'Pick';
    
    // Build video paths
    const videoPathLeft = `Media/Video/Transfer/${taskFolder}/iter_${iteration}.mp4`;
    const videoPathRight = `Media/Video/From%20Scratch/${taskFolder}/iter_${iteration}.mp4`;

    // Show loading states
    videoLeft.classList.add('loading');
    videoRight.classList.add('loading');
    
    setTimeout(() => {
        videoLeft.src = videoPathLeft;
        videoRight.src = videoPathRight;
        videoLeft.playbackRate = 3;
        videoRight.playbackRate = 3;
        videoLeft.load();
        videoRight.load();
        
        videoLeft.oncanplay = function() {
            this.playbackRate = 3;
            videoLeft.classList.remove('loading');
            videoLeft.play();
            videoLeft.oncanplay = null;
        };
        videoRight.oncanplay = function() {
            this.playbackRate = 3;
            videoRight.classList.remove('loading');
            videoRight.play();
            videoRight.oncanplay = null;
        };
    }, 150);
}

// Timeline navigation click handlers
document.querySelectorAll('.timeline-item').forEach(item => {
    item.addEventListener('click', function(e) {
        e.preventDefault();
        const sectionName = this.dataset.section;
        
        // Set navigation flag
        window.isTimelineNavigating = true;
        
        // Get sections
        const summarySection = document.getElementById('summary-section');
        const efficiencySection = document.getElementById('efficiency-section');
        const transferSection = document.getElementById('transfer-section');
        const generalizationSection = document.getElementById('generalization-section');
        const yourTurnSection = document.getElementById('your-turn-section');
        
        const methodSection = document.getElementById('method-section');
        const resultsOverviewSection = document.getElementById('results-overview-section');

        // Reset all classes
        summarySection.classList.remove('fade-out', 'instant-show');
        efficiencySection.classList.remove('fade-in', 'fade-out', 'hidden-below');
        transferSection.classList.remove('fade-in', 'fade-out', 'hidden-below');
        generalizationSection.classList.remove('fade-in', 'fade-out', 'hidden-below');
        if (yourTurnSection) {
            yourTurnSection.classList.remove('fade-in', 'fade-out', 'hidden-below');
        }
        
        // Update active timeline item
        document.querySelectorAll('.timeline-item').forEach(ti => {
            if (ti.dataset.section === sectionName) {
                ti.classList.add('active');
            } else {
                ti.classList.remove('active');
            }
        });
        
        // Determine scroll target and set section states
        let scrollTarget = 0;
        
        // Apply section states based on target
        if (sectionName === 'summary') {
            scrollTarget = window.innerHeight;
            summarySection.classList.add('instant-show');
            efficiencySection.classList.add('hidden-below');
            transferSection.classList.add('hidden-below');
            generalizationSection.classList.add('hidden-below');
            if (yourTurnSection) yourTurnSection.classList.add('hidden-below');
        } else if (sectionName === 'results-overview' && resultsOverviewSection) {
            scrollTarget = resultsOverviewSection.offsetTop;
        } else if (sectionName === 'side-by-side') {
            scrollTarget = efficiencySection.offsetTop;
            summarySection.classList.add('fade-out');
            efficiencySection.classList.add('fade-in');
            transferSection.classList.add('hidden-below');
            generalizationSection.classList.add('hidden-below');
            if (yourTurnSection) yourTurnSection.classList.add('hidden-below');
        } else if (sectionName === 'generalization') {
            scrollTarget = generalizationSection.offsetTop;
            summarySection.classList.add('fade-out');
            efficiencySection.classList.add('fade-out');
            transferSection.classList.add('fade-out');
            generalizationSection.classList.add('fade-in');
            if (yourTurnSection) yourTurnSection.classList.add('hidden-below');
        } else if (sectionName === 'your-turn' && yourTurnSection) {
            scrollTarget = yourTurnSection.offsetTop;
            summarySection.classList.add('fade-out');
            efficiencySection.classList.add('fade-out');
            transferSection.classList.add('fade-out');
            generalizationSection.classList.add('fade-out');
            yourTurnSection.classList.add('fade-in');
        } else if (sectionName === 'method' && methodSection) {
            scrollTarget = methodSection.offsetTop;
        }
        
        // Smooth scroll to target
        window.scrollTo({
            top: scrollTarget,
            behavior: 'smooth'
        });
        
        // Reset navigation flag after scroll
        setTimeout(() => {
            window.isTimelineNavigating = false;
            summarySection.classList.remove('instant-show');
        }, 800);
    });
});

function openVideoModal(section) {
    const modal = document.getElementById('video-modal');
    const video = document.getElementById('modal-video');
    
    // Determine video path based on section
    let videoPath;
    if (section === 'sample') {
        // Main sample videos
        if (currentTask === 'push') {
            videoPath = 'Media/Video/Full/MPAIL.mp4';
        } else {
            videoPath = 'Media/Video/Full/PickandPlace.mp4';
        }
    } else if (section === 'scratch') {
        // From scratch videos
        if (currentTask === 'push') {
            videoPath = 'Media/Video/Full/From_Scratch_Push.mp4';
        } else {
            videoPath = 'Media/Video/Full/From_Scratch_Pick.mp4';
        }
    } else if (section === 'transfer') {
        // Transfer videos
        if (currentTask === 'push') {
            videoPath = 'Media/Video/Full/Transfer_Push.mp4';
        } else {
            videoPath = 'Media/Video/Full/Transfer_PickandPlace.mp4';
        }
    } else if (section === 'comparison') {
        const methodFolder = sampleMethodFolderMap[currentSampleMethod] || 'mpail2_p';
        const taskFolder = currentTask === 'push' ? 'Push' : 'Pick';
        videoPath = `Media/Video/Comparison/Full/${methodFolder}_${taskFolder}.mp4`;
    }
    
    // Load video
    video.src = videoPath;
    video.load();
    video.playbackRate = 3;
    
    modal.classList.add('active');
    video.currentTime = 0;
    video.play();
    document.body.style.overflow = 'hidden';
}

function closeVideoModal() {
    const modal = document.getElementById('video-modal');
    const video = document.getElementById('modal-video');
    modal.classList.remove('active');
    video.pause();
    document.body.style.overflow = '';
}

// Copy the BibTeX entry to the clipboard, with brief button feedback
function copyBibtex(btn) {
    const code = document.getElementById('bibtex-entry');
    if (!code) return;
    const text = code.textContent;
    const done = () => { const o = btn.textContent; btn.textContent = 'Copied!'; btn.classList.add('is-copied'); setTimeout(() => { btn.textContent = o; btn.classList.remove('is-copied'); }, 1600); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(() => fallbackCopy(text, done));
    } else { fallbackCopy(text, done); }
}
function fallbackCopy(text, done) {
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); done(); } catch (e) { /* ignore */ }
    document.body.removeChild(ta);
}

// Modal event listeners
document.addEventListener('DOMContentLoaded', function() {
    const modal = document.getElementById('video-modal');
    const backdrop = modal.querySelector('.modal-backdrop');
    backdrop.addEventListener('click', closeVideoModal);
    
    // ESC key to close modal
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            closeVideoModal();
        }
    });
});

// Video synchronization and controls
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('video').forEach(function(video) {
        if (video.dataset.allowAudio !== undefined) return;
        video.muted = true;
        video.addEventListener('volumechange', function() {
            if (!this.muted) {
                this.muted = true;
            }
        });
    });

    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeName === 'VIDEO') {
                    if (node.dataset.allowAudio !== undefined) return;
                    node.muted = true;
                    node.addEventListener('volumechange', function() {
                        if (!this.muted) this.muted = true;
                    });
                }
                if (node.querySelectorAll) {
                    node.querySelectorAll('video').forEach(function(v) {
                        if (v.dataset.allowAudio !== undefined) return;
                        v.muted = true;
                        v.addEventListener('volumechange', function() {
                            if (!this.muted) this.muted = true;
                        });
                    });
                }
            });
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });

    // Copy-to-clipboard for BibTeX
    document.addEventListener('click', function(e) {
        const btn = e.target.closest && e.target.closest('.copy-bibtex-btn');
        if (!btn) return;

        const targetSelector = btn.getAttribute('data-copy-target');
        if (!targetSelector) return;
        const targetEl = document.querySelector(targetSelector);
        if (!targetEl) return;

        const text = targetEl.textContent.trim();
        if (!text) return;

        const statusEl = btn.parentElement.querySelector('.copy-bibtex-status');

        function showCopied() {
            if (statusEl) {
                statusEl.textContent = 'Copied!';
            }
            const originalTitle = btn.getAttribute('title') || '';
            btn.setAttribute('data-original-title', originalTitle);
            btn.setAttribute('title', 'Copied!');
            btn.disabled = true;
            setTimeout(() => {
                const savedTitle = btn.getAttribute('data-original-title') || originalTitle;
                if (savedTitle) {
                    btn.setAttribute('title', savedTitle);
                } else {
                    btn.removeAttribute('title');
                }
                btn.removeAttribute('data-original-title');
                btn.disabled = false;
                if (statusEl) {
                    statusEl.textContent = '';
                }
            }, 1500);
        }

        const fallbackCopy = () => {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            try {
                document.execCommand('copy');
                showCopied();
            } catch (err) {
                console.error('Failed to copy BibTeX:', err);
            } finally {
                document.body.removeChild(textarea);
            }
        };

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(showCopied).catch(fallbackCopy);
        } else {
            fallbackCopy();
        }
    });
});

// Theme toggle functionality
function toggleTheme() {
    document.body.classList.toggle('dark-mode');
    
    // Save theme preference
    const isDarkMode = document.body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDarkMode);
}

// Load theme preference on page load
(function() {
    const savedDarkMode = localStorage.getItem('darkMode');
    if (savedDarkMode === 'true') {
        document.body.classList.add('dark-mode');
    }
})();