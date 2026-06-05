// Load sections dynamically
async function loadSections() {
    const sectionMappings = {
        'summary-section-container': 'sections/summary.html',
        'method-section-container': 'sections/method.html',
        'results-overview-section-container': 'sections/experiments-overview.html',
        'efficiency-section-container': 'sections/efficiency.html',
        'transfer-section-container': 'sections/transfer.html',
        'generalization-section-container': 'sections/generalization.html',
        'your-turn-section-container': 'sections/your-turn.html'
    };

    const loadPromises = Object.entries(sectionMappings).map(async ([containerId, sectionFile]) => {
        try {
            const response = await fetch(sectionFile);
            if (response.ok) {
                const html = await response.text();
                const container = document.getElementById(containerId);
                if (container) {
                    container.innerHTML = html;
                }
            } else {
                console.error(`Failed to load section: ${sectionFile}`);
            }
        } catch (error) {
            console.error(`Error loading section ${sectionFile}:`, error);
        }
    });
    await Promise.all(loadPromises);
    // Initialize any interactive elements that depend on loaded sections
    initializeInteractiveSections();

    // Make results blocks collapsible (efficiency + transfer)
    document.querySelectorAll('.content-block .collapsible-body').forEach((bodyEl) => {
        const block = bodyEl.closest('.content-block');
        if (!block) return;
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
    // Trigger scroll event after all sections are loaded
    window.dispatchEvent(new Event('scroll'));
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

// Hero video and authors toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    var heroVideo = document.getElementById('hero-video');
    var heroContainer = document.getElementById('hero-video-container');
    if (heroVideo && heroContainer) {
        heroVideo.addEventListener('canplay', function() {
            this.classList.add('hero-ready');
            heroContainer.classList.add('hero-ready');
        });
        heroVideo.addEventListener('timeupdate', function() {
            if (this.duration && !isNaN(this.duration)) {
                var remaining = this.duration - this.currentTime;
                if (remaining <= 8 && remaining > 0) {
                    this.currentTime = 0;
                }
            }
        });
        heroVideo.addEventListener('ended', function() {
            this.currentTime = 0;
            this.play();
        });
    }
    
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

// Main functionality
document.addEventListener('DOMContentLoaded', function() {
    // One-time hint for task switching in Experiments
    const leftSidebarHint = document.createElement('div');
    leftSidebarHint.className = 'left-sidebar-hint';
    leftSidebarHint.innerHTML = `
        <button class="hint-close" aria-label="Dismiss hint" title="Dismiss">×</button>
        <strong>Tip:</strong> Use the left sidebar to switch between experiment tasks.
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

        const gap = 12;
        const left = sidebarRect.right + gap;
        const top = sidebarRect.top + (sidebarRect.height / 2) - (hintRect.height / 2);
        const clampedLeft = Math.max(8, Math.min(left, window.innerWidth - hintRect.width - 8));
        const clampedTop = Math.max(8, Math.min(top, window.innerHeight - hintRect.height - 8));

        leftSidebarHint.style.left = `${clampedLeft}px`;
        leftSidebarHint.style.top = `${clampedTop}px`;
        leftSidebarHint.style.transform = 'translateY(0)';

        // Put the arrow vertically aligned with sidebar center
        const arrowY = (sidebarRect.top + sidebarRect.height / 2) - clampedTop;
        leftSidebarHint.style.setProperty('--hint-arrow-y', `${Math.max(12, Math.min(hintRect.height - 12, arrowY))}px`);

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
        const heroSection = document.querySelector('.video-section');
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
        const efficiencySection = document.getElementById('efficiency-section');
        const transferSection = document.getElementById('transfer-section');
        const generalizationSection = document.getElementById('generalization-section');
        const yourTurnSection = document.getElementById('your-turn-section');

        // If core sections are not yet present, skip the rest of the scroll logic
        if (!summarySection || !methodSection || !resultsOverviewSection || !efficiencySection || !transferSection || !generalizationSection || !yourTurnSection) {
            return;
        }

        // Get section positions
        const summaryRect = summarySection.getBoundingClientRect();
        const methodRect = methodSection.getBoundingClientRect();
        const resultsOverviewRect = resultsOverviewSection.getBoundingClientRect();
        const efficiencyRect = efficiencySection.getBoundingClientRect();
        const transferRect = transferSection.getBoundingClientRect();
        const generalizationRect = generalizationSection.getBoundingClientRect();
        const yourTurnRect = yourTurnSection.getBoundingClientRect();
        
        // Show/hide left task buttons
        if (taskButtonsLeft) {
            const showLeftButtons = efficiencyRect.top < windowHeight * 0.75 && generalizationRect.top > windowHeight * 0.5;
            if (showLeftButtons) {
                taskButtonsLeft.classList.add('visible');
            } else {
                taskButtonsLeft.classList.remove('visible');
            }
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
        } else if (transferRect.top < windowHeight * 0.5) {
            activeSection = 'side-by-side';
        } else if (efficiencyRect.top < windowHeight * 0.5) {
            activeSection = 'side-by-side';
        } else if (resultsOverviewRect.top < windowHeight * 0.5) {
            activeSection = 'results-overview';
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
        const showTransfer = transferRect.top < windowHeight * 0.75;
        const showEfficiency = efficiencyRect.top < windowHeight * 0.75;

        // Reset section states
        summarySection.classList.remove('fade-out');
        efficiencySection.classList.remove('fade-in', 'fade-out', 'hidden-below');
        transferSection.classList.remove('fade-in', 'fade-out', 'hidden-below');
        generalizationSection.classList.remove('fade-in', 'fade-out', 'hidden-below');

        if (showGeneralization) {
            summarySection.classList.add('fade-out');
            efficiencySection.classList.add('fade-out');
            transferSection.classList.add('fade-out');
            generalizationSection.classList.add('fade-in');
        } else if (showTransfer) {
            summarySection.classList.add('fade-out');
            efficiencySection.classList.add('fade-out');
            transferSection.classList.add('fade-in');
            generalizationSection.classList.add('hidden-below');
        } else if (showEfficiency) {
            summarySection.classList.add('fade-out');
            efficiencySection.classList.add('fade-in');
            transferSection.classList.add('hidden-below');
            generalizationSection.classList.add('hidden-below');
        } else {
            efficiencySection.classList.add('hidden-below');
            transferSection.classList.add('hidden-below');
            generalizationSection.classList.add('hidden-below');
        }

    });
});

// Initialize sliders, dots, checkmarks, and video speed badges
function initializeInteractiveSections() {
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
        const isMethodCard = display.closest('.method-mppi-video') || display.closest('.method-card-media');
        if (isMethodCard) return;
        if (!isDemo && !isTeaser && !display.querySelector('.video-speed-badge')) {
            const badge = document.createElement('span');
            badge.className = 'video-speed-badge';
            badge.textContent = '3X';
            display.appendChild(badge);
        }
        display.querySelectorAll('video').forEach(v => {
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
        const rate = currentTask === 'pick-place' ? (68 / 150) : (49 / 100);
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

    updateSliderRange();
    updateVideo();
    updateSampleMethodVideo();
    updateTransferVideo();
    updateDemonstrationVideos();
    updateTransferDemonstrationVideos();
    updateInitialDemonstrationVideos();
    updateResultsImages();
    if (typeof refreshCharts === 'function') refreshCharts();
}

function updateSliderRange() {
    const maxIter = currentTask === 'push' ? 100 : 150;

    const slider = document.getElementById('iteration-slider');
    slider.max = maxIter;
    if (parseInt(slider.value) > maxIter) slider.value = maxIter;
    rebuildDots('slider-dots', maxIter, parseInt(slider.value));
    updateIteration(slider.value);

    const transferSlider = document.getElementById('transfer-iteration-slider');
    transferSlider.max = maxIter;
    if (parseInt(transferSlider.value) > maxIter) transferSlider.value = maxIter;
    rebuildDots('transfer-slider-dots', maxIter, parseInt(transferSlider.value));
    updateTransferIteration(transferSlider.value);

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
            const checkValues = currentTask === 'push' ? pushCheckValues : pickPlaceCheckValues;
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
    const folder = currentTask === 'push' ? 'Push' : 'Pick';
    const tableImg = document.getElementById('results-table-img');
    if (tableImg) tableImg.src = `Media/Image/Results/${folder}_table.png`;
    const tTableImg = document.getElementById('transfer-results-table-img');
    if (tTableImg) tTableImg.src = `Media/Image/Results/Transfer_${folder}_table.png`;
}

function updateDemonstrationVideos() {
    const taskFolder = currentTask === 'push' ? 'Push' : 'Pick';
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
    const iteration = parseInt(iterationSliderEl.value);
    const folder = sampleMethodFolderMap[currentSampleMethod] || 'mpail2_p';
    const taskFolder = currentTask === 'push' ? 'Push' : 'Pick';
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

    const taskFolder = currentTask === 'push' ? 'Push' : 'Pick';
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