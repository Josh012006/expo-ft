// ---------- Nav scroll border ----------
const nav = document.getElementById('nav');
const onScroll = () => {
  if (window.scrollY > 8) nav.classList.add('scrolled');
  else nav.classList.remove('scrolled');
};
window.addEventListener('scroll', onScroll, { passive: true });
onScroll();

// ---------- Evaluation: task tab switching ----------
const evalTasks = {
  cube:   { title: 'Cube Pick',                     desc: 'Autonomous 2x', src: 'videos/evaluation/pick_new_pirl_eval_2x.mp4' },
  flower: { title: 'Flower Insert',                 desc: 'Autonomous 2x', src: 'videos/evaluation/flower_pirl_eval_2x.mp4' },
  lights_route1: { title: 'Light - Route I',  desc: 'Autonomous 2x', src: 'videos/evaluation/light0_pirl_eval_2x.mp4' },
  lights_route2: { title: 'Light - Route II', desc: 'Autonomous 2x', src: 'videos/evaluation/light1_pirl_eval_2x.mp4' },
  lights_insert: { title: 'Light - Insert',   desc: 'Autonomous 2x', src: 'videos/evaluation/light2_pirl_eval_2x.mp4' },
  candy:  { title: 'Candy Scoop',                   desc: 'Autonomous 2x', src: 'videos/evaluation/scoop_pirl_eval_2x.mp4' },
  egg:    { title: 'Egg Flip',                      desc: 'Autonomous 2x', src: 'videos/evaluation/eggflip_pirl_eval_2x.mp4' },
  pool:   { title: 'Pool Shot',                     desc: 'Autonomous 2x', src: 'videos/evaluation/pooling_pirl_eval_2x.mp4' },
};

// Success counts out of 30 trials. 'EXPO-FT' is treated as "ours" and
// highlighted. Method sets differ per task (full baseline set vs. lite).
const FULL_METHODS = ['SFT', 'HG-DAgger', 'DSRL', 'HIL-SERL', 'EXPO-FT'];
const LITE_METHODS = ['SFT', 'HG-DAgger', 'EXPO-FT'];

const evalData = {
  cube:   { methods: FULL_METHODS, scores: [22, 22, 24, 0, 30] },
  flower: { methods: FULL_METHODS, scores: [14, 24, 12, 8, 30] },
  egg:    { methods: FULL_METHODS, scores: [16, 18, 15, 13, 30] },
  pool:   { methods: FULL_METHODS, scores: [23, 23, 25, 1, 30] },
  candy:  { methods: LITE_METHODS, scores: [22, 28, 30] },
  lights_route1: { methods: LITE_METHODS, scores: [23, 18, 30] },
  lights_route2: { methods: LITE_METHODS, scores: [21, 25, 30] },
  lights_insert: { methods: LITE_METHODS, scores: [23, 24, 30] },
};

const evalVideo = document.getElementById('eval-video');
const evalTitle = document.getElementById('eval-title');
const evalDesc  = document.getElementById('eval-desc');
const evalTabs  = document.querySelectorAll('#evaluation .task-tab');
const evalBars  = document.getElementById('eval-bars');

function buildBarGroup(methods, scores, title) {
  const group = document.createElement('div');
  group.className = 'bar-group';
  const cols = document.createElement('div');
  cols.className = 'bar-cols';
  methods.forEach((method, i) => {
    const v = scores[i];
    const pct = Math.round((v / 30) * 1000) / 10; // bar height %, 1 decimal
    const col = document.createElement('div');
    col.className = 'bar-col' + (method === 'EXPO-FT' ? ' is-ours' : '');
    col.style.setProperty('--h', pct + '%');
    const plot = document.createElement('div');
    plot.className = 'bar-plot';
    const val = document.createElement('span');
    val.className = 'bar-val';
    val.textContent = v + '/30';
    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    plot.append(val, fill);
    const label = document.createElement('span');
    label.className = 'bar-label';
    label.textContent = method;
    col.append(plot, label);
    cols.appendChild(col);
  });
  group.appendChild(cols);
  if (title) {
    const h = document.createElement('div');
    h.className = 'bar-group-title';
    h.textContent = title;
    group.appendChild(h);
  }
  return group;
}

function renderEvalBars(taskKey) {
  if (!evalBars) return;
  evalBars.innerHTML = '';
  const d = evalData[taskKey];
  if (!d) return;
  evalBars.appendChild(buildBarGroup(d.methods, d.scores));
}

evalTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const info = evalTasks[tab.dataset.task];
    if (!info) return;
    evalTabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    evalTitle.textContent = info.title;
    evalDesc.textContent = info.desc;
    evalVideo.querySelector('source').src = info.src;
    evalVideo.load();
    evalVideo.play().catch(() => {});
    renderEvalBars(tab.dataset.task);
  });
});

renderEvalBars('lights_route1');

// Keep the results panel the same height as the video
const evalResults = document.querySelector('.eval-results');
function syncEvalHeight() {
  if (!evalResults || !evalVideo) return;
  const h = evalVideo.getBoundingClientRect().height;
  if (h > 0) evalResults.style.height = h + 'px';
}
window.addEventListener('resize', syncEvalHeight);
window.addEventListener('load', syncEvalHeight);
evalVideo.addEventListener('loadedmetadata', syncEvalHeight);
requestAnimationFrame(syncEvalHeight);

// ---------- Copy BibTeX ----------
const copyBtn = document.getElementById('copy-bibtex');
copyBtn.addEventListener('click', async () => {
  const text = document.getElementById('bibtex-code').innerText;
  try {
    await navigator.clipboard.writeText(text);
    const orig = copyBtn.textContent;
    copyBtn.textContent = 'Copied';
    setTimeout(() => (copyBtn.textContent = orig), 1400);
  } catch (e) {
    copyBtn.textContent = 'Press Ctrl+C';
  }
});

// ---------- Horizontal scroll buttons ----------
document.querySelectorAll('.scroll-btn').forEach(btn => {
  const targetId = btn.dataset.target;
  const row = document.getElementById(targetId);
  if (!row) return;
  btn.addEventListener('click', () => {
    const dir = btn.classList.contains('scroll-btn-left') ? -1 : 1;
    const card = row.querySelector('.card');
    const step = card ? card.getBoundingClientRect().width + 20 : row.clientWidth * 0.8;
    row.scrollBy({ left: dir * step, behavior: 'smooth' });
  });
});

// Hide scroll buttons when at the edges
document.querySelectorAll('.scroll-row').forEach(row => {
  const wrap = row.closest('.scroll-row-wrap');
  if (!wrap) return;
  const left = wrap.querySelector('.scroll-btn-left');
  const right = wrap.querySelector('.scroll-btn-right');
  const update = () => {
    const atStart = row.scrollLeft <= 2;
    const atEnd = row.scrollLeft + row.clientWidth >= row.scrollWidth - 2;
    if (left) left.disabled = atStart;
    if (right) right.disabled = atEnd;
  };
  row.addEventListener('scroll', update, { passive: true });
  window.addEventListener('resize', update);
  update();
});

// ---------- Video bottom caption ----------
// Every card: name + speed tag as a gradient overlay (no top-left tag).
// Training is sped-up footage ("2x/10x", Pool "2x/6x"); Robustness uses the
// per-video label as the title; everything else is "Autonomous 1x".
document.querySelectorAll('figure.card').forEach(card => {
  const cap = card.querySelector('figcaption');
  if (!cap) return;
  const name = cap.textContent.trim();
  const section = card.closest('section');
  let title = name;
  let sub = 'Autonomous 1x';
  if (section && section.id === 'training') {
    sub = name === 'Pool Shot' ? '6x' : name === 'Candy Scoop' || name === 'String Light Routing - Route I' ? '15x' : '10x';
  } else if (section && section.id === 'robustness') {
    // Robustness: the figcaption is the label itself; "Autonomous 1x" below
    title = name;
    sub = 'Autonomous 1x';
  }

  const br = document.createElement('div');
  br.className = 'vid-tag-br';
  const brName = document.createElement('span');
  brName.className = 'vid-tag-name';
  brName.textContent = title;
  const brSub = document.createElement('span');
  brSub.className = 'vid-tag-sub';
  brSub.textContent = sub;
  br.append(brName, brSub);
  card.appendChild(br);
});

// ---------- Autoplay when in view, with a 3s pause between loops ----------
const allVideos = document.querySelectorAll('video');
const RESTART_DELAY = 2000;

const io = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    const v = entry.target;
    if (entry.isIntersecting) {
      v._inView = true;
      if (v.ended) v.currentTime = 0;
      v.play().catch(() => {});
    } else {
      v._inView = false;
      clearTimeout(v._restartTimer);
      v.pause();
    }
  });
}, { threshold: 0.1 });

allVideos.forEach(v => {
  v.muted = true;
  v.defaultMuted = true;
  v.loop = false; // manual loop so we can pause between plays
  v.playsInline = true;
  v.autoplay = true;
  v._inView = false;
  // Keep videos silent even if the controls are used to unmute
  v.addEventListener('volumechange', () => {
    if (!v.muted || v.volume > 0) {
      v.muted = true;
      v.volume = 0;
    }
  });
  // When a play finishes, wait before restarting (per-video delay if set)
  v.addEventListener('ended', () => {
    clearTimeout(v._restartTimer);
    v._restartTimer = setTimeout(() => {
      if (v._inView) {
        v.currentTime = 0;
        v.play().catch(() => {});
      }
    }, v._restartDelay || RESTART_DELAY);
  });
  v.play().catch(() => {});
  io.observe(v);
});

// ---------- Q / Sampling / Edit visualization video behavior ----------
// All three: hide the native control bar during the post-play pause.
// Sampling & Q: shorter 1s pause between loops.
// Sampling & Edit: a per-video button toggles speed (1x <-> 0.5x).
['vis-q', 'vis-sampling', 'vis-edit'].forEach(id => {
  const sec = document.getElementById(id);
  if (!sec) return;
  const withButton = id === 'vis-sampling' || id === 'vis-edit';
  sec.querySelectorAll('figure.card').forEach(card => {
    const v = card.querySelector('video');
    if (!v) return;

    // Sampling & Q clips restart faster (1s) than the default pause
    if (id === 'vis-q' || id === 'vis-sampling') v._restartDelay = 1000;

    // Hide the native control bar while the clip sits in its post-play pause,
    // restore it once playback resumes.
    v.addEventListener('ended', () => { v.removeAttribute('controls'); });
    v.addEventListener('play', () => { v.controls = true; });

    // Speed toggle button — Sampling & Edit only
    if (!withButton) return;

    const btn = document.createElement('button');
    btn.className = 'speed-btn';
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Toggle playback speed');
    card.appendChild(btn);

    let rate = 1; // default playback speed
    const sub = card.querySelector('.vid-tag-sub');
    const apply = () => {
      v.playbackRate = rate;
      btn.textContent = rate === 1 ? '1×' : '0.5×';
      if (sub) sub.textContent = 'Autonomous ' + (rate === 1 ? '1x' : '0.5x');
    };

    btn.addEventListener('click', e => {
      e.stopPropagation();
      rate = rate === 1 ? 0.5 : 1;
      apply();
    });

    // playbackRate can reset on metadata load; re-apply defensively
    v.addEventListener('loadedmetadata', apply);
    v.addEventListener('play', apply);
    apply();
  });
});

// These sections' videos are display-only: no controls, no pause/seek
['rollouts', 'robustness'].forEach(id => {
  const sec = document.getElementById(id);
  if (!sec) return;
  sec.querySelectorAll('video').forEach(v => {
    v.controls = false;
    v.removeAttribute('controls');
  });
});
