// Interactive visuals for SLAS (Lectures 0–10).
// Each visual is a small canvas "lab" with buttons and sliders.
// window.VIZ[id] = { lecture, title, desc, look: [what to observe], build(root) }
// window.SLIDE_VIZ[lecture][slide] = [ids] links visuals to the slides.
(function () {
  "use strict";

  // ------------------------------------------------------------------
  // Small helpers: colors, random numbers, linear algebra
  // ------------------------------------------------------------------
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function P() {
    return {
      c1: css("--c1"), c2: css("--c2"), c3: css("--c3"), c4: css("--c4"),
      ink: css("--ink"), muted: css("--muted"), line: css("--line"), surface: css("--surface"), paper: css("--paper")
    };
  }
  function rgb(hex) {
    const h = hex.replace("#", "");
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  function alpha(hex, a) { const [r, g, b] = rgb(hex); return `rgba(${r},${g},${b},${a})`; }
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function sigmoid(s) { return 1 / (1 + Math.exp(-s)); }
  function erf(x) { // Abramowitz–Stegun 7.1.26
    const t = 1 / (1 + 0.3275911 * Math.abs(x));
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return x >= 0 ? y : -y;
  }
  function Phi(x) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
  // Solve A w = b (Gaussian elimination with partial pivoting). A is n×n (array of rows).
  function solve(A, b) {
    const n = b.length;
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (Math.abs(M[p][c]) < 1e-12) return null;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < n; r++) {
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    const w = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) {
      let s = M[r][n];
      for (let k = r + 1; k < n; k++) s -= M[r][k] * w[k];
      w[r] = s / M[r][r];
    }
    return w;
  }
  // Polynomial least squares with ridge penalty lam on all weights: (XᵀX + λI)w = XᵀY
  function polyFit(xs, ys, deg, lam) {
    const m = deg + 1;
    const A = Array.from({ length: m }, () => new Array(m).fill(0));
    const b = new Array(m).fill(0);
    for (let i = 0; i < xs.length; i++) {
      const phi = [];
      let p = 1;
      for (let k = 0; k < m; k++) { phi.push(p); p *= xs[i]; }
      for (let r = 0; r < m; r++) {
        b[r] += phi[r] * ys[i];
        for (let c = 0; c < m; c++) A[r][c] += phi[r] * phi[c];
      }
    }
    for (let r = 0; r < m; r++) A[r][r] += lam;
    return solve(A, b);
  }
  function polyEval(w, x) { let s = 0, p = 1; for (const c of w) { s += c * p; p *= x; } return s; }
  function fmt(v, d) { return (Math.abs(v) < 1e-9 ? 0 : v).toFixed(d === undefined ? 2 : d); }

  // ------------------------------------------------------------------
  // Canvas stage, plot frame, axes
  // ------------------------------------------------------------------
  function stage(parent, ratio) {
    const wrap = document.createElement("div");
    wrap.className = "viz-canvas";
    const cv = document.createElement("canvas");
    wrap.append(cv);
    parent.append(wrap);
    const s = { cv, wrap, ctx: cv.getContext("2d"), W: 0, H: 0, draw: null };
    let lastW = 0;
    s.fit = function () {
      const w = Math.floor(wrap.clientWidth);
      if (!w || w === lastW) return;
      lastW = w;
      const h = Math.round(w * ratio);
      const d = window.devicePixelRatio || 1;
      cv.width = w * d; cv.height = h * d;
      cv.style.height = h + "px";
      s.ctx.setTransform(d, 0, 0, d, 0, 0);
      s.W = w; s.H = h;
      if (s.draw) s.draw();
    };
    if (window.ResizeObserver) new ResizeObserver(() => s.fit()).observe(wrap);
    requestAnimationFrame(() => s.fit());
    // Redraw when the theme changes
    const mq = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)");
    if (mq && mq.addEventListener) mq.addEventListener("change", () => s.draw && s.draw());
    new MutationObserver(() => s.draw && s.draw()).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    // Pointer position in CSS pixels
    s.onTap = function (fn) {
      cv.addEventListener("pointerdown", e => {
        const r = cv.getBoundingClientRect();
        fn(e.clientX - r.left, e.clientY - r.top, e);
      });
    };
    s.clear = function () {
      const p = P();
      s.ctx.fillStyle = p.surface;
      s.ctx.fillRect(0, 0, s.W, s.H);
    };
    return s;
  }

  function frame(s, xr, yr, o) {
    o = o || {};
    const pad = Object.assign({ l: 40, r: 12, t: 12, b: 30 }, o.pad || {});
    let [x0, x1] = xr, [y0, y1] = yr;
    const pw = Math.max(10, s.W - pad.l - pad.r), ph = Math.max(10, s.H - pad.t - pad.b);
    if (o.equal) {
      const sc = Math.max((x1 - x0) / pw, (y1 - y0) / ph);
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      x0 = cx - sc * pw / 2; x1 = cx + sc * pw / 2;
      y0 = cy - sc * ph / 2; y1 = cy + sc * ph / 2;
    }
    return {
      x0, x1, y0, y1, pad, pw, ph,
      X: x => pad.l + (x - x0) / (x1 - x0) * pw,
      Y: y => pad.t + (1 - (y - y0) / (y1 - y0)) * ph,
      IX: px => x0 + (px - pad.l) / pw * (x1 - x0),
      IY: py => y0 + (1 - (py - pad.t) / ph) * (y1 - y0)
    };
  }
  function ticks(a, b, n) {
    const span = b - a;
    if (span <= 0) return [a];
    const raw = span / (n || 5);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map(k => k * mag).find(k => k >= raw) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(Math.abs(v) < 1e-12 ? 0 : v);
    return out;
  }
  function axes(s, f, o) {
    o = o || {};
    const c = s.ctx, p = P();
    c.save();
    c.lineWidth = 1;
    c.font = "11px 'IBM Plex Mono', ui-monospace, monospace";
    c.fillStyle = p.muted;
    c.strokeStyle = alpha(p.muted.startsWith("#") ? p.muted : "#888888", 0.18);
    const tx = o.xticks || ticks(f.x0, f.x1, o.nx || 5);
    const ty = o.yticks || ticks(f.y0, f.y1, o.ny || 4);
    c.textAlign = "center"; c.textBaseline = "top";
    for (const v of tx) {
      const x = f.X(v);
      if (o.grid !== false) { c.beginPath(); c.moveTo(x, f.pad.t); c.lineTo(x, f.pad.t + f.ph); c.stroke(); }
      if (o.xlabels !== false) c.fillText(o.xfmt ? o.xfmt(v) : String(+v.toFixed(3)), x, f.pad.t + f.ph + 4);
    }
    c.textAlign = "right"; c.textBaseline = "middle";
    for (const v of ty) {
      const y = f.Y(v);
      if (o.grid !== false) { c.beginPath(); c.moveTo(f.pad.l, y); c.lineTo(f.pad.l + f.pw, y); c.stroke(); }
      if (o.ylabels !== false) c.fillText(o.yfmt ? o.yfmt(v) : String(+v.toFixed(3)), f.pad.l - 5, y);
    }
    c.strokeStyle = p.line;
    c.strokeRect(f.pad.l + 0.5, f.pad.t + 0.5, f.pw - 1, f.ph - 1);
    if (o.xl) { c.textAlign = "right"; c.textBaseline = "bottom"; c.fillStyle = p.muted; c.fillText(o.xl, f.pad.l + f.pw - 4, f.pad.t + f.ph - 3); }
    if (o.yl) { c.textAlign = "left"; c.textBaseline = "top"; c.fillStyle = p.muted; c.fillText(o.yl, f.pad.l + 5, f.pad.t + 4); }
    c.restore();
  }
  function clip(s, f) { s.ctx.save(); s.ctx.beginPath(); s.ctx.rect(f.pad.l, f.pad.t, f.pw, f.ph); s.ctx.clip(); }
  function curve(s, f, fn, color, width, dash, n) {
    const c = s.ctx;
    c.save();
    c.strokeStyle = color; c.lineWidth = width || 2;
    if (dash) c.setLineDash(dash);
    c.beginPath();
    n = n || 200;
    let started = false;
    for (let i = 0; i <= n; i++) {
      const x = f.x0 + (f.x1 - f.x0) * i / n;
      const y = fn(x);
      if (!isFinite(y)) { started = false; continue; }
      const py = Math.max(-1e4, Math.min(1e4, f.Y(y)));
      if (!started) { c.moveTo(f.X(x), py); started = true; } else c.lineTo(f.X(x), py);
    }
    c.stroke();
    c.restore();
  }
  // Class markers: +1 = filled circle (color 1), −1 = cross (color 2). Shape + color.
  function mark(s, x, y, cls, color, r) {
    const c = s.ctx;
    r = r || 4.5;
    c.save();
    if (cls > 0) {
      c.fillStyle = color; c.strokeStyle = P().surface; c.lineWidth = 1.5;
      c.beginPath(); c.arc(x, y, r, 0, 2 * Math.PI); c.fill(); c.stroke();
    } else if (cls < 0) {
      c.strokeStyle = color; c.lineWidth = 2.2;
      c.beginPath(); c.moveTo(x - r, y - r); c.lineTo(x + r, y + r); c.moveTo(x + r, y - r); c.lineTo(x - r, y + r); c.stroke();
    } else {
      c.fillStyle = color;
      c.beginPath(); c.arc(x, y, r * 0.8, 0, 2 * Math.PI); c.fill();
    }
    c.restore();
  }
  function label(s, text, x, y, color, align, base, bold) {
    const c = s.ctx;
    c.save();
    c.font = (bold ? "600 " : "") + "12px 'IBM Plex Sans', system-ui, sans-serif";
    c.textAlign = align || "left"; c.textBaseline = base || "middle";
    // halo for legibility over marks
    c.lineWidth = 3; c.strokeStyle = P().surface; c.strokeText(text, x, y);
    c.fillStyle = color || P().ink;
    c.fillText(text, x, y);
    c.restore();
  }
  // Low-resolution heatmap, stretched to the plot area. fn(x, y) -> [r,g,b,a(0..255)]
  function heat(s, f, nx, ny, fn) {
    const off = document.createElement("canvas");
    off.width = nx; off.height = ny;
    const oc = off.getContext("2d");
    const img = oc.createImageData(nx, ny);
    for (let j = 0; j < ny; j++) {
      const y = f.y1 - (j + 0.5) / ny * (f.y1 - f.y0);
      for (let i = 0; i < nx; i++) {
        const x = f.x0 + (i + 0.5) / nx * (f.x1 - f.x0);
        const col = fn(x, y);
        const k = 4 * (j * nx + i);
        img.data[k] = col[0]; img.data[k + 1] = col[1]; img.data[k + 2] = col[2]; img.data[k + 3] = col[3];
      }
    }
    oc.putImageData(img, 0, 0);
    s.ctx.save();
    s.ctx.imageSmoothingEnabled = true;
    s.ctx.drawImage(off, f.pad.l, f.pad.t, f.pw, f.ph);
    s.ctx.restore();
  }

  // Animation loop that stops by itself when the canvas leaves the page.
  function looper(s, step) {
    const L = { running: false };
    let id = null;
    function tick() {
      if (!s.cv.isConnected) { L.running = false; return; }
      const more = step();
      if (more === false) { L.running = false; if (L.onStop) L.onStop(); return; }
      id = requestAnimationFrame(tick);
    }
    L.start = function () { if (L.running) return; L.running = true; id = requestAnimationFrame(tick); };
    L.stop = function () { L.running = false; if (id) cancelAnimationFrame(id); id = null; };
    return L;
  }

  // ------------------------------------------------------------------
  // Controls
  // ------------------------------------------------------------------
  function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; }
  function controls(root) { const c = el("div", "viz-controls"); root.append(c); return c; }
  function btn(parent, text, fn, primary) {
    const b = el("button", "vbtn" + (primary ? " primary" : ""), text);
    b.type = "button";
    b.addEventListener("click", fn);
    parent.append(b);
    return b;
  }
  let sliderId = 0;
  function slider(parent, name, min, max, step, value, show, fn) {
    const id = "vs" + (++sliderId);
    const w = el("label", "vslider");
    w.setAttribute("for", id);
    const top = el("span", "vslider-top");
    const nm = el("span", "", name);
    const val = el("span", "vval tnum");
    top.append(nm, val);
    const inp = el("input");
    inp.type = "range"; inp.id = id; inp.min = min; inp.max = max; inp.step = step; inp.value = value;
    w.append(top, inp);
    parent.append(w);
    const S = {
      get value() { return +inp.value; },
      set value(v) { inp.value = v; val.textContent = show(+inp.value); },
      input: inp
    };
    val.textContent = show(+inp.value);
    inp.addEventListener("input", () => { val.textContent = show(+inp.value); fn(+inp.value); });
    return S;
  }
  function seg(parent, options, value, fn) {
    const w = el("div", "vseg");
    const buttons = options.map(([v, text]) => {
      const b = el("button", "", text);
      b.type = "button";
      b.setAttribute("aria-pressed", String(v === value));
      b.addEventListener("click", () => {
        buttons.forEach(x => x.setAttribute("aria-pressed", String(x === b)));
        fn(v);
      });
      w.append(b);
      return b;
    });
    parent.append(w);
    return w;
  }
  function readout(root) {
    const r = el("div", "viz-readout");
    root.append(r);
    return { set: html => { r.innerHTML = html; } };
  }
  function legend(root, items) {
    const l = el("div", "viz-legend");
    for (const [color, text, shape] of items) {
      const i = el("span", "vleg");
      const sw = el("i", "sw " + (shape || "line"));
      sw.style.setProperty("--sw", `var(${color})`);
      i.append(sw, document.createTextNode(text));
      l.append(i);
    }
    root.append(l);
  }

  const VIZ = {};
  window.VIZ = VIZ;

  // ==================================================================
  // L0 – Supervised vs unsupervised
  // ==================================================================
  VIZ.supunsup = {
    lecture: 0, title: "Supervised vs unsupervised",
    desc: "The same points, with and without labels.",
    look: ["With labels you learn a boundary that predicts the class of new points (classification).",
      "Without labels you can only discover structure: here k-means finds two groups (clustering)."],
    build(root) {
      const pts = [];
      for (let i = 0; i < 30; i++) pts.push({ x: 2 + randn() * 0.7, y: 2 + randn() * 0.7, c: -1 });
      for (let i = 0; i < 30; i++) pts.push({ x: 5 + randn() * 0.7, y: 5 + randn() * 0.7, c: 1 });
      let mode = "sup", groups = null, cents = null;
      const s = stage(root, 0.7);
      const ctl = controls(root);
      seg(ctl, [["sup", "With labels"], ["uns", "Without labels"]], mode, v => { mode = v; groups = null; s.draw(); upd(); });
      const kb = btn(ctl, "Find 2 groups (k-means)", () => {
        cents = [{ x: pts[0].x, y: pts[0].y }, { x: pts[59].x, y: pts[59].y }];
        for (let it = 0; it < 10; it++) {
          groups = pts.map(p => (Math.hypot(p.x - cents[0].x, p.y - cents[0].y) < Math.hypot(p.x - cents[1].x, p.y - cents[1].y) ? 0 : 1));
          for (let j = 0; j < 2; j++) {
            const m = pts.filter((_, i) => groups[i] === j);
            if (m.length) cents[j] = { x: m.reduce((a, p) => a + p.x, 0) / m.length, y: m.reduce((a, p) => a + p.y, 0) / m.length };
          }
        }
        s.draw();
      });
      const ro = readout(root);
      function upd() {
        kb.hidden = mode !== "uns";
        ro.set(mode === "sup"
          ? "<b>Supervised:</b> every point has a label (● dog, × cat). We learn the line that separates them."
          : groups ? "<b>Unsupervised:</b> no labels, but k-means found the two groups (colors) and their centroids (◆)." : "<b>Unsupervised:</b> same points, no labels. Tap <i>Find 2 groups</i>.");
      }
      s.draw = function () {
        const p = P(); s.clear();
        const f = frame(s, [0, 7.5], [0, 7.5], { equal: true });
        axes(s, f, { xl: "weight", yl: "height" });
        clip(s, f);
        if (mode === "sup") {
          s.ctx.strokeStyle = p.ink; s.ctx.lineWidth = 2; s.ctx.setLineDash([6, 4]);
          s.ctx.beginPath(); s.ctx.moveTo(f.X(0), f.Y(7)); s.ctx.lineTo(f.X(7), f.Y(0)); s.ctx.stroke(); s.ctx.setLineDash([]);
        }
        pts.forEach((q, i) => {
          if (mode === "sup") mark(s, f.X(q.x), f.Y(q.y), q.c, q.c > 0 ? p.c1 : p.c2);
          else mark(s, f.X(q.x), f.Y(q.y), 0, groups ? [p.c3, p.c4][groups[i]] : p.muted);
        });
        if (mode === "uns" && cents) cents.forEach((c0, j) => {
          const x = f.X(c0.x), y = f.Y(c0.y);
          s.ctx.fillStyle = [p.c3, p.c4][j]; s.ctx.strokeStyle = p.surface; s.ctx.lineWidth = 2;
          s.ctx.beginPath(); s.ctx.moveTo(x, y - 8); s.ctx.lineTo(x + 8, y); s.ctx.lineTo(x, y + 8); s.ctx.lineTo(x - 8, y); s.ctx.closePath(); s.ctx.fill(); s.ctx.stroke();
        });
        s.ctx.restore();
      };
      upd();
    }
  };

  // ==================================================================
  // L1 – Hoeffding: the bin experiment
  // ==================================================================
  VIZ.hoeffding = {
    lecture: 1, title: "Hoeffding: the bin experiment",
    desc: "Draw N marbles many times and look at how far p̂ falls from the true p.",
    look: ["The histogram of p̂ gets narrower as N grows: p̂ concentrates around p.",
      "The red area is the fraction of samples with |p̂ − p| > ε. It is always below the Hoeffding bound 2e^(−2ε²N).",
      "Small ε or small N makes the bound larger than 1: true but useless."],
    build(root) {
      let p = 0.6, N = 20, eps = 0.1, counts = new Array(41).fill(0), total = 0, bad = 0;
      const s = stage(root, 0.62);
      const ctl = controls(root);
      const run = looper(s, () => {
        for (let e = 0; e < 40; e++) {
          let blue = 0;
          for (let i = 0; i < N; i++) if (Math.random() < p) blue++;
          const ph = blue / N;
          counts[Math.round(ph * 40)]++;
          total++;
          if (Math.abs(ph - p) > eps + 1e-12) bad++;
        }
        s.draw(); upd();
        return total < 4000;
      });
      run.onStop = () => { playB.textContent = "Run"; };
      const playB = btn(ctl, "Run", () => { if (run.running) { run.stop(); playB.textContent = "Run"; } else { if (total >= 4000) reset(); run.start(); playB.textContent = "Pause"; } }, true);
      btn(ctl, "Reset", () => { reset(); s.draw(); upd(); });
      function reset() { counts = new Array(41).fill(0); total = 0; bad = 0; }
      slider(ctl, "N (sample size)", 5, 300, 5, N, v => v, v => { N = v; reset(); s.draw(); upd(); });
      slider(ctl, "ε (accuracy)", 0.02, 0.3, 0.01, eps, v => v.toFixed(2), v => { eps = v; reset(); s.draw(); upd(); });
      slider(ctl, "p (true blue fraction)", 0.1, 0.9, 0.05, p, v => v.toFixed(2), v => { p = v; reset(); s.draw(); upd(); });
      const ro = readout(root);
      function upd() {
        const bound = 2 * Math.exp(-2 * eps * eps * N);
        ro.set(`Samples drawn: <b class="tnum">${total}</b><br>Observed ℙ[|p̂ − p| > ε] = <b class="tnum">${total ? (100 * bad / total).toFixed(1) : "–"}%</b><br>Hoeffding bound 2e^(−2ε²N) = <b class="tnum">${bound >= 1 ? (100 * bound).toFixed(0) + "% (useless, ≥ 100%)" : (100 * bound).toFixed(1) + "%"}</b>`);
      }
      s.draw = function () {
        const P_ = P(); s.clear();
        const maxC = Math.max(1, ...counts);
        const f = frame(s, [0, 1], [0, 1.05], { pad: { l: 14 } });
        axes(s, f, { ylabels: false, xl: "p̂", ny: 2 });
        clip(s, f);
        const c = s.ctx;
        // shaded "bad" zones
        c.fillStyle = alpha(P_.c2, 0.10);
        c.fillRect(f.X(0), f.pad.t, f.X(Math.max(0, p - eps)) - f.X(0), f.ph);
        c.fillRect(f.X(Math.min(1, p + eps)), f.pad.t, f.X(1) - f.X(Math.min(1, p + eps)), f.ph);
        const bw = f.pw / 41;
        counts.forEach((n, i) => {
          if (!n) return;
          const v = i / 40;
          const h = n / maxC;
          c.fillStyle = Math.abs(v - p) > eps + 1e-9 ? P_.c2 : P_.c1;
          const x = f.X(v) - bw / 2 + 1, y = f.Y(h);
          c.beginPath();
          if (c.roundRect) c.roundRect(x, y, bw - 2, f.Y(0) - y, [3, 3, 0, 0]); else c.rect(x, y, bw - 2, f.Y(0) - y);
          c.fill();
        });
        c.strokeStyle = P_.ink; c.lineWidth = 2;
        c.beginPath(); c.moveTo(f.X(p), f.pad.t); c.lineTo(f.X(p), f.pad.t + f.ph); c.stroke();
        c.setLineDash([4, 4]); c.strokeStyle = P_.muted;
        [p - eps, p + eps].forEach(v => { c.beginPath(); c.moveTo(f.X(v), f.pad.t); c.lineTo(f.X(v), f.pad.t + f.ph); c.stroke(); });
        c.setLineDash([]);
        label(s, "p", f.X(p) + 4, f.pad.t + 10, P_.ink, "left", "middle", true);
        label(s, "p ± ε", f.X(Math.min(0.98, p + eps)) + 4, f.pad.t + 26, P_.muted);
        c.restore();
      };
      upd();
    }
  };

  // ==================================================================
  // L1 – Many hypotheses: the best-looking coin
  // ==================================================================
  VIZ.multibin = {
    lecture: 1, title: "Why M appears: the best-looking coin",
    desc: "M fair coins (p = 0.5), each flipped N times. We pick the coin with the fewest heads, like a learning algorithm picks the model with the lowest E_in.",
    look: ["With one coin, a sample far from 0.5 is rare.",
      "With many coins, some coin almost surely looks very good by pure chance: its p̂ is far from its real p.",
      "This is why the bound becomes 2M·e^(−2ε²N): the guarantee must cover all M candidates."],
    build(root) {
      const Ms = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
      let mi = 4, N = 10, eps = 0.3, last = [], trials = 0, badChosen = 0;
      const s = stage(root, 0.58);
      const ctl = controls(root);
      function flipAll() {
        const M = Ms[mi];
        last = [];
        for (let m = 0; m < M; m++) { let h = 0; for (let i = 0; i < N; i++) if (Math.random() < 0.5) h++; last.push(h / N); }
        const best = Math.min(...last);
        trials++;
        if (Math.abs(best - 0.5) > eps + 1e-9) badChosen++;
      }
      btn(ctl, "Flip all coins", () => { flipAll(); s.draw(); upd(); }, true);
      btn(ctl, "Repeat 500 times", () => { for (let i = 0; i < 500; i++) flipAll(); s.draw(); upd(); });
      slider(ctl, "M (number of coins)", 0, Ms.length - 1, 1, mi, v => Ms[v], v => { mi = v; trials = 0; badChosen = 0; flipAll(); s.draw(); upd(); });
      slider(ctl, "N (flips per coin)", 5, 50, 1, N, v => v, v => { N = v; trials = 0; badChosen = 0; flipAll(); s.draw(); upd(); });
      const ro = readout(root);
      function upd() {
        const M = Ms[mi];
        const one = 2 * Math.exp(-2 * eps * eps * N), bound = M * one;
        const best = last.length ? Math.min(...last) : NaN;
        ro.set(`Chosen coin (fewest heads): p̂ = <b class="tnum">${fmt(best)}</b>, real p = 0.5<br>
          Over ${trials} trials, the chosen coin had |p̂ − 0.5| > ${eps}: <b class="tnum">${trials ? (100 * badChosen / trials).toFixed(1) : "–"}%</b><br>
          Bound with M = ${M}: 2M·e^(−2ε²N) = <b class="tnum">${bound >= 1 ? "≥ 100% (useless)" : (100 * bound).toFixed(1) + "%"}</b> · one coin: ${(100 * Math.min(1, one)).toFixed(1)}%`);
      }
      s.draw = function () {
        const p = P(); s.clear();
        const bins = new Array(N + 1).fill(0);
        last.forEach(v => bins[Math.round(v * N)]++);
        const maxB = Math.max(1, ...bins);
        const f = frame(s, [-0.05, 1.05], [0, maxB * 1.15], { pad: { l: 34 } });
        axes(s, f, { xl: "p̂ of each coin", ny: 3 });
        clip(s, f);
        const c = s.ctx;
        const bw = Math.max(3, f.pw / (N + 1) * 0.8);
        const best = Math.min(...last);
        bins.forEach((n, k) => {
          if (!n) return;
          const v = k / N;
          c.fillStyle = Math.abs(v - best) < 1e-9 ? p.c2 : p.c1;
          const y = f.Y(n);
          c.beginPath();
          if (c.roundRect) c.roundRect(f.X(v) - bw / 2, y, bw, f.Y(0) - y, [3, 3, 0, 0]); else c.rect(f.X(v) - bw / 2, y, bw, f.Y(0) - y);
          c.fill();
        });
        c.strokeStyle = p.ink; c.setLineDash([4, 4]);
        c.beginPath(); c.moveTo(f.X(0.5), f.pad.t); c.lineTo(f.X(0.5), f.pad.t + f.ph); c.stroke(); c.setLineDash([]);
        label(s, "true p = 0.5", f.X(0.5) + 4, f.pad.t + 10, p.ink);
        if (isFinite(best)) label(s, "chosen coin", f.X(best), f.Y(bins[Math.round(best * N)]) - 10, p.c2, "center", "bottom", true);
        c.restore();
      };
      flipAll(); trials = 0; badChosen = 0;
      upd();
    }
  };

  // ==================================================================
  // L2 – Least squares: tap to add points
  // ==================================================================
  VIZ.ols = {
    lecture: 2, title: "Least squares, live",
    desc: "The OLS line ŵ = (XᵀX)⁻¹XᵀY minimizes the sum of the squared residuals (the shaded squares).",
    look: ["Tap the plot to add a point: the line and E_in update instantly.",
      "Add an outlier: one far point pulls the whole line, because its error counts squared (Lecture 10, data cleaning).",
      "With fewer than 2 different x values, XᵀX is singular: no unique line."],
    build(root) {
      let pts = [], showSq = true;
      function fresh() { pts = []; for (let i = 0; i < 12; i++) { const x = 0.5 + Math.random() * 9; pts.push({ x, y: 1 + 0.8 * x + randn() }); } }
      fresh();
      const s = stage(root, 0.66);
      const ctl = controls(root);
      btn(ctl, "New data", () => { fresh(); s.draw(); upd(); }, true);
      btn(ctl, "Add outlier", () => { pts.push({ x: 8.5 + Math.random(), y: -1.5 }); s.draw(); upd(); });
      btn(ctl, "Remove last", () => { pts.pop(); s.draw(); upd(); });
      btn(ctl, "Clear", () => { pts = []; s.draw(); upd(); });
      seg(ctl, [[true, "Show squares"], [false, "Hide squares"]], true, v => { showSq = v; s.draw(); });
      const ro = readout(root);
      let f = null;
      s.onTap((px, py) => {
        if (!f) return;
        const x = f.IX(px), y = f.IY(py);
        if (x < f.x0 || x > f.x1 || y < f.y0 || y > f.y1) return;
        pts.push({ x, y }); s.draw(); upd();
      });
      function fit() {
        if (pts.length < 2) return null;
        const w = polyFit(pts.map(p => p.x), pts.map(p => p.y), 1, 0);
        if (!w) return null;
        const xs = pts.map(p => p.x);
        if (Math.max(...xs) - Math.min(...xs) < 1e-6) return null;
        return w;
      }
      function upd() {
        const w = fit();
        if (!w) { ro.set(`N = ${pts.length}. <b>XᵀX is singular</b>: add points with at least two different x values.`); return; }
        const ein = pts.reduce((a, p) => a + Math.pow(p.y - w[0] - w[1] * p.x, 2), 0) / pts.length;
        ro.set(`ŷ = <b class="tnum">${fmt(w[0])}</b> + <b class="tnum">${fmt(w[1])}</b>·x &nbsp;·&nbsp; N = ${pts.length} &nbsp;·&nbsp; E_in = (1/N)Σ(y − ŷ)² = <b class="tnum">${fmt(ein, 3)}</b>`);
      }
      s.draw = function () {
        const p = P(); s.clear();
        f = frame(s, [0, 10], [-3, 12], { equal: false });
        axes(s, f, { xl: "x", yl: "y" });
        clip(s, f);
        const w = fit();
        const c = s.ctx;
        if (w) {
          pts.forEach(q => {
            const yh = w[0] + w[1] * q.x;
            const py = f.Y(q.y), ph = f.Y(yh);
            if (showSq) {
              const side = Math.abs(ph - py);
              c.fillStyle = alpha(p.c2, 0.13); c.strokeStyle = alpha(p.c2, 0.5); c.lineWidth = 1;
              c.fillRect(f.X(q.x), Math.min(py, ph), side, side); c.strokeRect(f.X(q.x), Math.min(py, ph), side, side);
            }
            c.strokeStyle = p.c2; c.lineWidth = 1.5;
            c.beginPath(); c.moveTo(f.X(q.x), py); c.lineTo(f.X(q.x), ph); c.stroke();
          });
          curve(s, f, x => w[0] + w[1] * x, p.c1, 2.5);
        }
        pts.forEach(q => mark(s, f.X(q.x), f.Y(q.y), 0, p.ink, 5));
        c.restore();
        if (!pts.length) label(s, "Tap to add points", s.W / 2, s.H / 2, p.muted, "center");
      };
      upd();
    }
  };

  // ==================================================================
  // L2 – The E_in bowl and the singular valley
  // ==================================================================
  VIZ.bowl = {
    lecture: 2, title: "E_in is a bowl (or a valley)",
    desc: "E_in(w₀, w₁) as a map: darker = lower error. Tap anywhere to start gradient descent from there.",
    look: ["With informative data the bowl has one lowest point: the OLS solution (positive definite Hessian).",
      "If all inputs are equal, XᵀX is singular: the minimum becomes a whole line (a valley). GD stops at different points depending on where it starts.",
      "This is the “data not informative” case of slide 13."],
    build(root) {
      let mode = "ok", data = [], path = [];
      function make() {
        data = [];
        for (let i = 0; i < 10; i++) {
          const x = mode === "ok" ? 1 + Math.random() * 4 : 3;
          data.push({ x, y: 1 + 0.6 * x + randn() * 0.4 });
        }
        path = [];
      }
      make();
      const s = stage(root, 0.66);
      const ctl = controls(root);
      seg(ctl, [["ok", "Varied inputs"], ["sing", "All x equal (singular)"]], mode, v => { mode = v; make(); s.draw(); upd(); });
      btn(ctl, "New data", () => { make(); s.draw(); upd(); });
      const ro = readout(root);
      const E = (w0, w1) => data.reduce((a, d) => a + Math.pow(d.y - w0 - w1 * d.x, 2), 0) / data.length;
      // Gradient and step size 1/L, L = largest eigenvalue of the Hessian (2/N)XᵀX
      function grad(w0, w1) {
        let g0 = 0, g1 = 0;
        for (const d of data) { const r = w0 + w1 * d.x - d.y; g0 += 2 * r / data.length; g1 += 2 * r * d.x / data.length; }
        return [g0, g1];
      }
      function stepSize() {
        let a = 0, b = 0, c = 0;
        for (const d of data) { a += 1; b += d.x; c += d.x * d.x; }
        a *= 2 / data.length; b *= 2 / data.length; c *= 2 / data.length;
        const L = (a + c) / 2 + Math.sqrt(((a - c) / 2) ** 2 + b * b);
        return 1 / L;
      }
      const run = looper(s, () => {
        const [w0, w1] = path[path.length - 1];
        const g = grad(w0, w1), eta = stepSize();
        path.push([w0 - eta * g[0], w1 - eta * g[1]]);
        s.draw(); upd();
        return path.length < 120 && Math.hypot(g[0], g[1]) > 1e-4;
      });
      let f = null;
      s.onTap((px, py) => {
        if (!f) return;
        path = [[f.IX(px), f.IY(py)]];
        run.stop(); run.start();
      });
      function upd() {
        const w = mode === "ok" ? polyFit(data.map(d => d.x), data.map(d => d.y), 1, 0) : null;
        const end = path.length ? path[path.length - 1] : null;
        ro.set((mode === "ok" && w ? `Unique minimum (OLS): w₀ = <b class="tnum">${fmt(w[0])}</b>, w₁ = <b class="tnum">${fmt(w[1])}</b>.` : `<b>XᵀX singular</b>: every point on the dashed line w₀ + 3w₁ = ȳ is a minimum.`) +
          (end ? `<br>GD after ${path.length - 1} steps: (${fmt(end[0])}, ${fmt(end[1])}), E_in = <b class="tnum">${fmt(E(end[0], end[1]), 3)}</b>` : "<br>Tap the map to start gradient descent."));
      }
      s.draw = function () {
        const p = P(); s.clear();
        f = frame(s, [-3, 5], [-0.8, 2.2], { pad: { l: 40 } });
        // sequential single hue: low error = strong accent
        let emin = Infinity, emax = 0;
        for (let i = 0; i < 20; i++) for (let j = 0; j < 20; j++) {
          const e = E(f.x0 + (f.x1 - f.x0) * i / 19, f.y0 + (f.y1 - f.y0) * j / 19);
          emin = Math.min(emin, e); emax = Math.max(emax, e);
        }
        const [r, g, b] = rgb(p.c1);
        heat(s, f, 90, 60, (x, y) => {
          const t = Math.log(1 + Math.max(0, E(x, y) - emin)) / Math.log(1 + emax - emin);
          return [r, g, b, Math.round(255 * 0.85 * Math.pow(1 - Math.min(1, t), 1.6))];
        });
        axes(s, f, { xl: "w₀", yl: "w₁", grid: false });
        clip(s, f);
        const c = s.ctx;
        if (mode === "ok") {
          const w = polyFit(data.map(d => d.x), data.map(d => d.y), 1, 0);
          if (w) { c.fillStyle = p.c2; c.strokeStyle = p.surface; c.lineWidth = 2; c.beginPath(); c.arc(f.X(w[0]), f.Y(w[1]), 6, 0, 7); c.fill(); c.stroke(); label(s, "minimum", f.X(w[0]) + 9, f.Y(w[1]), p.ink, "left", "middle", true); }
        } else {
          const yb = data.reduce((a, d) => a + d.y, 0) / data.length;
          c.strokeStyle = p.c2; c.lineWidth = 2; c.setLineDash([6, 4]);
          c.beginPath(); c.moveTo(f.X(yb - 3 * f.y0), f.Y(f.y0)); c.lineTo(f.X(yb - 3 * f.y1), f.Y(f.y1)); c.stroke(); c.setLineDash([]);
          label(s, "valley of minimizers", f.X(yb - 3 * 1.6) + 8, f.Y(1.6), p.ink, "left", "middle", true);
        }
        if (path.length) {
          c.strokeStyle = p.ink; c.lineWidth = 2;
          c.beginPath(); path.forEach(([a, b2], i) => (i ? c.lineTo(f.X(a), f.Y(b2)) : c.moveTo(f.X(a), f.Y(b2)))); c.stroke();
          const [a, b2] = path[path.length - 1];
          c.fillStyle = p.ink; c.beginPath(); c.arc(f.X(a), f.Y(b2), 4, 0, 7); c.fill();
        }
        c.restore();
      };
      upd();
    }
  };

  // ==================================================================
  // L3 – Perceptron Learning Algorithm
  // ==================================================================
  VIZ.pla = {
    lecture: 3, title: "PLA step by step",
    desc: "At each step PLA picks one misclassified point (ringed) and updates w ← w + y·x. The arrow is w: it points to the +1 side.",
    look: ["Each update rotates/shifts the line toward classifying the chosen point correctly (not always in one step).",
      "On separable data PLA stops after a finite number of updates (E_in = 0).",
      "Switch to non-separable data: there is always a misclassified point, so PLA never stops."],
    build(root) {
      let pts = [], w = [0, 0, 0], it = 0, pick = -1, sep = true;
      function make() {
        pts = [];
        const a = Math.random() * Math.PI, nx = Math.cos(a), ny = Math.sin(a), off = (Math.random() - 0.5) * 0.6;
        while (pts.length < 30) {
          const x = Math.random() * 2 - 1, y = Math.random() * 2 - 1;
          const d = nx * x + ny * y - off;
          if (Math.abs(d) < 0.12) continue;
          let c = d > 0 ? 1 : -1;
          if (!sep && Math.random() < 0.12) c = -c;
          pts.push({ x, y, c });
        }
        w = [0, 0, 0]; it = 0; pick = -1;
      }
      make();
      const s = stage(root, 0.8);
      const ctl = controls(root);
      const sign = v => (v >= 0 ? 1 : -1);
      const mis = () => pts.map((q, i) => (sign(w[0] + w[1] * q.x + w[2] * q.y) !== q.c ? i : -1)).filter(i => i >= 0);
      function step() {
        const m = mis();
        if (!m.length) { pick = -1; return false; }
        pick = m[Math.floor(Math.random() * m.length)];
        const q = pts[pick];
        w = [w[0] + q.c, w[1] + q.c * q.x, w[2] + q.c * q.y];
        it++;
        return true;
      }
      let last = 0;
      const run = looper(s, () => {
        const now = performance.now();
        if (now - last < 350) return true;
        last = now;
        const more = step(); s.draw(); upd();
        return more && it < 2000;
      });
      run.onStop = () => { playB.textContent = "Play"; };
      btn(ctl, "One step", () => { run.stop(); playB.textContent = "Play"; step(); s.draw(); upd(); }, true);
      const playB = btn(ctl, "Play", () => { if (run.running) { run.stop(); playB.textContent = "Play"; } else { run.start(); playB.textContent = "Pause"; } });
      btn(ctl, "New data", () => { run.stop(); playB.textContent = "Play"; make(); s.draw(); upd(); });
      seg(ctl, [[true, "Separable"], [false, "Not separable"]], true, v => { sep = v; run.stop(); playB.textContent = "Play"; make(); s.draw(); upd(); });
      const ro = readout(root);
      function upd() {
        const m = mis().length;
        ro.set(`Updates: <b class="tnum">${it}</b> &nbsp;·&nbsp; misclassified: <b class="tnum">${m}</b> (E_in = ${fmt(m / pts.length)})<br>w = [${fmt(w[0])}, ${fmt(w[1])}, ${fmt(w[2])}] ${m === 0 && it > 0 ? "&nbsp;·&nbsp; <b>Converged: perfect classification.</b>" : ""}${!sep && it > 200 ? "&nbsp;·&nbsp; Still not converged: the data are not separable." : ""}`);
      }
      s.draw = function () {
        const p = P(); s.clear();
        const f = frame(s, [-1.1, 1.1], [-1.1, 1.1], { equal: true });
        axes(s, f, { xl: "x₁", yl: "x₂", nx: 4, ny: 4 });
        clip(s, f);
        const c = s.ctx;
        // shade the +1 half-plane
        if (Math.hypot(w[1], w[2]) > 1e-9) {
          const [r, g, b] = rgb(p.c1), [r2, g2, b2] = rgb(p.c2);
          heat(s, f, 60, 60, (x, y) => (w[0] + w[1] * x + w[2] * y >= 0 ? [r, g, b, 26] : [r2, g2, b2, 22]));
          // boundary line w0 + w1 x + w2 y = 0
          c.strokeStyle = p.ink; c.lineWidth = 2;
          c.beginPath();
          if (Math.abs(w[2]) > Math.abs(w[1])) { c.moveTo(f.X(-2), f.Y((-w[0] + 2 * w[1]) / w[2])); c.lineTo(f.X(2), f.Y((-w[0] - 2 * w[1]) / w[2])); }
          else { c.moveTo(f.X((-w[0] + 2 * w[2]) / w[1]), f.Y(-2)); c.lineTo(f.X((-w[0] - 2 * w[2]) / w[1]), f.Y(2)); }
          c.stroke();
          // normal vector from the point of the line closest to the origin
          const n2 = w[1] * w[1] + w[2] * w[2];
          const ox = -w[0] * w[1] / n2, oy = -w[0] * w[2] / n2;
          const L = 0.35 / Math.sqrt(n2);
          const ex = ox + w[1] * L, ey = oy + w[2] * L;
          c.strokeStyle = p.c3; c.fillStyle = p.c3; c.lineWidth = 2.5;
          c.beginPath(); c.moveTo(f.X(ox), f.Y(oy)); c.lineTo(f.X(ex), f.Y(ey)); c.stroke();
          const ang = Math.atan2(f.Y(ey) - f.Y(oy), f.X(ex) - f.X(ox));
          c.beginPath(); c.moveTo(f.X(ex), f.Y(ey));
          c.lineTo(f.X(ex) - 10 * Math.cos(ang - 0.4), f.Y(ey) - 10 * Math.sin(ang - 0.4));
          c.lineTo(f.X(ex) - 10 * Math.cos(ang + 0.4), f.Y(ey) - 10 * Math.sin(ang + 0.4));
          c.closePath(); c.fill();
          label(s, "w", f.X(ex) + 6, f.Y(ey), p.c3, "left", "middle", true);
        }
        pts.forEach((q, i) => {
          mark(s, f.X(q.x), f.Y(q.y), q.c, q.c > 0 ? p.c1 : p.c2);
          if (i === pick) { c.strokeStyle = p.ink; c.lineWidth = 2; c.beginPath(); c.arc(f.X(q.x), f.Y(q.y), 11, 0, 7); c.stroke(); }
        });
        c.restore();
        if (it === 0) label(s, "Press One step", s.W / 2, f.pad.t + 14, p.muted, "center");
      };
      upd();
    }
  };

  // ==================================================================
  // L4 – Logistic regression: sigmoid fit
  // ==================================================================
  VIZ.sigmoid = {
    lecture: 4, title: "Logistic regression: fit the sigmoid",
    desc: "Labels are drawn from a true probability (dashed). Move w and b, or let gradient descent minimize the cross-entropy.",
    look: ["h(s) = eˢ/(1 + eˢ) always stays between 0 and 1: it is a probability.",
      "w controls how steep the curve is, b shifts it. The 50% point is at x = −b/w (the decision boundary).",
      "Gradient descent lowers E_in step by step and ends close to the true curve."],
    build(root) {
      let data = [], w = 0.5, b = 0, tw = 1.5, tb = -0.5;
      function make() {
        data = [];
        for (let i = 0; i < 40; i++) { const x = Math.random() * 8 - 4; data.push({ x, y: Math.random() < sigmoid(tw * x + tb) ? 1 : -1, j: Math.random() }); }
      }
      make();
      const Ein = () => data.reduce((a, d) => a + Math.log(1 + Math.exp(-d.y * (w * d.x + b))), 0) / data.length;
      const s = stage(root, 0.62);
      const ctl = controls(root);
      const run = looper(s, () => {
        let g0 = 0, g1 = 0;
        for (const d of data) { const t = -d.y / (1 + Math.exp(d.y * (w * d.x + b))); g0 += t / data.length; g1 += t * d.x / data.length; }
        b -= 0.8 * g0; w -= 0.8 * g1;
        sw.value = Math.max(-4, Math.min(4, w)); sb.value = Math.max(-4, Math.min(4, b));
        s.draw(); upd();
        return Math.hypot(g0, g1) > 2e-3 && ++steps < 400;
      });
      let steps = 0;
      run.onStop = () => { gdB.textContent = "Gradient descent"; };
      const gdB = btn(ctl, "Gradient descent", () => { if (run.running) { run.stop(); gdB.textContent = "Gradient descent"; } else { steps = 0; run.start(); gdB.textContent = "Stop"; } }, true);
      btn(ctl, "New data", () => { make(); s.draw(); upd(); });
      const sw = slider(ctl, "w (steepness)", -4, 4, 0.05, w, v => v.toFixed(2), v => { w = v; s.draw(); upd(); });
      const sb = slider(ctl, "b (shift)", -4, 4, 0.05, b, v => v.toFixed(2), v => { b = v; s.draw(); upd(); });
      const ro = readout(root);
      function upd() {
        ro.set(`E_in = (1/N)Σ ln(1 + e^(−y·(wx + b))) = <b class="tnum">${fmt(Ein(), 3)}</b><br>Decision boundary (h = 0.5): x = −b/w = <b class="tnum">${Math.abs(w) > 1e-6 ? fmt(-b / w) : "∞"}</b> &nbsp;·&nbsp; true: ${fmt(-tb / tw)}`);
      }
      legend(root, [["--c1", "your model h(wx + b)"], ["--muted", "true P(y = +1 | x)", "dash"]]);
      s.draw = function () {
        const p = P(); s.clear();
        const f = frame(s, [-4, 4], [-0.08, 1.08]);
        axes(s, f, { xl: "x", yticks: [0, 0.5, 1] });
        clip(s, f);
        curve(s, f, x => sigmoid(tw * x + tb), p.muted, 2, [6, 4]);
        curve(s, f, x => sigmoid(w * x + b), p.c1, 2.5);
        data.forEach(d => mark(s, f.X(d.x), f.Y(d.y > 0 ? 1 - 0.04 * d.j : 0.04 * d.j), d.y, d.y > 0 ? p.c1 : p.c2, 4));
        s.ctx.restore();
        label(s, "y = +1", f.pad.l + 6, f.Y(0.93), p.muted);
        label(s, "y = −1", f.pad.l + 6, f.Y(0.09), p.muted);
      };
      upd();
    }
  };

  // ==================================================================
  // L4 – Pointwise errors compared
  // ==================================================================
  VIZ.errfun = {
    lecture: 4, title: "Pointwise errors compared",
    desc: "The error of one point as a function of y·s (positive = correct side, large = confident).",
    look: ["0/1 error (perceptron) is a step: flat almost everywhere, so its gradient is useless.",
      "Cross-entropy ln(1 + e^(−y·s)) is smooth: confident and right costs ≈ 0, confident and wrong grows linearly.",
      "Squared error (1 − y·s)² also punishes points that are “too correct”."],
    build(root) {
      let z = 0.5;
      const s = stage(root, 0.6);
      const ctl = controls(root);
      slider(ctl, "y·s for one point", -3, 3, 0.05, z, v => v.toFixed(2), v => { z = v; s.draw(); upd(); });
      const ro = readout(root);
      legend(root, [["--c2", "0/1 error ⟦y ≠ sign(s)⟧"], ["--c1", "cross-entropy ln(1 + e^(−y·s))"], ["--c3", "squared (1 − y·s)²"]]);
      const fns = [z => (z < 0 ? 1 : 0), z => Math.log(1 + Math.exp(-z)), z => (1 - z) * (1 - z)];
      function upd() {
        ro.set(`At y·s = ${fmt(z)}: &nbsp;0/1 = <b class="tnum">${fns[0](z)}</b> · cross-entropy = <b class="tnum">${fmt(fns[1](z), 3)}</b> · squared = <b class="tnum">${fmt(fns[2](z), 2)}</b>`);
      }
      s.draw = function () {
        const p = P(); s.clear();
        const f = frame(s, [-3, 3], [0, 4]);
        axes(s, f, { xl: "y·s", yl: "error" });
        clip(s, f);
        const cols = [p.c2, p.c1, p.c3];
        // step function drawn as two segments
        const c = s.ctx;
        c.strokeStyle = cols[0]; c.lineWidth = 2.5;
        c.beginPath(); c.moveTo(f.X(-3), f.Y(1)); c.lineTo(f.X(0), f.Y(1)); c.lineTo(f.X(0), f.Y(0)); c.lineTo(f.X(3), f.Y(0)); c.stroke();
        curve(s, f, fns[1], cols[1], 2.5);
        curve(s, f, fns[2], cols[2], 2.5);
        c.strokeStyle = p.muted; c.setLineDash([3, 3]); c.lineWidth = 1;
        c.beginPath(); c.moveTo(f.X(z), f.pad.t); c.lineTo(f.X(z), f.pad.t + f.ph); c.stroke(); c.setLineDash([]);
        fns.forEach((fn, i) => { const v = fn(z); if (v <= 4) { c.fillStyle = cols[i]; c.strokeStyle = p.surface; c.lineWidth = 2; c.beginPath(); c.arc(f.X(z), f.Y(v), 5, 0, 7); c.fill(); c.stroke(); } });
        c.restore();
        label(s, "← wrong side", f.X(-1.5), f.Y(2.6), p.muted, "center");
        label(s, "right side →", f.X(1.5), f.Y(2.6), p.muted, "center");
      };
      upd();
    }
  };

  // ==================================================================
  // L4 – Gradient descent: step size, SGD, Newton
  // ==================================================================
  VIZ.gd = {
    lecture: 4, title: "Gradient descent: the step size",
    desc: "Minimize a stretched bowl E(w) starting from the same point. Try different η and methods.",
    look: ["Small η: safe but slow. Larger η: faster, then it starts to bounce across the narrow direction.",
      "Above η = 2/a_max the iterates diverge.",
      "SGD uses a noisy gradient (one point at a time): a zig-zag path that keeps jittering near the minimum.",
      "Newton's method uses the Hessian: on a quadratic it reaches the minimum in one step."],
    build(root) {
      const a1 = 1, a2 = 6, th = Math.PI / 6, ct = Math.cos(th), st = Math.sin(th);
      // rotate into the bowl's own axes: u along the flat direction, v along the steep one
      const toUV = (x, y) => [ct * x + st * y, -st * x + ct * y];
      const E = (x, y) => { const [u, v] = toUV(x, y); return 0.5 * (a1 * u * u + a2 * v * v); };
      const grad = (x, y) => { const [u, v] = toUV(x, y); const gu = a1 * u, gv = a2 * v; return [ct * gu - st * gv, st * gu + ct * gv]; };
      let eta = 0.1, method = "gd", path = [[-2.6, 1.6]];
      const s = stage(root, 0.75);
      const ctl = controls(root);
      let last = 0;
      const run = looper(s, () => {
        const now = performance.now();
        if (now - last < 110) return true;
        last = now;
        const [x, y] = path[path.length - 1];
        let g = grad(x, y);
        let nx, ny;
        if (method === "newton") {
          // H⁻¹∇E: in (u,v) coordinates divide each component by its curvature
          const [u, v] = toUV(x, y);
          const du = u, dv = v; // (a1 u)/a1, (a2 v)/a2
          nx = x - (ct * du - st * dv); ny = y - (st * du + ct * dv);
        } else {
          if (method === "sgd") g = [g[0] + randn() * 1.2, g[1] + randn() * 1.2];
          nx = x - eta * g[0]; ny = y - eta * g[1];
        }
        path.push([nx, ny]);
        s.draw(); upd();
        const big = Math.hypot(nx, ny) > 50;
        return path.length < 80 && !big && (method === "sgd" || Math.hypot(nx, ny) > 1e-3);
      });
      run.onStop = () => { goB.textContent = "Run"; };
      const goB = btn(ctl, "Run", () => { if (run.running) { run.stop(); goB.textContent = "Run"; } else { path = [path[0]]; run.start(); goB.textContent = "Stop"; } }, true);
      seg(ctl, [["gd", "GD"], ["sgd", "SGD"], ["newton", "Newton"]], method, v => { method = v; run.stop(); goB.textContent = "Run"; path = [path[0]]; s.draw(); upd(); });
      slider(ctl, "η (step size)", 0.01, 0.4, 0.01, eta, v => v.toFixed(2) + (v > 2 / a2 ? "  > 2/a_max" : ""), v => { eta = v; run.stop(); goB.textContent = "Run"; path = [path[0]]; s.draw(); upd(); });
      const ro = readout(root);
      let f = null;
      s.onTap((px, py) => { if (!f) return; run.stop(); goB.textContent = "Run"; path = [[f.IX(px), f.IY(py)]]; s.draw(); upd(); });
      function upd() {
        const [x, y] = path[path.length - 1];
        const e = E(x, y);
        let status = "";
        if (method === "gd") status = eta > 2 / a2 ? "<b>diverges</b> (η > 2/a_max)" : eta > 1 / a2 ? "converges while <b>bouncing</b> (1/a_max < η < 2/a_max)" : "converges <b>monotonically</b> along the steep direction";
        if (method === "sgd") status = "noisy steps: never settles exactly at the minimum";
        if (method === "newton") status = "one step to the minimum (exact on a quadratic)";
        ro.set(`Step <b class="tnum">${path.length - 1}</b> &nbsp;·&nbsp; E = <b class="tnum">${e > 1e5 ? "huge" : fmt(e, 4)}</b><br>Curvatures a = ${a1} and ${a2} ⇒ 2/a_max = ${fmt(2 / a2, 3)} &nbsp;·&nbsp; ${status}<br><span class="muted">Tap the plot to choose the starting point.</span>`);
      }
      s.draw = function () {
        const p = P(); s.clear();
        f = frame(s, [-3.2, 3.2], [-2.4, 2.4], { equal: true });
        axes(s, f, { xl: "w₁", yl: "w₂", nx: 6, ny: 4 });
        clip(s, f);
        const c = s.ctx;
        // contour ellipses E = level
        [0.05, 0.2, 0.5, 1, 2, 3.5, 5.5, 8].forEach((lv, i) => {
          c.strokeStyle = alpha(p.c1, 0.25 + 0.05 * (8 - i)); c.lineWidth = 1.2;
          c.beginPath();
          for (let k = 0; k <= 80; k++) {
            const t = 2 * Math.PI * k / 80;
            const u = Math.sqrt(2 * lv / a1) * Math.cos(t), v = Math.sqrt(2 * lv / a2) * Math.sin(t);
            const x = ct * u - st * v, y = st * u + ct * v;
            k ? c.lineTo(f.X(x), f.Y(y)) : c.moveTo(f.X(x), f.Y(y));
          }
          c.stroke();
        });
        c.fillStyle = p.c3; c.beginPath(); c.arc(f.X(0), f.Y(0), 5, 0, 7); c.fill();
        label(s, "minimum", f.X(0) + 8, f.Y(0) + 12, p.c3, "left", "middle", true);
        c.strokeStyle = p.c2; c.lineWidth = 2;
        c.beginPath();
        path.forEach(([x, y], i) => { const X = Math.max(-1e4, Math.min(1e4, f.X(x))), Y = Math.max(-1e4, Math.min(1e4, f.Y(y))); i ? c.lineTo(X, Y) : c.moveTo(X, Y); });
        c.stroke();
        path.forEach(([x, y], i) => { c.fillStyle = i === 0 ? p.ink : p.c2; c.beginPath(); c.arc(f.X(x), f.Y(y), i === 0 ? 5 : 3, 0, 7); c.fill(); });
        c.restore();
      };
      upd();
    }
  };

  // ==================================================================
  // L5 – Bias and variance on a sine target
  // ==================================================================
  VIZ.biasvar = {
    lecture: 5, title: "Bias and variance, simulated",
    desc: "Target f(x) = sin(πx). Draw many datasets of N points, fit a model to each one, and average the fits.",
    look: ["Each thin line is one model h^𝒟 learned from one dataset. The thick line is the average model ḡ(x).",
      "Bias = distance between ḡ and f. Variance = how spread the thin lines are around ḡ.",
      "With N = 2 the constant beats the line (0.75 vs ≈1.9): the line has huge variance. Increase N and the line wins."],
    build(root) {
      let model = 0, N = 2, fits = [];
      const grid = Array.from({ length: 61 }, (_, i) => -1 + 2 * i / 60);
      const f0 = x => Math.sin(Math.PI * x);
      const s = stage(root, 0.62);
      const ctl = controls(root);
      function draw1() {
        const xs = [], ys = [];
        for (let i = 0; i < N; i++) { const x = Math.random() * 2 - 1; xs.push(x); ys.push(f0(x)); }
        const w = polyFit(xs, ys, model, model === 0 ? 0 : 1e-6) || [0];
        fits.push(w);
      }
      const run = looper(s, () => { for (let i = 0; i < 8; i++) draw1(); s.draw(); upd(); return fits.length < 1000; });
      run.onStop = () => { goB.textContent = "Draw datasets"; };
      const goB = btn(ctl, "Draw datasets", () => { if (run.running) { run.stop(); goB.textContent = "Draw datasets"; } else { if (fits.length >= 1000) fits = []; run.start(); goB.textContent = "Pause"; } }, true);
      btn(ctl, "Reset", () => { run.stop(); goB.textContent = "Draw datasets"; fits = []; s.draw(); upd(); });
      seg(ctl, [[0, "Constant"], [1, "Line"], [3, "Cubic"]], model, v => { model = v; fits = []; s.draw(); upd(); });
      slider(ctl, "N (points per dataset)", 2, 20, 1, N, v => v, v => { N = v; fits = []; s.draw(); upd(); });
      const ro = readout(root);
      legend(root, [["--muted", "target f(x) = sin(πx)", "dash"], ["--c1", "one fit h^𝒟 (thin)"], ["--c2", "average model ḡ(x)"]]);
      function stats() {
        if (!fits.length) return null;
        let bias = 0, varr = 0;
        const gbar = grid.map(x => fits.reduce((a, w) => a + polyEval(w, x), 0) / fits.length);
        grid.forEach((x, i) => {
          bias += Math.pow(gbar[i] - f0(x), 2);
          varr += fits.reduce((a, w) => a + Math.pow(polyEval(w, x) - gbar[i], 2), 0) / fits.length;
        });
        return { gbar, bias: bias / grid.length, varr: varr / grid.length };
      }
      function upd() {
        const st = stats();
        ro.set(st ? `Datasets: <b class="tnum">${fits.length}</b> &nbsp;·&nbsp; bias = <b class="tnum">${fmt(st.bias)}</b> &nbsp;·&nbsp; variance = <b class="tnum">${fmt(st.varr)}</b> &nbsp;·&nbsp; 𝔼[E_out] = bias + var = <b class="tnum">${fmt(st.bias + st.varr)}</b>` : "Press <b>Draw datasets</b>.");
      }
      s.draw = function () {
        const p = P(); s.clear();
        const f = frame(s, [-1, 1], [-2.2, 2.2]);
        axes(s, f, { xl: "x", nx: 4 });
        clip(s, f);
        const shown = fits.slice(-120);
        shown.forEach(w => curve(s, f, x => polyEval(w, x), alpha(p.c1, 0.16), 1.2, null, 60));
        curve(s, f, f0, p.muted, 2, [6, 4]);
        const st = stats();
        if (st) {
          const c = s.ctx;
          c.strokeStyle = p.c2; c.lineWidth = 3; c.beginPath();
          grid.forEach((x, i) => (i ? c.lineTo(f.X(x), f.Y(st.gbar[i])) : c.moveTo(f.X(x), f.Y(st.gbar[i]))));
          c.stroke();
        }
        s.ctx.restore();
      };
      upd();
    }
  };

  // ==================================================================
  // L5 – Learning curves
  // ==================================================================
  VIZ.curves = {
    lecture: 5, title: "Learning curves",
    desc: "Average E_in and E_out as the number of data N grows, for a fixed model (sine target + noise σ = 0.3).",
    look: ["E_out decreases with N (variance goes down), E_in increases (more points are harder to fit).",
      "Both converge to the same level: bias + σ² (dashed).",
      "Compare models: the simple one converges fast to a high level; the complex one needs more data but ends lower."],
    build(root) {
      let model = 1, res = null;
      const sig = 0.3, f0 = x => Math.sin(Math.PI * x);
      const s = stage(root, 0.62);
      const ctl = controls(root);
      seg(ctl, [[0, "Constant"], [1, "Line"], [3, "Cubic"]], model, v => { model = v; compute(); s.draw(); upd(); });
      const ro = readout(root);
      legend(root, [["--c1", "𝔼[E_out]"], ["--c2", "𝔼[E_in]"], ["--muted", "bias + σ²", "dash"]]);
      const gridX = Array.from({ length: 41 }, (_, i) => -1 + i / 20);
      function compute() {
        const Ns = [];
        for (let n = model + 2; n <= 40; n += (n < 10 ? 1 : 3)) Ns.push(n);
        res = { Ns, ein: [], eout: [], floor: 0 };
        // floor: best possible E_out for this model (large-N fit) + noise
        const big = gridX.map(x => x);
        const wInf = polyFit(big, big.map(f0), model, 1e-9);
        res.floor = gridX.reduce((a, x) => a + Math.pow(polyEval(wInf, x) - f0(x), 2), 0) / gridX.length + sig * sig;
        for (const n of Ns) {
          let ein = 0, eout = 0;
          const T = 200;
          for (let t = 0; t < T; t++) {
            const xs = [], ys = [];
            for (let i = 0; i < n; i++) { const x = Math.random() * 2 - 1; xs.push(x); ys.push(f0(x) + sig * randn()); }
            const w = polyFit(xs, ys, model, 1e-6);
            ein += xs.reduce((a, x, i) => a + Math.pow(polyEval(w, x) - ys[i], 2), 0) / n;
            eout += gridX.reduce((a, x) => a + Math.pow(polyEval(w, x) - f0(x), 2), 0) / gridX.length + sig * sig;
          }
          res.ein.push(ein / T); res.eout.push(eout / T);
        }
      }
      function upd() {
        const k = res.Ns.length - 1;
        ro.set(`At N = ${res.Ns[k]}: E_in ≈ <b class="tnum">${fmt(res.ein[k], 3)}</b>, E_out ≈ <b class="tnum">${fmt(res.eout[k], 3)}</b>, asymptote bias + σ² ≈ <b class="tnum">${fmt(res.floor, 3)}</b>`);
      }
      s.draw = function () {
        const p = P(); s.clear();
        const ymax = Math.min(1.2, Math.max(...res.eout.slice(1), res.floor * 1.6, 0.3));
        const f = frame(s, [0, 40], [0, ymax]);
        axes(s, f, { xl: "N", yl: "expected error" });
        clip(s, f);
        const c = s.ctx;
        const line = (arr, col) => { c.strokeStyle = col; c.lineWidth = 2.5; c.beginPath(); res.Ns.forEach((n, i) => { const y = f.Y(Math.min(arr[i], ymax * 3)); i ? c.lineTo(f.X(n), y) : c.moveTo(f.X(n), y); }); c.stroke(); };
        c.strokeStyle = p.muted; c.setLineDash([6, 4]); c.lineWidth = 1.5;
        c.beginPath(); c.moveTo(f.X(0), f.Y(res.floor)); c.lineTo(f.X(40), f.Y(res.floor)); c.stroke(); c.setLineDash([]);
        line(res.eout, p.c1); line(res.ein, p.c2);
        c.restore();
      };
      compute(); upd();
    }
  };

  // ==================================================================
  // L6 – Overfitting and regularization (polynomial fit)
  // ==================================================================
  VIZ.polyfit = {
    lecture: 6, title: "Overfitting and ridge regularization",
    desc: "Fit a polynomial of degree m to 15 noisy points of sin(πx). A big test set measures E_out.",
    look: ["Low degree: underfitting (both errors high). Very high degree: E_in → 0 but the curve wiggles and E_out explodes (overfitting).",
      "Keep degree 10 and raise λ: the wiggles flatten and E_out drops. Too much λ underfits again (w → 0).",
      "The bottom chart is the U-shaped E_out of Lecture 5, measured."],
    build(root) {
      let deg = 3, lgl = -6, train = [], test = [];
      const f0 = x => Math.sin(Math.PI * x), sig = 0.25;
      function make() {
        train = []; test = [];
        for (let i = 0; i < 15; i++) { const x = -1 + 2 * (i + Math.random()) / 15; train.push({ x, y: f0(x) + sig * randn() }); }
        for (let i = 0; i < 300; i++) { const x = Math.random() * 2 - 1; test.push({ x, y: f0(x) + sig * randn() }); }
      }
      make();
      const lam = () => (lgl <= -6 ? 1e-9 : Math.pow(10, lgl));
      const err = (w, set) => set.reduce((a, d) => a + Math.pow(polyEval(w, d.x) - d.y, 2), 0) / set.length;
      const fitW = (m, l) => polyFit(train.map(d => d.x), train.map(d => d.y), m, l) || [0];
      const s = stage(root, 0.58);
      const s2 = stage(root, 0.36);
      const ctl = controls(root);
      btn(ctl, "New data", () => { make(); s.draw(); s2.draw(); upd(); }, true);
      slider(ctl, "degree m", 0, 11, 1, deg, v => v, v => { deg = v; s.draw(); s2.draw(); upd(); });
      slider(ctl, "λ (log₁₀)", -6, 1, 0.1, lgl, v => (v <= -6 ? "0" : "10^" + v.toFixed(1)), v => { lgl = v; s.draw(); s2.draw(); upd(); });
      const ro = readout(root);
      legend(root, [["--c1", "fit"], ["--muted", "target sin(πx)", "dash"], ["--c2", "E_in (train)"], ["--c3", "E_out (test)"]]);
      function upd() {
        const w = fitW(deg, lam());
        ro.set(`degree ${deg}, λ = ${lgl <= -6 ? "0" : Math.pow(10, lgl).toPrecision(2)} &nbsp;·&nbsp; E_in = <b class="tnum">${fmt(err(w, train), 3)}</b> &nbsp;·&nbsp; E_out ≈ <b class="tnum">${fmt(err(w, test), 3)}</b> &nbsp;·&nbsp; ‖w‖ = ${fmt(Math.sqrt(w.reduce((a, v) => a + v * v, 0)), 1)}`);
      }
      s.draw = function () {
        const p = P(); s.clear();
        const f = frame(s, [-1, 1], [-2, 2]);
        axes(s, f, { xl: "x", nx: 4 });
        clip(s, f);
        curve(s, f, f0, p.muted, 2, [6, 4]);
        const w = fitW(deg, lam());
        curve(s, f, x => polyEval(w, x), p.c1, 2.5, null, 300);
        train.forEach(d => mark(s, f.X(d.x), f.Y(d.y), 0, p.ink, 5));
        s.ctx.restore();
      };
      s2.draw = function () {
        const p = P(); s2.clear();
        const ein = [], eout = [];
        for (let m = 0; m <= 11; m++) { const w = fitW(m, lam()); ein.push(err(w, train)); eout.push(err(w, test)); }
        const f = frame(s2, [-0.5, 11.5], [0, Math.min(1, Math.max(0.3, ...eout.slice(0, 8)))], { pad: { b: 26, t: 8 } });
        axes(s2, f, { xl: "degree m", ny: 3, xticks: [0, 2, 4, 6, 8, 10] });
        clip(s2, f);
        const c = s2.ctx;
        const line = (arr, col) => { c.strokeStyle = col; c.lineWidth = 2; c.beginPath(); arr.forEach((v, m) => (m ? c.lineTo(f.X(m), f.Y(Math.min(v, 50))) : c.moveTo(f.X(m), f.Y(v)))); c.stroke(); arr.forEach((v, m) => { c.fillStyle = col; c.beginPath(); c.arc(f.X(m), f.Y(Math.min(v, 50)), m === deg ? 5 : 2.5, 0, 7); c.fill(); }); };
        line(ein, p.c2); line(eout, p.c3);
        c.restore();
      };
      upd();
    }
  };

  // ==================================================================
  // L6 – Cross-validation
  // ==================================================================
  VIZ.cv = {
    lecture: 6, title: "Cross-validation picks the degree",
    desc: "Split 24 points into K folds. Each fold is used once for validation while the model is trained on the others.",
    look: ["Play: the highlighted fold is the validation set; the line is trained on the rest; the red bars are its validation errors.",
      "Bottom: E_cv for every degree. Its minimum is the model we choose. E_in (orange) keeps decreasing and would always pick the most complex model.",
      "K = 24 is leave-one-out."],
    build(root) {
      let K = 4, deg = 3, fold = 0, data = [];
      const f0 = x => Math.sin(Math.PI * x);
      function make() { data = []; for (let i = 0; i < 24; i++) { const x = Math.random() * 2 - 1; data.push({ x, y: f0(x) + 0.3 * randn(), k: 0 }); } shuffleFolds(); }
      function shuffleFolds() { const idx = data.map((_, i) => i).sort(() => Math.random() - 0.5); idx.forEach((i, j) => { data[i].k = j % K; }); }
      make();
      const fitOn = (set, m) => polyFit(set.map(d => d.x), set.map(d => d.y), m, 1e-6) || [0];
      function ecv(m) {
        let e = 0;
        for (let k = 0; k < K; k++) {
          const w = fitOn(data.filter(d => d.k !== k), m);
          data.filter(d => d.k === k).forEach(d => { e += Math.pow(polyEval(w, d.x) - d.y, 2); });
        }
        return e / data.length;
      }
      const s = stage(root, 0.55);
      const s2 = stage(root, 0.36);
      const ctl = controls(root);
      let last = 0;
      const run = looper(s, () => { const now = performance.now(); if (now - last > 900) { last = now; fold = (fold + 1) % K; s.draw(); upd(); } return true; });
      const playB = btn(ctl, "Play folds", () => { if (run.running) { run.stop(); playB.textContent = "Play folds"; } else { run.start(); playB.textContent = "Pause"; } }, true);
      btn(ctl, "New data", () => { make(); fold = 0; s.draw(); s2.draw(); upd(); });
      slider(ctl, "K (folds)", 0, 4, 1, 1, v => [2, 4, 6, 12, 24][v] + ([2, 4, 6, 12, 24][v] === 24 ? " (LOO)" : ""), v => { K = [2, 4, 6, 12, 24][v]; fold = 0; shuffleFolds(); s.draw(); s2.draw(); upd(); });
      slider(ctl, "degree m", 0, 8, 1, deg, v => v, v => { deg = v; s.draw(); s2.draw(); upd(); });
      const ro = readout(root);
      legend(root, [["--c1", "training points / fit"], ["--c2", "validation fold", "dot"], ["--c3", "E_cv"], ["--c4", "E_in"]]);
      function upd() {
        const all = []; for (let m = 0; m <= 8; m++) all.push(ecv(m));
        const best = all.indexOf(Math.min(...all));
        ro.set(`Fold ${fold + 1} of ${K} is validating &nbsp;·&nbsp; E_cv(degree ${deg}) = <b class="tnum">${fmt(all[deg], 3)}</b> &nbsp;·&nbsp; best degree by CV: <b>${best}</b>`);
      }
      s.draw = function () {
        const p = P(); s.clear();
        const f = frame(s, [-1, 1], [-2, 2]);
        axes(s, f, { xl: "x", nx: 4 });
        clip(s, f);
        const w = fitOn(data.filter(d => d.k !== fold), deg);
        curve(s, f, x => polyEval(w, x), p.c1, 2.5, null, 200);
        const c = s.ctx;
        data.forEach(d => {
          if (d.k === fold) {
            c.strokeStyle = p.c2; c.lineWidth = 2; c.beginPath(); c.moveTo(f.X(d.x), f.Y(d.y)); c.lineTo(f.X(d.x), f.Y(polyEval(w, d.x))); c.stroke();
            mark(s, f.X(d.x), f.Y(d.y), 0, p.c2, 6);
          } else mark(s, f.X(d.x), f.Y(d.y), 0, alpha(p.c1, 0.7), 4);
        });
        c.restore();
      };
      s2.draw = function () {
        const p = P(); s2.clear();
        const cv = [], ein = [];
        for (let m = 0; m <= 8; m++) { cv.push(ecv(m)); const w = fitOn(data, m); ein.push(data.reduce((a, d) => a + Math.pow(polyEval(w, d.x) - d.y, 2), 0) / data.length); }
        const f = frame(s2, [-0.5, 8.5], [0, Math.min(1.2, Math.max(0.3, ...cv.slice(0, 7)) * 1.1)], { pad: { t: 8, b: 26 } });
        axes(s2, f, { xl: "degree m", ny: 3, xticks: [0, 2, 4, 6, 8] });
        clip(s2, f);
        const c = s2.ctx;
        const best = cv.indexOf(Math.min(...cv));
        const line = (arr, col) => { c.strokeStyle = col; c.lineWidth = 2; c.beginPath(); arr.forEach((v, m) => (m ? c.lineTo(f.X(m), f.Y(Math.min(v, 9))) : c.moveTo(f.X(m), f.Y(v)))); c.stroke(); };
        line(ein, p.c4); line(cv, p.c3);
        cv.forEach((v, m) => { c.fillStyle = p.c3; c.beginPath(); c.arc(f.X(m), f.Y(Math.min(v, 9)), m === best ? 6 : 3, 0, 7); c.fill(); });
        c.restore();
        label(s2, "min E_cv", f.X(best), Math.max(14, f.Y(cv[best]) - 12), p.ink, "center", "bottom", true);
      };
      upd();
    }
  };

  // ==================================================================
  // L7 – Feature transform: circle → line
  // ==================================================================
  VIZ.transform = {
    lecture: 7, title: "Feature transform: circle → line",
    desc: "Map every point x = (x₁, x₂) to z = (x₁², x₂²). Slide to morph between the two spaces.",
    look: ["In x-space the classes are separated by a circle: no line can do it.",
      "In z-space the same points are separated by the straight line z₁ + z₂ = r²: a perceptron can learn it.",
      "Linear in z, nonlinear in x: the trick behind Lecture 7."],
    build(root) {
      const r = 0.65, pts = [];
      while (pts.length < 80) {
        const x = Math.random() * 2 - 1, y = Math.random() * 2 - 1;
        const d = Math.hypot(x, y);
        if (Math.abs(d - r) < 0.08) continue;
        pts.push({ x, y, c: d < r ? 1 : -1 });
      }
      let t = 0;
      const s = stage(root, 0.85);
      const ctl = controls(root);
      let dir = 1;
      const run = looper(s, () => { t = Math.max(0, Math.min(1, t + dir * 0.02)); sl.value = t; s.draw(); upd(); return t > 0 && t < 1; });
      btn(ctl, "Animate", () => { dir = t >= 1 ? -1 : 1; if (t >= 1) t = 0.999; if (t <= 0) t = 0.001; run.start(); }, true);
      const sl = slider(ctl, "x-space → z-space", 0, 1, 0.01, t, v => (v < 0.02 ? "x = (x₁, x₂)" : v > 0.98 ? "z = (x₁², x₂²)" : Math.round(100 * v) + "%"), v => { t = v; s.draw(); upd(); });
      const ro = readout(root);
      function upd() {
        ro.set(t < 0.02 ? "x-space: the boundary is the <b>circle</b> x₁² + x₂² = r²." : t > 0.98 ? "z-space: the boundary is the <b>line</b> z₁ + z₂ = r²: ŷ = sign(r² − z₁ − z₂) = sign(wᵀz)." : "Morphing: every point moves from (x₁, x₂) to (x₁², x₂²).");
      }
      s.draw = function () {
        const p = P(); s.clear();
        const e = t * t * (3 - 2 * t);
        const f = frame(s, [-1.1 + 1.1 * e, 1.1], [-1.1 + 1.1 * e, 1.1], { equal: true });
        axes(s, f, { xl: e > 0.5 ? "z₁ = x₁²" : "x₁", yl: e > 0.5 ? "z₂ = x₂²" : "x₂", nx: 4, ny: 4 });
        clip(s, f);
        const c = s.ctx;
        // boundary: interpolate the circle points the same way
        c.strokeStyle = p.ink; c.lineWidth = 2; c.setLineDash([6, 4]);
        c.beginPath();
        for (let k = 0; k <= 120; k++) {
          const a = 2 * Math.PI * k / 120;
          const x = r * Math.cos(a), y = r * Math.sin(a);
          const X = (1 - e) * x + e * x * x, Y = (1 - e) * y + e * y * y;
          k ? c.lineTo(f.X(X), f.Y(Y)) : c.moveTo(f.X(X), f.Y(Y));
        }
        c.stroke(); c.setLineDash([]);
        pts.forEach(q => {
          const X = (1 - e) * q.x + e * q.x * q.x, Y = (1 - e) * q.y + e * q.y * q.y;
          mark(s, f.X(X), f.Y(Y), q.c, q.c > 0 ? p.c1 : p.c2);
        });
        c.restore();
      };
      upd();
    }
  };

  // ==================================================================
  // L7 – A small neural network learning live
  // ==================================================================
  VIZ.nn = {
    lecture: 7, title: "A neural network learning live",
    desc: "A 2-H-1 network with tanh units, trained by SGD (backpropagation). The background is its output: blue = +1, orange = −1.",
    look: ["With 1 hidden unit the network is basically a perceptron: one straight boundary.",
      "With a few hidden units it can solve XOR and the circle: the hidden layer builds the features Φ by itself.",
      "Watch E_in decrease noisily over the epochs (SGD)."],
    build(root) {
      let H = 4, task = "xor", data = [], net = null, epoch = 0, hist = [];
      function make() {
        data = [];
        for (let i = 0; i < 120; i++) {
          const x = Math.random() * 2 - 1, y = Math.random() * 2 - 1;
          let c;
          if (task === "xor") { if (Math.abs(x) < 0.08 || Math.abs(y) < 0.08) { i--; continue; } c = x * y > 0 ? 1 : -1; }
          else if (task === "circle") { const d = Math.hypot(x, y); if (Math.abs(d - 0.6) < 0.06) { i--; continue; } c = d < 0.6 ? 1 : -1; }
          else { const d = y - 0.4 * x; if (Math.abs(d) < 0.08) { i--; continue; } c = d > 0 ? 1 : -1; }
          data.push({ x, y, c });
        }
      }
      function init() {
        const rw = () => randn() * 0.8;
        net = { W1: Array.from({ length: H }, () => [rw(), rw(), rw()]), W2: Array.from({ length: H + 1 }, rw) };
        epoch = 0; hist = [];
      }
      function forward(x, y) {
        const h = net.W1.map(w => Math.tanh(w[0] + w[1] * x + w[2] * y));
        let s = net.W2[0];
        for (let j = 0; j < H; j++) s += net.W2[j + 1] * h[j];
        return { h, out: Math.tanh(s) };
      }
      function sgdStep(d, eta) {
        const { h, out } = forward(d.x, d.y);
        const dOut = 2 * (out - d.c) * (1 - out * out);          // δ at the output
        const g2 = [dOut].concat(h.map(v => dOut * v));
        for (let j = 0; j < H; j++) {
          const dh = dOut * net.W2[j + 1] * (1 - h[j] * h[j]);    // backpropagated δ
          net.W1[j][0] -= eta * dh; net.W1[j][1] -= eta * dh * d.x; net.W1[j][2] -= eta * dh * d.y;
        }
        for (let j = 0; j <= H; j++) net.W2[j] -= eta * g2[j];
      }
      const Ein = () => data.reduce((a, d) => a + Math.pow(forward(d.x, d.y).out - d.c, 2), 0) / data.length;
      const miscl = () => data.filter(d => (forward(d.x, d.y).out >= 0 ? 1 : -1) !== d.c).length;
      make(); init();
      const s = stage(root, 0.85);
      const s2 = stage(root, 0.22);
      const ctl = controls(root);
      const run = looper(s, () => {
        for (let k = 0; k < 4; k++) { for (let i = 0; i < data.length; i++) sgdStep(data[Math.floor(Math.random() * data.length)], 0.06); epoch++; }
        hist.push(Ein());
        s.draw(); s2.draw(); upd();
        return epoch < 4000;
      });
      run.onStop = () => { tb.textContent = "Train"; };
      const tb = btn(ctl, "Train", () => { if (run.running) { run.stop(); tb.textContent = "Train"; } else { run.start(); tb.textContent = "Pause"; } }, true);
      btn(ctl, "Reset weights", () => { init(); s.draw(); s2.draw(); upd(); });
      seg(ctl, [["xor", "XOR"], ["circle", "Circle"], ["line", "Line"]], task, v => { task = v; make(); init(); s.draw(); s2.draw(); upd(); });
      slider(ctl, "hidden units H", 1, 10, 1, H, v => v, v => { H = v; init(); s.draw(); s2.draw(); upd(); });
      const ro = readout(root);
      function upd() {
        ro.set(`Epochs: <b class="tnum">${epoch}</b> &nbsp;·&nbsp; E_in (squared) = <b class="tnum">${fmt(Ein(), 3)}</b> &nbsp;·&nbsp; misclassified: <b class="tnum">${miscl()}</b> / ${data.length} &nbsp;·&nbsp; weights: ${3 * H + H + 1}`);
      }
      s.draw = function () {
        const p = P(); s.clear();
        const f = frame(s, [-1, 1], [-1, 1], { equal: true });
        const c1 = rgb(p.c1), c2 = rgb(p.c2);
        heat(s, f, 48, 48, (x, y) => { const o = forward(x, y).out; const col = o >= 0 ? c1 : c2; return [col[0], col[1], col[2], Math.round(70 * Math.min(1, Math.abs(o)) + 10)]; });
        axes(s, f, { xl: "x₁", yl: "x₂", nx: 4, ny: 4, grid: false });
        clip(s, f);
        data.forEach(d => mark(s, f.X(d.x), f.Y(d.y), d.c, d.c > 0 ? p.c1 : p.c2, 4));
        s.ctx.restore();
      };
      s2.draw = function () {
        const p = P(); s2.clear();
        const f = frame(s2, [0, Math.max(10, hist.length)], [0, Math.max(0.5, ...hist.slice(0, 5), 0.1)], { pad: { t: 6, b: 20, l: 40 } });
        axes(s2, f, { ny: 2, nx: 3, xfmt: v => String(v * 4), xl: "epochs" });
        clip(s2, f);
        const c = s2.ctx;
        c.strokeStyle = p.c3; c.lineWidth = 2; c.beginPath();
        hist.forEach((v, i) => (i ? c.lineTo(f.X(i), f.Y(v)) : c.moveTo(f.X(i), f.Y(v))));
        c.stroke();
        c.restore();
        label(s2, "E_in", f.pad.l + 6, f.pad.t + 8, p.c3, "left", "middle", true);
      };
      upd();
    }
  };

  // ==================================================================
  // L8 – k-nearest neighbours
  // ==================================================================
  VIZ.knn = {
    lecture: 8, title: "k-NN: tap to classify",
    desc: "The background shows the class k-NN predicts everywhere. Tap anywhere to see the k nearest neighbours of that point vote.",
    look: ["k = 1: a jagged boundary that follows every noisy point (E_in = 0, high variance).",
      "Larger k: smoother boundary. k close to N: everything becomes the majority class (high bias).",
      "The LOO error is cheap for k-NN (no training): use it to choose k. Compare with ⌊√N⌋."],
    build(root) {
      let k = 1, data = [], q = null;
      function make() {
        data = [];
        for (let i = 0; i < 35; i++) data.push({ x: -0.35 + randn() * 0.42, y: 0.2 + randn() * 0.42, c: 1 });
        for (let i = 0; i < 35; i++) data.push({ x: 0.35 + randn() * 0.42, y: -0.2 + randn() * 0.42, c: -1 });
        q = null;
      }
      make();
      function predict(x, y, skip) {
        const d = [];
        for (let i = 0; i < data.length; i++) if (i !== skip) d.push([Math.pow(data[i].x - x, 2) + Math.pow(data[i].y - y, 2), i]);
        d.sort((a, b) => a[0] - b[0]);
        const nn = d.slice(0, k);
        const vote = nn.reduce((a, [, i]) => a + data[i].c, 0);
        return { c: vote >= 0 ? 1 : -1, nn, vote };
      }
      const s = stage(root, 0.85);
      const ctl = controls(root);
      btn(ctl, "New data", () => { make(); s.draw(); upd(); }, true);
      slider(ctl, "k (neighbours)", 1, 69, 2, k, v => v + (v === 1 ? " (1-NN)" : v >= 69 ? " (≈ N)" : ""), v => { k = v; s.draw(); upd(); });
      const ro = readout(root);
      let f = null;
      s.onTap((px, py) => { if (!f) return; q = { x: f.IX(px), y: f.IY(py) }; s.draw(); upd(); });
      function upd() {
        let loo = 0, ein = 0;
        data.forEach((d, i) => { if (predict(d.x, d.y, i).c !== d.c) loo++; if (predict(d.x, d.y, -1).c !== d.c) ein++; });
        let txt = `k = ${k} &nbsp;·&nbsp; E_in = <b class="tnum">${fmt(ein / data.length)}</b> &nbsp;·&nbsp; LOO error = <b class="tnum">${fmt(loo / data.length)}</b> &nbsp;·&nbsp; rule of thumb ⌊√N⌋ = ${Math.floor(Math.sqrt(data.length))}`;
        if (q) { const r = predict(q.x, q.y, -1); const plus = r.nn.filter(([, i]) => data[i].c > 0).length; txt += `<br>Tapped point: ${plus} ● vs ${k - plus} × ⇒ predicted <b>${r.c > 0 ? "● (+1)" : "× (−1)"}</b>`; }
        else txt += "<br>Tap the plot to classify a new point.";
        ro.set(txt);
      }
      s.draw = function () {
        const p = P(); s.clear();
        f = frame(s, [-1.4, 1.4], [-1.4, 1.4], { equal: true });
        const c1 = rgb(p.c1), c2 = rgb(p.c2);
        heat(s, f, 70, 70, (x, y) => { const r = predict(x, y, -1); const col = r.c > 0 ? c1 : c2; return [col[0], col[1], col[2], 40 + Math.round(30 * Math.abs(r.vote) / k)]; });
        axes(s, f, { xl: "x₁", yl: "x₂", nx: 4, ny: 4, grid: false });
        clip(s, f);
        const c = s.ctx;
        data.forEach(d => mark(s, f.X(d.x), f.Y(d.y), d.c, d.c > 0 ? p.c1 : p.c2, 4));
        if (q) {
          const r = predict(q.x, q.y, -1);
          const rad = Math.sqrt(r.nn[r.nn.length - 1][0]);
          c.strokeStyle = p.ink; c.lineWidth = 1.5; c.setLineDash([4, 4]);
          c.beginPath(); c.arc(f.X(q.x), f.Y(q.y), rad * f.pw / (f.x1 - f.x0), 0, 7); c.stroke(); c.setLineDash([]);
          r.nn.forEach(([, i]) => { c.strokeStyle = alpha(p.ink.startsWith("#") ? p.ink : "#000000", 0.35); c.beginPath(); c.moveTo(f.X(q.x), f.Y(q.y)); c.lineTo(f.X(data[i].x), f.Y(data[i].y)); c.stroke(); });
          c.fillStyle = p.ink; c.strokeStyle = p.surface; c.lineWidth = 2;
          c.beginPath(); c.arc(f.X(q.x), f.Y(q.y), 7, 0, 7); c.fill(); c.stroke();
        }
        c.restore();
      };
      upd();
    }
  };

  // ==================================================================
  // L8 – Threshold, confusion matrix, precision and recall
  // ==================================================================
  VIZ.threshold = {
    lecture: 8, title: "Threshold, precision and recall",
    desc: "Classifier scores for real positives and real negatives (areas ∝ how many there are). Everything right of the threshold is predicted positive.",
    look: ["Move the threshold right: fewer false positives (precision ↑) but more missed positives (recall ↓).",
      "Make positives rare: accuracy stays high even for a useless classifier, while precision and recall reveal the problem.",
      "F1 combines precision and recall into one number."],
    build(root) {
      let t = 0, share = 0.3;
      const s = stage(root, 0.55);
      const ctl = controls(root);
      slider(ctl, "threshold", -3, 3, 0.05, t, v => v.toFixed(2), v => { t = v; s.draw(); upd(); });
      slider(ctl, "share of positives", 0.02, 0.5, 0.01, share, v => Math.round(100 * v) + "%", v => { share = v; s.draw(); upd(); });
      const ro = readout(root);
      legend(root, [["--c1", "real positives", "box"], ["--c2", "real negatives", "box"]]);
      function upd() {
        const n = 1000, np = n * share, nn = n - np;
        const a = np * (1 - Phi(t - 1)), c = np - a, b = nn * (1 - Phi(t + 1)), d = nn - b;
        const prec = a / Math.max(1e-9, a + b), rec = a / np, f1 = 2 * prec * rec / Math.max(1e-9, prec + rec), acc = (a + d) / n;
        ro.set(`<table class="cm"><tr><th></th><th>real +1</th><th>real −1</th></tr>
          <tr><th>pred +1</th><td>a = ${Math.round(a)} <small>TP</small></td><td>b = ${Math.round(b)} <small>FP</small></td></tr>
          <tr><th>pred −1</th><td>c = ${Math.round(c)} <small>FN</small></td><td>d = ${Math.round(d)} <small>TN</small></td></tr></table>
          precision a/(a+b) = <b class="tnum">${fmt(prec)}</b> · recall a/(a+c) = <b class="tnum">${fmt(rec)}</b> · F1 = <b class="tnum">${fmt(f1)}</b> · accuracy = <b class="tnum">${fmt(acc)}</b> <span class="muted">(per 1000 cases)</span>`);
      }
      s.draw = function () {
        const p = P(); s.clear();
        const pdf = (x, m) => Math.exp(-0.5 * (x - m) * (x - m)) / Math.sqrt(2 * Math.PI);
        const f = frame(s, [-4, 4], [0, 0.42], { pad: { l: 14 } });
        axes(s, f, { xl: "score s", ylabels: false, ny: 2 });
        clip(s, f);
        const c = s.ctx;
        const area = (m, w, col, from, to, a) => {
          c.fillStyle = alpha(col, a);
          c.beginPath(); c.moveTo(f.X(from), f.Y(0));
          for (let k = 0; k <= 80; k++) { const x = from + (to - from) * k / 80; c.lineTo(f.X(x), f.Y(w * pdf(x, m))); }
          c.lineTo(f.X(to), f.Y(0)); c.closePath(); c.fill();
        };
        area(-1, 1 - share, p.c2, -4, t, 0.12); area(-1, 1 - share, p.c2, t, 4, 0.45);
        area(1, share, p.c1, -4, t, 0.12); area(1, share, p.c1, t, 4, 0.45);
        curve(s, f, x => (1 - share) * pdf(x, -1), p.c2, 2);
        curve(s, f, x => share * pdf(x, 1), p.c1, 2);
        c.strokeStyle = p.ink; c.lineWidth = 2;
        c.beginPath(); c.moveTo(f.X(t), f.pad.t); c.lineTo(f.X(t), f.pad.t + f.ph); c.stroke();
        c.restore();
        label(s, "predict −1 ←", f.X(t) - 6, f.pad.t + 12, p.muted, "right");
        label(s, "→ predict +1", f.X(t) + 6, f.pad.t + 12, p.muted, "left");
      };
      upd();
    }
  };

  // ==================================================================
  // L9 – k-means (Lloyd's algorithm)
  // ==================================================================
  VIZ.kmeans = {
    lecture: 9, title: "k-means: Lloyd's algorithm",
    desc: "Alternate the two steps: assign each point to the nearest centroid, then move each centroid to the mean of its points.",
    look: ["Each step can only decrease E_in (bottom chart): the algorithm always stops.",
      "Random initialization can end in a bad local minimum (try a few times); farthest-point is much more reliable.",
      "The E_in vs k chart (right numbers) shows the knee at the true number of groups."],
    build(root) {
      let K = 3, init = "random", data = [], cents = [], assign = [], phase = "assign", hist = [], steps = 0;
      function make() {
        data = [];
        const centers = [[-0.6, 0.5], [0.6, 0.6], [0, -0.6]];
        centers.forEach(([cx, cy]) => { for (let i = 0; i < 30; i++) data.push({ x: cx + randn() * 0.2, y: cy + randn() * 0.2 }); });
      }
      function d2(a, b) { return (a.x - b.x) ** 2 + (a.y - b.y) ** 2; }
      function initC() {
        if (init === "random") { cents = []; for (let j = 0; j < K; j++) cents.push({ x: Math.random() * 2 - 1, y: Math.random() * 2 - 1 }); }
        else {
          cents = [Object.assign({}, data[Math.floor(Math.random() * data.length)])];
          while (cents.length < K) {
            let best = null, bd = -1;
            data.forEach(p0 => { const m = Math.min(...cents.map(c0 => d2(p0, c0))); if (m > bd) { bd = m; best = p0; } });
            cents.push(Object.assign({}, best));
          }
        }
        assign = data.map(() => -1); phase = "assign"; steps = 0; hist = [];
      }
      const Ein = () => data.reduce((a, p0, i) => a + (assign[i] >= 0 ? d2(p0, cents[assign[i]]) : 0), 0);
      function step() {
        if (phase === "assign") {
          const old = assign.join();
          assign = data.map(p0 => { let b = 0; cents.forEach((c0, j) => { if (d2(p0, c0) < d2(p0, cents[b])) b = j; }); return b; });
          hist.push(Ein());
          phase = "update";
          steps++;
          return old !== assign.join();
        }
        cents = cents.map((c0, j) => { const m = data.filter((_, i) => assign[i] === j); return m.length ? { x: m.reduce((a, q) => a + q.x, 0) / m.length, y: m.reduce((a, q) => a + q.y, 0) / m.length } : c0; });
        hist.push(Ein());
        phase = "assign";
        return true;
      }
      // best E_in for k = 1..8 (farthest-point init + Lloyd), for the knee chart
      function kneeCurve() {
        const out = [];
        for (let k = 1; k <= 8; k++) {
          let best = Infinity;
          for (let rep = 0; rep < 4; rep++) {
            let C = [Object.assign({}, data[Math.floor(Math.random() * data.length)])];
            while (C.length < k) { let bp = null, bd = -1; data.forEach(p0 => { const m = Math.min(...C.map(c0 => d2(p0, c0))); if (m > bd) { bd = m; bp = p0; } }); C.push(Object.assign({}, bp)); }
            let A = [];
            for (let it = 0; it < 30; it++) {
              A = data.map(p0 => { let b = 0; C.forEach((c0, j) => { if (d2(p0, c0) < d2(p0, C[b])) b = j; }); return b; });
              C = C.map((c0, j) => { const m = data.filter((_, i) => A[i] === j); return m.length ? { x: m.reduce((a, q) => a + q.x, 0) / m.length, y: m.reduce((a, q) => a + q.y, 0) / m.length } : c0; });
            }
            best = Math.min(best, data.reduce((a, p0, i) => a + d2(p0, C[A[i]]), 0));
          }
          out.push(best);
        }
        return out;
      }
      make(); initC();
      let knee = kneeCurve();
      const s = stage(root, 0.8);
      const s2 = stage(root, 0.3);
      const ctl = controls(root);
      let last = 0, stable = 0;
      const run = looper(s, () => {
        const now = performance.now();
        if (now - last < 600) return true;
        last = now;
        const changed = step();
        if (!changed) stable++; else stable = 0;
        s.draw(); s2.draw(); upd();
        return stable < 1 && steps < 50;
      });
      run.onStop = () => { playB.textContent = "Play"; upd(); };
      btn(ctl, "One step", () => { run.stop(); playB.textContent = "Play"; step(); s.draw(); s2.draw(); upd(); }, true);
      const playB = btn(ctl, "Play", () => { if (run.running) { run.stop(); playB.textContent = "Play"; } else { stable = 0; run.start(); playB.textContent = "Pause"; } });
      btn(ctl, "Restart", () => { run.stop(); playB.textContent = "Play"; initC(); s.draw(); s2.draw(); upd(); });
      btn(ctl, "New data", () => { run.stop(); playB.textContent = "Play"; make(); initC(); knee = kneeCurve(); s.draw(); s2.draw(); upd(); });
      seg(ctl, [["random", "Random init"], ["far", "Farthest-point init"]], init, v => { init = v; run.stop(); playB.textContent = "Play"; initC(); s.draw(); s2.draw(); upd(); });
      slider(ctl, "k (clusters)", 1, 4, 1, K, v => v, v => { K = v; run.stop(); playB.textContent = "Play"; initC(); s.draw(); s2.draw(); upd(); });
      const ro = readout(root);
      function upd() {
        ro.set(`Next: <b>${phase === "assign" ? "(2) assign points to the nearest centroid" : "(3) move centroids to the mean"}</b> &nbsp;·&nbsp; E_in = <b class="tnum">${hist.length ? fmt(hist[hist.length - 1], 3) : "–"}</b><br>Best E_in for k = 1…8: ${knee.map((v, i) => `<span class="tnum">${i + 1}: ${fmt(v, 1)}</span>`).join(" · ")} → knee at k = 3`);
      }
      const cols = p => [p.c1, p.c2, p.c3, p.c4];
      s.draw = function () {
        const p = P(); s.clear();
        const f = frame(s, [-1.3, 1.3], [-1.3, 1.3], { equal: true });
        if (assign[0] >= 0) {
          const cs = cols(p).map(rgb);
          heat(s, f, 60, 60, (x, y) => { let b = 0; cents.forEach((c0, j) => { if ((x - c0.x) ** 2 + (y - c0.y) ** 2 < (x - cents[b].x) ** 2 + (y - cents[b].y) ** 2) b = j; }); const c0 = cs[b]; return [c0[0], c0[1], c0[2], 28]; });
        }
        axes(s, f, { xl: "x₁", yl: "x₂", nx: 4, ny: 4, grid: false });
        clip(s, f);
        const c = s.ctx;
        data.forEach((q, i) => mark(s, f.X(q.x), f.Y(q.y), 0, assign[i] >= 0 ? cols(p)[assign[i]] : p.muted, 5));
        cents.forEach((c0, j) => {
          const x = f.X(c0.x), y = f.Y(c0.y);
          c.fillStyle = cols(p)[j]; c.strokeStyle = p.ink; c.lineWidth = 2;
          c.beginPath(); c.moveTo(x, y - 10); c.lineTo(x + 10, y); c.lineTo(x, y + 10); c.lineTo(x - 10, y); c.closePath(); c.fill(); c.stroke();
        });
        c.restore();
      };
      s2.draw = function () {
        const p = P(); s2.clear();
        const f = frame(s2, [0, Math.max(6, hist.length)], [0, Math.max(1, ...hist) * 1.1], { pad: { t: 6, b: 20 } });
        axes(s2, f, { ny: 2, nx: 3, xl: "half-steps" });
        clip(s2, f);
        const c = s2.ctx;
        c.strokeStyle = p.c3; c.lineWidth = 2; c.beginPath();
        hist.forEach((v, i) => (i ? c.lineTo(f.X(i + 1), f.Y(v)) : c.moveTo(f.X(i + 1), f.Y(v)))); c.stroke();
        hist.forEach((v, i) => { c.fillStyle = p.c3; c.beginPath(); c.arc(f.X(i + 1), f.Y(v), 3, 0, 7); c.fill(); });
        c.restore();
        label(s2, "E_in never increases", f.pad.l + 6, f.pad.t + 8, p.c3, "left", "middle", true);
      };
      upd();
    }
  };

  // ==================================================================
  // L9 – Hierarchical clustering and the dendrogram
  // ==================================================================
  VIZ.dendro = {
    lecture: 9, title: "Hierarchical clustering: the dendrogram",
    desc: "Points {1, 2, 4, 7, 8, 10}. At each step merge the two clusters whose merge increases E_in the least.",
    look: ["Each merge is drawn at the height of the total E_in reached (square-root scale).",
      "The last merge costs 54: a long vertical gap means “cut here”, so k = 2.",
      "Move the cut line to read the clustering for any k without re-running anything."],
    build(root) {
      const xs = [1, 2, 4, 7, 8, 10];
      let merges = [], clusters = [], cut = 7;
      function reset() { clusters = xs.map((x, i) => ({ ids: [i], pos: x, h: 0 })); merges = []; }
      const sse = ids => { const m = ids.reduce((a, i) => a + xs[i], 0) / ids.length; return ids.reduce((a, i) => a + (xs[i] - m) ** 2, 0); };
      const total = () => clusters.reduce((a, c0) => a + sse(c0.ids), 0);
      function step() {
        if (clusters.length < 2) return false;
        let bi = 0, bj = 1, bc = Infinity;
        for (let i = 0; i < clusters.length; i++) for (let j = i + 1; j < clusters.length; j++) {
          const cost = sse(clusters[i].ids.concat(clusters[j].ids)) - sse(clusters[i].ids) - sse(clusters[j].ids);
          if (cost < bc) { bc = cost; bi = i; bj = j; }
        }
        const A = clusters[bi], B = clusters[bj];
        const ids = A.ids.concat(B.ids);
        clusters.splice(bj, 1); clusters.splice(bi, 1);
        const C = { ids, pos: (A.pos + B.pos) / 2, h: 0 };
        clusters.push(C);
        C.h = total();
        merges.push({ A, B, C, cost: bc });
        return true;
      }
      reset();
      const s = stage(root, 0.7);
      const ctl = controls(root);
      btn(ctl, "Merge next", () => { step(); s.draw(); upd(); }, true);
      btn(ctl, "All merges", () => { while (step()); s.draw(); upd(); });
      btn(ctl, "Restart", () => { reset(); s.draw(); upd(); });
      slider(ctl, "cut height √E_in", 0.3, 8, 0.1, cut, v => "E_in = " + (v * v).toFixed(1), v => { cut = v; s.draw(); upd(); });
      const ro = readout(root);
      function groupsAtCut() {
        // undo merges above the cut
        let groups = xs.map((_, i) => [i]);
        merges.forEach(m => { if (Math.sqrt(m.C.h) <= cut) { groups = groups.filter(g => !m.C.ids.some(i => g.includes(i))); groups.push(m.C.ids.slice()); } });
        return groups;
      }
      function upd() {
        const last = merges[merges.length - 1];
        const g = groupsAtCut();
        ro.set(`${last ? `Last merge cost <b class="tnum">+${fmt(last.cost)}</b> → total E_in = <b class="tnum">${fmt(last.C.h)}</b>` : "Press <b>Merge next</b>."}<br>Cut gives k = <b>${g.length}</b>: ${g.map(gr => "{" + gr.map(i => xs[i]).sort((a, b) => a - b).join(", ") + "}").join(" ")}`);
      }
      s.draw = function () {
        const p = P(); s.clear();
        const f = frame(s, [0, 11], [-1.2, 8.3], { pad: { l: 40 } });
        axes(s, f, { xticks: xs, ylabels: true, yticks: [0, 1, 2, 4, 6, 8], yfmt: v => (v * v).toFixed(v < 2 ? 0 : 0), yl: "E_in (√ scale)", grid: false });
        clip(s, f);
        const c = s.ctx;
        const g = groupsAtCut();
        const colOf = i => [p.c1, p.c2, p.c3, p.c4][g.findIndex(gr => gr.includes(i)) % 4];
        // dendrogram: each merge draws a ⊓ shape at height √E_in
        const H = node => Math.sqrt(node.h);
        c.strokeStyle = p.ink; c.lineWidth = 2;
        merges.forEach(m => {
          const y = f.Y(H(m.C));
          c.beginPath();
          c.moveTo(f.X(m.A.pos), f.Y(H(m.A))); c.lineTo(f.X(m.A.pos), y);
          c.lineTo(f.X(m.B.pos), y); c.lineTo(f.X(m.B.pos), f.Y(H(m.B)));
          c.stroke();
        });
        xs.forEach((x, i) => { mark(s, f.X(x), f.Y(0), 0, colOf(i), 7); });
        c.strokeStyle = p.c2; c.setLineDash([6, 4]); c.lineWidth = 1.5;
        c.beginPath(); c.moveTo(f.pad.l, f.Y(cut)); c.lineTo(f.pad.l + f.pw, f.Y(cut)); c.stroke(); c.setLineDash([]);
        c.restore();
        label(s, "cut", f.pad.l + f.pw - 4, f.Y(cut) - 9, p.c2, "right", "middle", true);
      };
      upd();
    }
  };

  // ==================================================================
  // L10 – Centering, normalization, whitening
  // ==================================================================
  VIZ.preprocess = {
    lecture: 10, title: "Centering, normalization, whitening",
    desc: "The same 200 measurements after each pre-processing step. The ellipse shows their covariance.",
    look: ["Centering moves the cloud to the origin (subtract x̄).",
      "Normalization divides each feature by its σⱼ: equal spread along the axes, but the cloud is still tilted (correlated).",
      "Whitening z = A^(−1/2)x makes the covariance the identity: a round cloud, no correlation left."],
    build(root) {
      const raw = [];
      for (let i = 0; i < 200; i++) { const a = randn(), b = randn(); raw.push([3 + 2 * a, 2 + 1.2 * a + 0.6 * b]); }
      const mean = v => [v.reduce((s0, q) => s0 + q[0], 0) / v.length, v.reduce((s0, q) => s0 + q[1], 0) / v.length];
      const cov = v => { const m = mean(v); let a = 0, b = 0, c = 0; v.forEach(q => { const x = q[0] - m[0], y = q[1] - m[1]; a += x * x; b += x * y; c += y * y; }); return [a / v.length, b / v.length, c / v.length]; };
      const m0 = mean(raw);
      const cen = raw.map(q => [q[0] - m0[0], q[1] - m0[1]]);
      const C = cov(cen);
      const nor = cen.map(q => [q[0] / Math.sqrt(C[0]), q[1] / Math.sqrt(C[2])]);
      // A^(-1/2) via eigen-decomposition of the 2×2 covariance
      const tr = C[0] + C[2], det = C[0] * C[2] - C[1] * C[1];
      const l1 = tr / 2 + Math.sqrt(tr * tr / 4 - det), l2 = tr / 2 - Math.sqrt(tr * tr / 4 - det);
      const th = 0.5 * Math.atan2(2 * C[1], C[0] - C[2]);
      const q1 = [Math.cos(th), Math.sin(th)], q2 = [-Math.sin(th), Math.cos(th)];
      const whi = cen.map(q => {
        const u = (q[0] * q1[0] + q[1] * q1[1]) / Math.sqrt(l1), v = (q[0] * q2[0] + q[1] * q2[1]) / Math.sqrt(l2);
        return [u * q1[0] + v * q2[0], u * q1[1] + v * q2[1]];
      });
      const sets = { raw, cen, nor, whi };
      let cur = raw.map(q => q.slice()), from = cur, to = raw, t = 1, which = "raw";
      const s = stage(root, 0.8);
      const ctl = controls(root);
      const run = looper(s, () => { t = Math.min(1, t + 0.04); const e = t * t * (3 - 2 * t); cur = from.map((q, i) => [q[0] + (to[i][0] - q[0]) * e, q[1] + (to[i][1] - q[1]) * e]); s.draw(); return t < 1; });
      seg(ctl, [["raw", "Raw"], ["cen", "Centered"], ["nor", "Normalized"], ["whi", "Whitened"]], which, v => { which = v; from = cur.map(q => q.slice()); to = sets[v]; t = 0; run.start(); upd(); });
      const ro = readout(root);
      function upd() {
        const v = sets[which], m = mean(v), c = cov(v);
        ro.set(`mean = (<b class="tnum">${fmt(m[0])}</b>, <b class="tnum">${fmt(m[1])}</b>) &nbsp;·&nbsp; covariance A = [<span class="tnum">${fmt(c[0])} ${fmt(c[1])}; ${fmt(c[1])} ${fmt(c[2])}</span>]` +
          (which === "whi" ? " = <b>I</b>" : which === "nor" ? " (diagonal = 1, still correlated)" : ""));
      }
      s.draw = function () {
        const p = P(); s.clear();
        const f = frame(s, [-5, 9], [-4, 7], { equal: true });
        axes(s, f, { xl: "x₁", yl: "x₂" });
        clip(s, f);
        const c = s.ctx;
        c.strokeStyle = p.muted; c.lineWidth = 1;
        c.beginPath(); c.moveTo(f.X(0), f.pad.t); c.lineTo(f.X(0), f.pad.t + f.ph); c.moveTo(f.pad.l, f.Y(0)); c.lineTo(f.pad.l + f.pw, f.Y(0)); c.stroke();
        cur.forEach(q => mark(s, f.X(q[0]), f.Y(q[1]), 0, alpha(p.c1, 0.75), 3.2));
        // 2σ covariance ellipse of the current cloud
        const m = mean(cur), cv = cov(cur);
        const trc = cv[0] + cv[2], dt = cv[0] * cv[2] - cv[1] * cv[1];
        const e1 = trc / 2 + Math.sqrt(Math.max(0, trc * trc / 4 - dt)), e2 = trc / 2 - Math.sqrt(Math.max(0, trc * trc / 4 - dt));
        const ang = 0.5 * Math.atan2(2 * cv[1], cv[0] - cv[2]);
        c.strokeStyle = p.c2; c.lineWidth = 2;
        c.beginPath();
        for (let k = 0; k <= 80; k++) {
          const a = 2 * Math.PI * k / 80;
          const u = 2 * Math.sqrt(Math.max(e1, 0)) * Math.cos(a), v = 2 * Math.sqrt(Math.max(e2, 0)) * Math.sin(a);
          const x = m[0] + u * Math.cos(ang) - v * Math.sin(ang), y = m[1] + u * Math.sin(ang) + v * Math.cos(ang);
          k ? c.lineTo(f.X(x), f.Y(y)) : c.moveTo(f.X(x), f.Y(y));
        }
        c.stroke();
        c.restore();
      };
      upd();
    }
  };

  // ==================================================================
  // L10 – PCA: find the direction of largest variance
  // ==================================================================
  VIZ.pca = {
    lecture: 10, title: "PCA: the direction of largest variance",
    desc: "Rotate the direction v and project every point on it (z = vᵀx). PCA picks the v with the largest variance vᵀAv.",
    look: ["The projections (dots on the line) spread most along the first principal component v₁.",
      "Variance along v + variance lost across v = total: maximizing one minimizes the other (the reconstruction error).",
      "The bottom curve is vᵀAv as a function of the angle: its maximum is v₁, its minimum v₂ (90° away)."],
    build(root) {
      const pts = [];
      for (let i = 0; i < 120; i++) { const a = randn(), b = randn(); pts.push([1.6 * a + 0.2 * b, 0.9 * a + 0.55 * b]); }
      const m = [pts.reduce((s0, q) => s0 + q[0], 0) / pts.length, pts.reduce((s0, q) => s0 + q[1], 0) / pts.length];
      pts.forEach(q => { q[0] -= m[0]; q[1] -= m[1]; });
      let A = [0, 0, 0];
      pts.forEach(q => { A[0] += q[0] * q[0]; A[1] += q[0] * q[1]; A[2] += q[1] * q[1]; });
      A = A.map(v => v / pts.length);
      const varAt = t => { const c = Math.cos(t), s0 = Math.sin(t); return c * c * A[0] + 2 * c * s0 * A[1] + s0 * s0 * A[2]; };
      const tStar = 0.5 * Math.atan2(2 * A[1], A[0] - A[2]);
      const total = A[0] + A[2];
      let deg = 100;
      const s = stage(root, 0.75);
      const s2 = stage(root, 0.3);
      const ctl = controls(root);
      let target = null;
      const run = looper(s, () => { const d = target - deg; deg += Math.sign(d) * Math.min(Math.abs(d), 3); sl.value = deg; s.draw(); s2.draw(); upd(); return Math.abs(target - deg) > 0.01; });
      btn(ctl, "Snap to 1st principal component", () => { let t = (tStar * 180 / Math.PI + 360) % 180; target = t; run.start(); }, true);
      const sl = slider(ctl, "direction of v (degrees)", 0, 180, 1, deg, v => v.toFixed(0) + "°", v => { deg = v; s.draw(); s2.draw(); upd(); });
      const ro = readout(root);
      function upd() {
        const v = varAt(deg * Math.PI / 180);
        ro.set(`Variance along v: vᵀAv = <b class="tnum">${fmt(v)}</b> (${Math.round(100 * v / total)}% of the total ${fmt(total)}) &nbsp;·&nbsp; lost across v: <b class="tnum">${fmt(total - v)}</b><br>Maximum at ${((tStar * 180 / Math.PI + 360) % 180).toFixed(0)}° (v₁): γ₁²/(γ₁² + γ₂²) = ${Math.round(100 * varAt(tStar) / total)}% kept with k = 1`);
      }
      s.draw = function () {
        const p = P(); s.clear();
        const f = frame(s, [-4.2, 4.2], [-3, 3], { equal: true });
        axes(s, f, { xl: "x₁", yl: "x₂" });
        clip(s, f);
        const c = s.ctx;
        const t = deg * Math.PI / 180, v = [Math.cos(t), Math.sin(t)];
        c.strokeStyle = p.c3; c.lineWidth = 2.5;
        c.beginPath(); c.moveTo(f.X(-6 * v[0]), f.Y(-6 * v[1])); c.lineTo(f.X(6 * v[0]), f.Y(6 * v[1])); c.stroke();
        pts.forEach(q => {
          const z = q[0] * v[0] + q[1] * v[1];
          c.strokeStyle = alpha(p.c2, 0.35); c.lineWidth = 1;
          c.beginPath(); c.moveTo(f.X(q[0]), f.Y(q[1])); c.lineTo(f.X(z * v[0]), f.Y(z * v[1])); c.stroke();
        });
        pts.forEach(q => mark(s, f.X(q[0]), f.Y(q[1]), 0, alpha(p.c1, 0.8), 3.2));
        pts.forEach(q => { const z = q[0] * v[0] + q[1] * v[1]; c.fillStyle = p.c3; c.beginPath(); c.arc(f.X(z * v[0]), f.Y(z * v[1]), 2.2, 0, 7); c.fill(); });
        c.restore();
        label(s, "v", f.X(3.2 * v[0]) + 8, f.Y(3.2 * v[1]) - 8, p.c3, "left", "middle", true);
      };
      s2.draw = function () {
        const p = P(); s2.clear();
        const f = frame(s2, [0, 180], [0, Math.max(...Array.from({ length: 37 }, (_, i) => varAt(i * 5 * Math.PI / 180))) * 1.1], { pad: { t: 6, b: 22 } });
        axes(s2, f, { xticks: [0, 45, 90, 135, 180], ny: 2, xl: "angle of v" });
        clip(s2, f);
        curve(s2, f, d => varAt(d * Math.PI / 180), p.c3, 2);
        const c = s2.ctx;
        c.fillStyle = p.c2; c.strokeStyle = p.surface; c.lineWidth = 2;
        c.beginPath(); c.arc(f.X(deg), f.Y(varAt(deg * Math.PI / 180)), 5, 0, 7); c.fill(); c.stroke();
        c.restore();
        label(s2, "vᵀAv", f.pad.l + 6, f.pad.t + 8, p.c3, "left", "middle", true);
      };
      upd();
    }
  };

  // ------------------------------------------------------------------
  // Which visuals go with which slides
  // ------------------------------------------------------------------
  window.SLIDE_VIZ = {
    0: { 37: ["supunsup"], 42: ["supunsup"], 43: ["supunsup"], 45: ["supunsup"] },
    1: { 8: ["hoeffding"], 9: ["hoeffding"], 10: ["hoeffding"], 13: ["hoeffding"], 14: ["multibin"], 15: ["multibin"] },
    2: { 5: ["ols"], 6: ["ols"], 7: ["ols"], 8: ["ols"], 9: ["bowl"], 10: ["bowl"], 12: ["ols", "bowl"], 13: ["bowl"] },
    3: { 3: ["pla"], 4: ["pla"], 5: ["pla"], 6: ["pla"], 7: ["pla"], 8: ["pla"], 9: ["pla"] },
    4: { 4: ["sigmoid"], 5: ["sigmoid"], 6: ["sigmoid"], 7: ["sigmoid"], 10: ["errfun"], 11: ["sigmoid", "errfun"], 12: ["sigmoid"], 13: ["gd"], 14: ["gd"], 15: ["gd"], 16: ["gd"], 17: ["gd"], 18: ["gd"], 19: ["gd"], 20: ["gd"], 21: ["gd"] },
    5: { 1: ["biasvar"], 3: ["biasvar"], 4: ["biasvar"], 7: ["biasvar"], 8: ["biasvar"], 9: ["polyfit"], 10: ["curves"], 11: ["curves"] },
    6: { 1: ["polyfit"], 2: ["polyfit"], 4: ["polyfit"], 5: ["polyfit"], 6: ["polyfit"], 7: ["cv"], 8: ["cv"], 9: ["cv"], 10: ["cv"], 11: ["cv"], 12: ["cv"], 13: ["cv"], 14: ["cv", "polyfit"] },
    7: { 2: ["transform"], 3: ["transform"], 4: ["transform"], 7: ["nn"], 10: ["nn"], 11: ["nn"], 12: ["nn"], 13: ["nn"], 14: ["nn"], 15: ["nn"], 16: ["nn"] },
    8: { 1: ["knn"], 3: ["knn"], 5: ["knn"], 6: ["knn"], 7: ["knn"], 8: ["knn"], 9: ["knn"], 10: ["threshold"], 11: ["threshold"], 12: ["threshold"] },
    9: { 2: ["supunsup"], 4: ["kmeans"], 5: ["kmeans"], 6: ["kmeans"], 7: ["kmeans"], 8: ["kmeans"], 9: ["dendro"] },
    10: { 4: ["preprocess"], 5: ["preprocess"], 6: ["preprocess"], 7: ["ols"], 9: ["pca"], 10: ["pca"], 11: ["pca"], 12: ["pca"], 15: ["pca"] }
  };

  // Mount a visual inside a container: title, description, canvas, controls, "what to look at".
  window.mountViz = function (id, container, compact) {
    const v = VIZ[id];
    if (!v) return;
    const box = el("section", "viz card");
    box.append(el("div", "viz-head", `<span class="eyebrow">Interactive · Lecture ${v.lecture}</span><h3>${v.title}</h3><p class="small muted">${v.desc}</p>`));
    const body = el("div", "viz-body");
    box.append(body);
    try { v.build(body); } catch (e) { body.append(el("p", "small muted", "This visual could not start: " + e.message)); }
    const look = el("details", "viz-look");
    if (!compact) look.open = true;
    look.innerHTML = `<summary>What to look at</summary><ul>${v.look.map(x => `<li>${x}</li>`).join("")}</ul>`;
    box.append(look);
    container.append(box);
    return box;
  };
})();
