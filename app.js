// ======================================================================
// SLAS Study Notebook – app logic.
// Course configuration first. To add a course later: add an entry with
// its own question bank (.md), quiz file and exam page images.
// ======================================================================
const COURSE = {
  key: "slas",
  tag: "SLAS",
  name: "Statistical Learning for Automation Systems",
  bank: "bank.md",
  exams: [
    { date: "7 Nov 2022", iso: "2022-11-07", pages: 3, topics: ["Classification", "Neural networks", "Model selection", "Clustering"] },
    { date: "6 Feb 2023", iso: "2023-02-06", pages: 4, topics: ["k-nearest neighbors", "Neural networks", "Model selection", "Clustering"] },
    { date: "3 Jul 2023", iso: "2023-07-03", pages: 4, topics: ["Model selection", "Linear classification", "Nonparametric classification", "Stochastic processes"] },
    { date: "7 Sep 2023", iso: "2023-09-07", pages: 4, topics: ["Linear regression", "Neural networks", "Nonparametric classification", "Write your own question"] },
    { date: "12 Nov 2024", iso: "2024-11-12", pages: 9, official: true },
    { date: "9 Jan 2025", iso: "2025-01-09", pages: 8, official: true },
    { date: "13 Feb 2025", iso: "2025-02-13", pages: 8, official: true },
    { date: "21 Jan 2026", iso: "2026-01-21", pages: 8, official: true },
    { date: "11 Feb 2026", iso: "2026-02-11", pages: 5, official: true }
  ]
};

// ---------------------------------------------------------------------
// Saved progress (only on this device). Every access is wrapped because
// storage can be blocked (private mode, previews).
// ---------------------------------------------------------------------
const STORE_KEY = "study-" + COURSE.key;
const DEFAULTS = { cards: {}, quizBest: {}, saved: {}, cover: false, theme: "auto", size: "m", level: "mixed", lastTab: "learn", lastLecture: null };
let progress = JSON.parse(JSON.stringify(DEFAULTS));
try {
  const stored = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
  if (stored) progress = Object.assign(progress, stored);
} catch (e) { /* start fresh */ }
function save() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(progress)); } catch (e) { /* ignore */ }
}

// Leitner boxes: 0 = new/again, 1..5 = learnt more and more. Box >= 3 counts as "known".
const INTERVAL_HOURS = [0, 4, 24, 72, 168, 336];
function cardState(id) { return progress.cards[id] || { box: 0, due: 0, seen: false }; }
function isKnown(id) { return cardState(id).box >= 3; }
function isDue(id) { return cardState(id).due <= Date.now(); }
function grade(id, how) {
  const s = cardState(id);
  let box = s.box;
  if (how === "again") box = 0;
  if (how === "hard") box = Math.max(1, box);
  if (how === "easy") box = Math.min(5, Math.max(3, box + 1));
  progress.cards[id] = { box: box, due: Date.now() + INTERVAL_HOURS[box] * 3600e3, seen: true };
  save();
}
function isSaved(id) { return !!progress.saved[id]; }
function toggleSaved(id) {
  if (progress.saved[id]) delete progress.saved[id]; else progress.saved[id] = 1;
  save();
}

// ---------------------------------------------------------------------
// Appearance settings: theme and text size.
// ---------------------------------------------------------------------
const SIZES = { s: "15px", m: "16px", l: "18px", xl: "20px" };
function applySettings() {
  const root = document.documentElement;
  if (progress.theme === "auto") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", progress.theme);
  root.style.setProperty("--fs", SIZES[progress.size] || SIZES.m);
}
applySettings();

// ---------------------------------------------------------------------
// Parse the question bank markdown into lectures, questions, boards and
// past-exam topics. Structure (from the file):
//   # Blackboard sheet  -> ## Board N – title
//   # Part I            -> ## Lecture N – title -> ### Key formulas, ### LN.Qk – question
//   # Part II           -> ## A. Topic -> ### EX.A1 – question
// ---------------------------------------------------------------------
const data = { boards: [], lectures: [], topics: [], byId: {} };

function parseBank(text) {
  const lines = text.split("\n");
  let part = "";            // current "# " heading
  let group = null;         // current "## " object
  let item = null;          // current "### " object (question or formulas)
  let inFence = false;

  for (const line of lines) {
    if (line.startsWith("```")) inFence = !inFence;
    if (!inFence && /^# /.test(line)) {
      part = line.slice(2).trim();
      group = null; item = null;
      continue;
    }
    if (!inFence && /^## /.test(line)) {
      const title = line.slice(3).trim();
      item = null;
      if (part.startsWith("Blackboard")) {
        group = { title: title, body: "" };
        data.boards.push(group);
      } else if (part.startsWith("Part I –")) {
        const m = title.match(/^Lecture (\d+) – (.*)$/);
        group = { id: m ? "L" + m[1] : "BP", num: m ? m[1] : "", title: m ? m[2] : title, formulas: "", questions: [] };
        data.lectures.push(group);
      } else if (part.startsWith("Part II")) {
        const m = title.match(/^([A-Z])\. (.*)$/);
        group = m ? { letter: m[1], title: m[2], questions: [] } : null;
        if (group) data.topics.push(group);
      } else {
        group = null;
      }
      continue;
    }
    if (!inFence && /^### /.test(line) && group && group.questions) {
      const title = line.slice(4).trim();
      if (title === "Key formulas") { item = { formulas: true }; continue; }
      const m = title.match(/^(\S+) – (.*)$/);
      if (!m) { item = null; continue; }
      item = { id: m[1], title: m[2].replace(/\s*★\s*$/, ""), star: /★\s*$/.test(m[2]), source: "", body: "", group: group };
      group.questions.push(item);
      data.byId[item.id] = item;
      continue;
    }
    // Plain content line
    if (item && item.formulas) { group.formulas += line + "\n"; continue; }
    if (item) {
      if (!item.source && !item.body.trim() && /^\*(Source|Exams?):/.test(line)) {
        item.source = line.replace(/^\*|\*$/g, "");
        continue;
      }
      item.body += line + "\n";
      continue;
    }
    if (group && part.startsWith("Blackboard")) group.body += line + "\n";
  }
  // Drop trailing "---" separators
  for (const q of Object.values(data.byId)) q.body = q.body.replace(/\n-{3,}\s*$/, "").trim();
  for (const l of data.lectures) l.formulas = l.formulas.replace(/\n-{3,}\s*$/, "").trim();
  for (const b of data.boards) b.body = b.body.trim();
}

function allPart1() { return data.lectures.flatMap(l => l.questions); }
function allPart2() { return data.topics.flatMap(t => t.questions); }
function lectureById(id) { return data.lectures.find(l => l.id === id); }
function lectureLabel(l) { return l.id === "BP" ? "Big picture" : "Lecture " + l.num; }
function lectureShort(l) { return l.id === "BP" ? "BP" : "L" + l.num; }
function lectureProgress(l) {
  const k = l.questions.filter(q => isKnown(q.id)).length;
  return l.questions.length ? Math.round(100 * k / l.questions.length) : 0;
}
function quizPool(lecture, level) {
  return window.QUIZ.filter(x =>
    (lecture === "all" || x.l === lecture) &&
    (level === "mixed" || (level === "hard" ? x.h : !x.h)));
}

// Markdown -> HTML, with every table wrapped so it can scroll sideways on a phone.
function md(text) {
  const html = marked.parse(text || "", { mangle: false, headerIds: false });
  return html.replace(/<table>/g, '<div class="tbl"><table>').replace(/<\/table>/g, "</table></div>");
}
function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

// Small icons used in buttons
const ICON = {
  bookmark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 3h12v18l-6-4-6 4z"/></svg>',
  bookmarkOn: '<svg viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 3h12v18l-6-4-6 4z"/></svg>',
  cards: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="6" width="14" height="14" rx="2"/><path d="M7 6V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-2"/></svg>',
  quiz: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14"/><path d="M12 17.5v.01"/></svg>',
  flame: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3c1 3 5 5 5 10a5 5 0 0 1-10 0c0-2 1-3.5 2-4.5 0 2 1 3 2 3 0-3-1-5 1-8.5z"/></svg>',
  book: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 19V5"/></svg>',
  slides: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="12" rx="1.5"/><path d="M12 16v4M8 20h8"/><path d="M7 8h6M7 11h9"/></svg>',
  arrow: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>'
};

// ---------------------------------------------------------------------
// Navigation: a stack of screens per tab, linked to the browser/phone
// back button through the History API.
// ---------------------------------------------------------------------
const view = document.getElementById("view");
const titleEl = document.getElementById("title");
const backBtn = document.getElementById("backBtn");
let currentTab = "learn";
let stack = [];

function go(screen) {
  stack[stack.length - 1].scroll = window.scrollY;
  stack.push(screen);
  try { history.pushState({ depth: stack.length }, ""); } catch (e) { /* ignore */ }
  render();
}
function back() {
  if (stack.length > 1) history.back();
}
window.addEventListener("popstate", () => {
  if (stack.length > 1) { stack.pop(); render(); }
});
function switchTab(tab) {
  currentTab = tab;
  progress.lastTab = tab; save();
  document.querySelectorAll(".tab").forEach(t => t.setAttribute("aria-current", t.dataset.tab === tab ? "page" : "false"));
  // Drop any screens pushed on the previous tab
  const extra = stack.length - 1;
  stack = [{ name: tab }];
  if (extra > 0) { try { history.go(-extra); } catch (e) { /* ignore */ } }
  render();
}
function render() {
  const screen = stack[stack.length - 1];
  backBtn.hidden = stack.length <= 1;
  view.innerHTML = "";
  SCREENS[screen.name](screen);
  window.scrollTo(0, screen.scroll || 0);
}
backBtn.addEventListener("click", back);
document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => switchTab(t.dataset.tab)));
document.getElementById("settingsBtn").addEventListener("click", openSettings);

function setTitle(t) { titleEl.textContent = t; }

// Brief message at the bottom of the screen
let toastTimer = null;
function toast(msg) {
  let t = document.getElementById("toast");
  if (!t) { t = el(`<div id="toast" class="toast" role="status"></div>`); document.body.append(t); }
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 1800);
}

// A question rendered as a collapsible card (used in Learn, search and Exams).
function questionCard(q, open) {
  const s = cardState(q.id);
  const cls = isKnown(q.id) ? "known" : (s.seen ? "learning" : "");
  const node = el(`
    <article class="card q" id="q-${esc(q.id)}">
      <div class="qhead-row">
        <button class="qhead" aria-expanded="${open ? "true" : "false"}">
          <span class="state ${cls}" aria-label="${cls === "known" ? "Known" : cls === "learning" ? "Learning" : "Not studied yet"}"></span>
          <span class="grow">
            <span class="qid">${esc(q.id)}</span> ${q.star ? '<span class="badge-star">★ must-know</span>' : ""}
            <span class="qtitle">${esc(q.title)}</span>
          </span>
        </button>
        <button class="iconbtn save-btn ${isSaved(q.id) ? "on" : ""}" aria-label="Save question" aria-pressed="${isSaved(q.id)}">${isSaved(q.id) ? ICON.bookmarkOn : ICON.bookmark}</button>
      </div>
      <div class="qbody" ${open ? "" : "hidden"}>
        ${q.source ? `<p class="src">${esc(q.source)}</p>` : ""}
        <div class="md">${md(q.body)}</div>
        <div class="mark-row">
          <button class="btn ghost" data-mark="again">Review again</button>
          <button class="btn" data-mark="easy">I know it</button>
        </div>
      </div>
    </article>`);
  const head = node.querySelector(".qhead");
  const body = node.querySelector(".qbody");
  head.addEventListener("click", () => {
    body.hidden = !body.hidden;
    head.setAttribute("aria-expanded", String(!body.hidden));
  });
  const sb = node.querySelector(".save-btn");
  sb.addEventListener("click", () => {
    toggleSaved(q.id);
    const on = isSaved(q.id);
    sb.classList.toggle("on", on);
    sb.setAttribute("aria-pressed", String(on));
    sb.innerHTML = on ? ICON.bookmarkOn : ICON.bookmark;
    toast(on ? "Saved for later" : "Removed from saved");
  });
  node.querySelectorAll("[data-mark]").forEach(b => b.addEventListener("click", () => {
    grade(q.id, b.dataset.mark);
    node.querySelector(".state").className = "state " + (isKnown(q.id) ? "known" : "learning");
    body.hidden = true;
    head.setAttribute("aria-expanded", "false");
    toast(b.dataset.mark === "easy" ? "Marked as known" : "Will come back in the cards");
  }));
  return node;
}

function openQuestion(id) {
  if (data.byId[id]) go({ name: "question", id: id });
}

// A big tappable row: icon, title, subtitle, chevron.
function actionRow(icon, title, sub, onClick, tone) {
  const b = el(`<button class="card action ${tone || ""}">
    <span class="aicon">${icon}</span>
    <span class="grow"><b>${esc(title)}</b><span class="small muted">${esc(sub)}</span></span>
    <span class="chev">${ICON.arrow}</span>
  </button>`);
  b.addEventListener("click", onClick);
  return b;
}

// ======================================================================
// Screens
// ======================================================================
const SCREENS = {};

// ---------------- Learn (home) ----------------
SCREENS.learn = function () {
  setTitle("Learn");
  const p1 = allPart1();
  const known = p1.filter(q => isKnown(q.id)).length;
  const stars = p1.filter(q => q.star);
  const starsKnown = stars.filter(q => isKnown(q.id)).length;
  const due = p1.filter(q => cardState(q.id).seen && isDue(q.id)).length;

  view.append(el(`<div><div class="eyebrow">Prof. Formentin · Lectures 1–10</div>
    <h2 class="section">${esc(COURSE.name)}</h2></div>`));

  view.append(el(`<div class="summary">
    <div class="card stat"><b>${known}<span class="muted small">/${p1.length}</span></b><span>questions known</span></div>
    <div class="card stat"><b class="star">${starsKnown}<span class="muted small">/${stars.length}</span></b><span>★ must-know</span></div>
    <div class="card stat"><b>${due}</b><span>cards to review</span></div>
  </div>`));

  // "Up next": the most useful things to do now
  const next = el(`<section class="stackgap"><div class="eyebrow">Up next</div></section>`);
  if (due > 0) {
    next.append(actionRow(ICON.cards, `Review ${due} card${due > 1 ? "s" : ""}`, "Spaced repetition: these are due now", () => { switchTab("cards"); startDeck("review"); }, "accent"));
  }
  const last = progress.lastLecture && lectureById(progress.lastLecture);
  if (last) {
    next.append(actionRow(ICON.book, `Continue ${lectureLabel(last)}`, `${last.title} · ${lectureProgress(last)}% known`, () => go({ name: "lecture", id: last.id })));
  }
  const unfinished = data.lectures.filter(l => l.id !== "BP" && lectureProgress(l) < 100);
  if (unfinished.length) {
    const weakest = unfinished.slice().sort((a, b) => lectureProgress(a) - lectureProgress(b))[0];
    if (!last || weakest.id !== last.id) {
      next.append(actionRow(ICON.flame, `Weakest: ${lectureLabel(weakest)}`, `${weakest.title} · ${lectureProgress(weakest)}% known`, () => go({ name: "lecture", id: weakest.id })));
    }
  }
  if (starsKnown < stars.length) {
    next.append(actionRow(ICON.cards, "★ Must-know flashcards", `${stars.length - starsKnown} still to learn`, () => { switchTab("cards"); startDeck("star"); }));
  }
  view.append(next);

  const search = el(`<input class="search" id="search" type="search" placeholder="Search all questions (e.g. Hoeffding, ridge, PCA)" autocomplete="off">`);
  view.append(search);
  const results = el(`<div class="qlist" hidden></div>`);
  view.append(results);

  const list = el(`<section class="stackgap"><div class="eyebrow">Lectures</div></section>`);
  for (const l of data.lectures) {
    const pct = lectureProgress(l);
    const nStar = l.questions.filter(q => q.star).length;
    const b = el(`<button class="card lect">
      <span class="num">${lectureShort(l)}</span>
      <span style="min-width:0">
        <div class="t">${esc(l.title)}</div>
        <div class="meta"><span class="tnum">${l.questions.length} Q</span>${nStar ? `<span class="star tnum">★ ${nStar}</span>` : ""}<span class="bar"><i style="width:${pct}%"></i></span></div>
      </span>
      <span class="pct">${pct}%</span>
    </button>`);
    b.addEventListener("click", () => go({ name: "lecture", id: l.id }));
    list.append(b);
  }
  view.append(list);

  search.addEventListener("input", () => {
    const term = search.value.trim().toLowerCase();
    results.innerHTML = "";
    if (term.length < 2) { results.hidden = true; list.hidden = false; next.hidden = false; return; }
    const hits = allPart1().concat(allPart2()).filter(q => (q.title + " " + q.body).toLowerCase().includes(term)).slice(0, 40);
    results.hidden = false; list.hidden = true; next.hidden = true;
    results.append(el(`<div class="small muted">${hits.length === 40 ? "First 40" : hits.length} result${hits.length === 1 ? "" : "s"}</div>`));
    if (!hits.length) results.append(el(`<div class="empty">No questions match "${esc(search.value)}".</div>`));
    for (const q of hits) results.append(questionCard(q, false));
  });
};

SCREENS.lecture = function (screen) {
  const l = lectureById(screen.id);
  progress.lastLecture = l.id; save();
  setTitle(lectureLabel(l));
  const filter = screen.filter || "all";
  const lectNum = Number(l.num);
  const nBasic = l.id === "BP" ? 0 : quizPool(lectNum, "basic").length;
  const nHard = l.id === "BP" ? 0 : quizPool(lectNum, "hard").length;

  view.append(el(`<div><div class="eyebrow">${l.id === "BP" ? "Lectures 1–10" : "Lecture " + l.num} · ${lectureProgress(l)}% known</div>
    <h2 class="section">${esc(l.title)}</h2></div>`));

  // Quick actions for this lecture
  const actions = el(`<div class="quick"></div>`);
  const qa = (icon, label, sub, fn) => {
    const b = el(`<button class="card qbtn"><span class="aicon">${icon}</span><b>${label}</b><span class="small muted">${sub}</span></button>`);
    b.addEventListener("click", fn);
    actions.append(b);
  };
  const deckInfo = slideDeck(lectNum);
  if (l.id !== "BP" && deckInfo) qa(ICON.slides, "Slides", `${shownSlides(deckInfo).length} with notes`, () => { switchTab("slides"); go({ name: "deck", lecture: lectNum, i: 0 }); });
  qa(ICON.cards, "Flashcards", `${l.questions.length} cards`, () => { switchTab("cards"); startDeck("L:" + l.id, true); });
  if (nBasic) qa(ICON.quiz, "Quiz", `${nBasic} questions`, () => { switchTab("quiz"); startQuiz(lectNum, "basic"); });
  if (nHard) qa(ICON.flame, "Hard quiz", `${nHard} questions`, () => { switchTab("quiz"); startQuiz(lectNum, "hard"); });
  view.append(actions);

  const counts = {
    all: l.questions.length,
    star: l.questions.filter(q => q.star).length,
    todo: l.questions.filter(q => !isKnown(q.id)).length,
    saved: l.questions.filter(q => isSaved(q.id)).length
  };
  const filters = el(`<div class="chips">
    <button class="chip" aria-pressed="${filter === "all"}" data-f="all">All ${counts.all}</button>
    <button class="chip" aria-pressed="${filter === "star"}" data-f="star">★ Must-know ${counts.star}</button>
    <button class="chip" aria-pressed="${filter === "todo"}" data-f="todo">To learn ${counts.todo}</button>
    <button class="chip" aria-pressed="${filter === "saved"}" data-f="saved">Saved ${counts.saved}</button>
  </div>`);
  filters.querySelectorAll("[data-f]").forEach(b => b.addEventListener("click", () => {
    screen.filter = b.dataset.f;
    screen.scroll = window.scrollY;
    render();
  }));
  view.append(filters);

  if (l.formulas && filter === "all") {
    const f = el(`<details class="card formulas" ${screen.formulasClosed ? "" : "open"}><summary>Key formulas</summary><div class="md">${md(l.formulas)}</div></details>`);
    f.addEventListener("toggle", () => { screen.formulasClosed = !f.open; });
    view.append(f);
  }

  const shown = l.questions.filter(q =>
    filter === "all" || (filter === "star" && q.star) || (filter === "todo" && !isKnown(q.id)) || (filter === "saved" && isSaved(q.id)));
  const bar = el(`<div class="row small muted"><span class="grow">${shown.length} question${shown.length === 1 ? "" : "s"} · tap to see the answer</span>
    <button class="linkbtn small" id="expandAll">${screen.expanded ? "Collapse all" : "Expand all"}</button></div>`);
  bar.querySelector("#expandAll").addEventListener("click", () => {
    screen.expanded = !screen.expanded;
    screen.scroll = window.scrollY;
    render();
  });
  view.append(bar);
  const qs = el(`<div class="qlist"></div>`);
  for (const q of shown) qs.append(questionCard(q, !!screen.expanded));
  if (!shown.length) qs.append(el(`<div class="card empty">${filter === "saved" ? "Tap the bookmark on a question to save it here." : filter === "todo" ? "You know every question in this lecture. 🎉" : "Nothing here."}</div>`));
  view.append(qs);

  // Previous / next lecture
  const i = data.lectures.indexOf(l);
  const nav = el(`<div class="row"></div>`);
  if (i > 0) {
    const p = data.lectures[i - 1];
    const b = el(`<button class="btn ghost grow">‹ ${lectureShort(p)}</button>`);
    b.addEventListener("click", () => { stack[stack.length - 1] = { name: "lecture", id: p.id }; render(); window.scrollTo(0, 0); });
    nav.append(b);
  }
  if (i < data.lectures.length - 1) {
    const n = data.lectures[i + 1];
    const b = el(`<button class="btn ghost grow">${lectureShort(n)} ›</button>`);
    b.addEventListener("click", () => { stack[stack.length - 1] = { name: "lecture", id: n.id }; render(); window.scrollTo(0, 0); });
    nav.append(b);
  }
  view.append(nav);
};

SCREENS.question = function (screen) {
  const q = data.byId[screen.id];
  setTitle(q.id);
  view.append(questionCard(q, true));
  const groupName = q.group.num !== undefined
    ? (q.group.id === "BP" ? "Big picture" : "Lecture " + q.group.num + " – " + q.group.title)
    : q.group.letter + ". " + q.group.title;
  view.append(el(`<p class="muted small">From: ${esc(groupName)}</p>`));
};

// ---------------- Flashcards ----------------
let deck = null; // { key, queue: [ids], index, flipped, reviewed, total }

const DECKS = [["review", "Due now"], ["star", "★ Must-know"], ["all", "All lectures"], ["weak", "Weak spots"], ["saved", "Saved"], ["exam", "Past exam Qs"]];
function deckCards(key) {
  if (key === "review") return allPart1().filter(q => cardState(q.id).seen);
  if (key === "star") return allPart1().filter(q => q.star);
  if (key === "all") return allPart1();
  if (key === "exam") return allPart2();
  if (key === "weak") return allPart1().filter(q => cardState(q.id).seen && !isKnown(q.id));
  if (key === "saved") return allPart1().concat(allPart2()).filter(q => isSaved(q.id));
  if (key.startsWith("L:")) { const l = lectureById(key.slice(2)); return l ? l.questions : []; }
  return [];
}
function buildDeck(key, includeAll) {
  let cards = deckCards(key);
  if (!includeAll) cards = cards.filter(q => isDue(q.id));
  // New/again cards first, ★ before the others, random order within each group.
  const order = q => cardState(q.id).box * 2 + (q.star ? 0 : 1);
  cards = shuffle(cards).sort((a, b) => order(a) - order(b));
  return { key: key, queue: cards.map(q => q.id), index: 0, flipped: false, reviewed: 0, total: cards.length };
}
function startDeck(key, includeAll) {
  deck = buildDeck(key, includeAll);
  render();
}
function gradeCurrent(how) {
  const id = deck.queue[deck.index];
  grade(id, how);
  deck.reviewed++;
  // A card you didn't know comes back 4 cards later in the same session.
  if (how === "again") deck.queue.splice(Math.min(deck.queue.length, deck.index + 5), 0, id);
  deck.index++;
  deck.flipped = false;
  render();
}

SCREENS.cards = function () {
  setTitle("Flashcards");
  if (!deck) deck = buildDeck(allPart1().some(q => cardState(q.id).seen && isDue(q.id)) ? "review" : "star");
  const decks = DECKS.concat(data.lectures.filter(l => l.id !== "BP").map(l => ["L:" + l.id, "L" + l.num]));
  const picker = el(`<div class="chips scroll"></div>`);
  for (const [k, label] of decks) {
    const n = deckCards(k).filter(q => isDue(q.id)).length;
    const b = el(`<button class="chip" aria-pressed="${k === deck.key}">${esc(label)} <span class="muted tnum">${n}</span></button>`);
    b.addEventListener("click", () => startDeck(k));
    picker.append(b);
  }
  view.append(picker);
  // Keep the selected deck chip visible
  const sel = picker.querySelector('[aria-pressed="true"]');
  if (sel) requestAnimationFrame(() => { picker.scrollLeft = sel.offsetLeft - 16; });

  const id = deck.queue[deck.index];
  if (!id) {
    const allN = deckCards(deck.key).length;
    const box = el(`<div class="card pad empty">
      <p style="font-size:1.05rem;color:var(--ink);margin:0 0 6px"><b>${deck.total ? "Deck finished 🎉" : "Nothing due in this deck"}</b></p>
      <p style="margin:0 0 14px">${deck.reviewed ? `You reviewed ${deck.reviewed} cards. ` : ""}Cards come back after 4 hours to 2 weeks, depending on how well you knew them.</p>
      ${allN ? `<button class="btn" id="again">Study all ${allN} cards anyway</button>` : `<p class="small">${deck.key === "saved" ? "Save questions with the bookmark icon to build this deck." : "Start with the ★ Must-know deck."}</p>`}
    </div>`);
    const a = box.querySelector("#again");
    if (a) a.addEventListener("click", () => startDeck(deck.key, true));
    view.append(box);
    return;
  }
  const q = data.byId[id];
  const left = deck.queue.length - deck.index;
  const done = deck.index;
  view.append(el(`<div class="row small muted"><span class="grow tnum">${left} left · ${done} done</span><span class="qid">${esc(q.id)}</span></div>`));
  view.append(el(`<div class="bar"><i style="width:${deck.queue.length ? 100 * done / deck.queue.length : 0}%;background:var(--accent)"></i></div>`));
  const card = el(`<article class="card flash ${deck.flipped ? "flipped" : ""}">
    <div class="row"><span class="grow">${q.star ? '<span class="badge-star">★ must-know</span>' : ""}</span>
      <button class="iconbtn save-btn ${isSaved(id) ? "on" : ""}" aria-label="Save question" aria-pressed="${isSaved(id)}">${isSaved(id) ? ICON.bookmarkOn : ICON.bookmark}</button></div>
    <div class="front">${esc(q.title)}</div>
    <div class="answer" ${deck.flipped ? "" : "hidden"}>
      ${q.source ? `<p class="src small muted" style="margin:0 0 8px;font-style:italic">${esc(q.source)}</p>` : ""}
      <div class="md">${md(q.body)}</div>
    </div>
    <div class="hint" ${deck.flipped ? "hidden" : ""}>Answer out loud, then tap to check</div>
  </article>`);
  card.querySelector(".save-btn").addEventListener("click", e => {
    e.stopPropagation();
    toggleSaved(id);
    toast(isSaved(id) ? "Saved for later" : "Removed from saved");
    render();
  });
  card.addEventListener("click", () => { if (!deck.flipped) { deck.flipped = true; render(); } });
  // Swipe on a flipped card: right = knew it, left = didn't know
  let x0 = null, y0 = null;
  card.addEventListener("touchstart", e => { x0 = e.touches[0].clientX; y0 = e.touches[0].clientY; }, { passive: true });
  card.addEventListener("touchend", e => {
    if (x0 === null || !deck.flipped) return;
    const dx = e.changedTouches[0].clientX - x0, dy = e.changedTouches[0].clientY - y0;
    x0 = null;
    if (Math.abs(dx) > 80 && Math.abs(dx) > 2 * Math.abs(dy)) gradeCurrent(dx > 0 ? "easy" : "again");
  });
  view.append(card);

  if (deck.flipped) {
    const g = el(`<div class="grade">
      <button class="again" data-g="again">Didn't know<small>see again soon</small></button>
      <button class="hard" data-g="hard">Partly<small>in 4 h</small></button>
      <button class="easy" data-g="easy">Knew it<small>in 1+ days</small></button>
    </div>`);
    g.querySelectorAll("[data-g]").forEach(b => b.addEventListener("click", () => gradeCurrent(b.dataset.g)));
    view.append(g);
    view.append(el(`<p class="small muted center">Tip: swipe the card right if you knew it, left if you didn't.</p>`));
  } else {
    const b = el(`<button class="btn">Show answer</button>`);
    b.addEventListener("click", () => { deck.flipped = true; render(); });
    view.append(b);
  }
};

// ---------------- Quiz ----------------
let quiz = null; // { lecture, level, items, index, answered, correct, wrong: [] }
const LEVELS = [["basic", "Basic"], ["hard", "Hard"], ["mixed", "Mixed"]];

function startQuiz(lecture, level) {
  level = level || progress.level;
  const pool = quizPool(lecture, level);
  const items = shuffle(pool).slice(0, 10).map(x => ({ src: x, order: shuffle(x.o.map((_, i) => i)) }));
  quiz = { lecture: lecture, level: level, items: items, index: 0, answered: null, correct: 0, wrong: [] };
  render();
}
function answerQuiz(oi) {
  if (quiz.answered != null) return;
  const x = quiz.items[quiz.index].src;
  quiz.answered = oi;
  if (oi === x.a) quiz.correct++; else quiz.wrong.push(x);
  render();
  const fb = document.querySelector(".feedback");
  if (fb) fb.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
function nextQuiz() {
  quiz.index++;
  quiz.answered = null;
  render();
  window.scrollTo(0, 0);
}

SCREENS.quiz = function () {
  setTitle("Quiz");
  if (!quiz) {
    view.append(el(`<div><div class="eyebrow">${window.QUIZ.length} questions · 10 per round</div>
      <h2 class="section">Pick a level and a lecture</h2></div>`));
    const seg = el(`<div class="seg" role="radiogroup" aria-label="Difficulty"></div>`);
    for (const [k, label] of LEVELS) {
      const b = el(`<button role="radio" aria-checked="${progress.level === k}">${label}</button>`);
      b.addEventListener("click", () => { progress.level = k; save(); render(); });
      seg.append(b);
    }
    view.append(seg);
    view.append(el(`<p class="lede">${progress.level === "hard"
      ? "Hard: calculations, multi-step reasoning and exam traps."
      : progress.level === "basic" ? "Basic: definitions and key results." : "Mixed: basic and hard questions together."} Every answer links to the full explanation in your bank.</p>`));
    const grid = el(`<div class="lect-list"></div>`);
    const options = [["all", "Mixed: all lectures"]].concat(
      data.lectures.filter(l => l.id !== "BP").map(l => [Number(l.num), l.title]));
    for (const [key, label] of options) {
      const n = quizPool(key, progress.level).length;
      const best = progress.quizBest[key + "-" + progress.level];
      const b = el(`<button class="card lect">
        <span class="num">${key === "all" ? "∑" : "L" + key}</span>
        <span style="min-width:0"><div class="t">${esc(label)}</div><div class="meta"><span class="tnum">${n} questions</span></div></span>
        <span class="pct">${best != null ? "best " + best + "%" : ""}</span>
      </button>`);
      b.addEventListener("click", () => startQuiz(key, progress.level));
      grid.append(b);
    }
    view.append(grid);
    return;
  }

  // Finished
  if (quiz.index >= quiz.items.length) {
    const pct = Math.round(100 * quiz.correct / quiz.items.length);
    const bestKey = quiz.lecture + "-" + quiz.level;
    const prev = progress.quizBest[bestKey];
    if (prev == null || pct > prev) { progress.quizBest[bestKey] = pct; save(); }
    view.append(el(`<div class="card pad center">
      <div class="eyebrow">${quiz.lecture === "all" ? "All lectures" : "Lecture " + quiz.lecture} · ${quiz.level}</div>
      <div class="score-big">${quiz.correct}/${quiz.items.length}</div>
      <p class="muted" style="margin:6px 0 0">${pct >= 80 ? (quiz.level === "hard" ? "Excellent: exam-ready on this topic." : "Strong. Try the hard level next.") : pct >= 50 ? "Good base. Review the ones you missed below." : "Go back to this lecture in Learn, then retry."}</p>
    </div>`));
    if (quiz.wrong.length) {
      view.append(el(`<h2 class="section">To review</h2>`));
      const list = el(`<div class="qlist"></div>`);
      for (const w of quiz.wrong) {
        const r = data.byId[w.ref];
        const node = el(`<div class="card pad">
          <div style="font-weight:600">${esc(w.q)}</div>
          <p class="small" style="margin:6px 0"><span style="color:var(--good)">✓ ${esc(w.o[w.a])}</span></p>
          <p class="small muted" style="margin:0 0 8px">${esc(w.e)}</p>
          ${r ? `<button class="linkbtn small">Read ${esc(w.ref)} in full</button>` : ""}
        </div>`);
        const lb = node.querySelector(".linkbtn");
        if (lb) lb.addEventListener("click", () => openQuestion(w.ref));
        list.append(node);
      }
      view.append(list);
    }
    const row = el(`<div class="row"><button class="btn ghost grow" id="pick">Choose another</button><button class="btn grow" id="retry">New round</button></div>`);
    row.querySelector("#pick").addEventListener("click", () => { quiz = null; render(); });
    row.querySelector("#retry").addEventListener("click", () => startQuiz(quiz.lecture, quiz.level));
    view.append(row);
    if (quiz.level !== "hard" && pct >= 80) {
      const h = el(`<button class="btn ghost">Try the hard level</button>`);
      h.addEventListener("click", () => { progress.level = "hard"; save(); startQuiz(quiz.lecture, "hard"); });
      view.append(h);
    }
    return;
  }

  const it = quiz.items[quiz.index];
  const x = it.src;
  const top = el(`<div class="row small muted"><span class="grow tnum">Question ${quiz.index + 1} of ${quiz.items.length} · ${quiz.correct} correct</span>
    ${x.h ? '<span class="badge-hard">HARD</span>' : ""}<span class="qid">L${x.l}</span><button class="linkbtn small" id="quit">Quit</button></div>`);
  top.querySelector("#quit").addEventListener("click", () => { quiz = null; render(); });
  view.append(top);
  view.append(el(`<div class="bar"><i style="width:${100 * quiz.index / quiz.items.length}%;background:var(--accent)"></i></div>`));
  view.append(el(`<div class="card pad"><div class="quiz-q">${esc(x.q)}</div></div>`));
  const opts = el(`<div class="qlist"></div>`);
  it.order.forEach((oi, pos) => {
    const answered = quiz.answered != null;
    let cls = "";
    if (answered && oi === x.a) cls = "right";
    if (answered && oi === quiz.answered && oi !== x.a) cls = "wrong";
    const b = el(`<button class="opt ${cls}" ${answered ? "disabled" : ""}><span class="letter">${"ABCD"[pos]}</span><span>${esc(x.o[oi])}</span></button>`);
    b.addEventListener("click", () => answerQuiz(oi));
    opts.append(b);
  });
  view.append(opts);

  if (quiz.answered != null) {
    const ok = quiz.answered === x.a;
    const fb = el(`<div class="feedback ${ok ? "ok" : "no"}">
      <b>${ok ? "Correct" : "Not quite"}</b>
      <span>${esc(x.e)}</span>
      ${data.byId[x.ref] ? `<div style="margin-top:8px"><button class="linkbtn small">Read ${esc(x.ref)} in full</button></div>` : ""}
    </div>`);
    const lb = fb.querySelector(".linkbtn");
    if (lb) lb.addEventListener("click", () => openQuestion(x.ref));
    view.append(fb);
    const next = el(`<button class="btn">${quiz.index + 1 < quiz.items.length ? "Next question" : "See results"}</button>`);
    next.addEventListener("click", nextQuiz);
    view.append(next);
  }
};

// Keyboard shortcuts (iPad with keyboard, laptop)
document.addEventListener("keydown", e => {
  if (e.target.closest && e.target.closest("input, textarea")) return;
  const screen = stack[stack.length - 1];
  if (!screen) return;
  if (screen.name === "cards" && deck && deck.queue[deck.index]) {
    if ((e.key === " " || e.key === "Enter") && !deck.flipped) { e.preventDefault(); deck.flipped = true; render(); }
    else if (deck.flipped && e.key === "1") gradeCurrent("again");
    else if (deck.flipped && e.key === "2") gradeCurrent("hard");
    else if (deck.flipped && e.key === "3") gradeCurrent("easy");
  }
  if (screen.name === "deck") {
    const d = slideDeck(screen.lecture);
    if (e.key === "ArrowRight" && screen.i < shownSlides(d).length - 1) { screen.i++; screen.scroll = 0; render(); }
    if (e.key === "ArrowLeft" && screen.i > 0) { screen.i--; screen.scroll = 0; render(); }
  }
  if (screen.name === "quiz" && quiz && quiz.index < quiz.items.length) {
    const it = quiz.items[quiz.index];
    const pos = "1234".indexOf(e.key) >= 0 ? "1234".indexOf(e.key) : "abcd".indexOf(e.key.toLowerCase());
    if (quiz.answered == null && pos >= 0 && pos < it.order.length) answerQuiz(it.order[pos]);
    else if (quiz.answered != null && e.key === "Enter") nextQuiz();
  }
});

// ---------------- Exams ----------------
function examQuestions(date) {
  // Part II questions whose exam line mentions this date (e.g. "7 Nov 2022").
  return allPart2().filter(q => q.source.includes(date));
}
// Topics of an exam: listed by hand, or taken from the bank's past-question topics.
function examTopics(ex) {
  if (ex.topics) return ex.topics;
  const names = [];
  for (const q of examQuestions(ex.date)) if (!names.includes(q.group.title)) names.push(q.group.title);
  return names;
}

SCREENS.exams = function () {
  setTitle("Past exams");
  view.append(el(`<div><div class="eyebrow">Nov 2022 – Feb 2026</div><h2 class="section">Exam papers and solutions</h2></div>`));
  view.append(el(`<p class="lede">Try the paper first, then open the solutions from your question bank.</p>`));
  const list = el(`<div class="lect-list"></div>`);
  for (const ex of COURSE.exams.slice().reverse()) {
    const qs = examQuestions(ex.date);
    const k = qs.filter(q => isKnown(q.id)).length;
    const b = el(`<button class="card exam-card">
      <span style="min-width:0">
        <div class="d">${esc(ex.date)}</div>
        <div class="small muted">${esc(examTopics(ex).join(" · "))}</div>
      </span>
      <span style="display:flex;flex-direction:column;gap:4px;align-items:flex-end">
        <span class="pill">${ex.pages} pages</span>
        ${ex.official ? `<span class="pill">+ official sol.</span>` : ""}
        <span class="pill gray tnum">${k}/${qs.length} known</span>
      </span>
    </button>`);
    b.addEventListener("click", () => go({ name: "exam", date: ex.date }));
    list.append(b);
  }
  view.append(list);

  view.append(el(`<h2 class="section">Past questions by topic</h2>`));
  const tl = el(`<div class="lect-list"></div>`);
  for (const t of data.topics) {
    const k = t.questions.filter(q => isKnown(q.id)).length;
    const b = el(`<button class="card lect">
      <span class="num">${esc(t.letter)}</span>
      <span style="min-width:0"><div class="t">${esc(t.title)}</div><div class="meta"><span class="tnum">${t.questions.length} questions</span></div></span>
      <span class="pct tnum">${k}/${t.questions.length}</span>
    </button>`);
    b.addEventListener("click", () => go({ name: "topic", letter: t.letter }));
    tl.append(b);
  }
  view.append(tl);
};

SCREENS.exam = function (screen) {
  const ex = COURSE.exams.find(e => e.date === screen.date);
  setTitle(ex.date);
  const qs = examQuestions(ex.date);
  view.append(el(`<div><div class="eyebrow">Model Identification and Data Analysis – Part 1</div><h2 class="section">Exam of ${esc(ex.date)}</h2></div>`));

  // Two views: the paper itself, or the worked solutions
  const mode = screen.mode || "paper";
  const seg = el(`<div class="seg" role="tablist">
    <button role="tab" aria-checked="${mode === "paper"}" data-m="paper">Paper (${ex.pages} pages)</button>
    <button role="tab" aria-checked="${mode === "sol"}" data-m="sol">Solutions (${qs.length})</button>
  </div>`);
  seg.querySelectorAll("[data-m]").forEach(b => b.addEventListener("click", () => { screen.mode = b.dataset.m; screen.scroll = 0; render(); }));
  view.append(seg);

  if (mode === "paper") {
    view.append(el(`<p class="lede">Tap a page to open it full screen and zoom.${ex.official ? " The official solutions are printed after each exercise: cover them if you want to try first." : ""}</p>`));
    const pages = el(`<div class="pages"></div>`);
    for (let p = 1; p <= ex.pages; p++) {
      const src = `exams/${ex.iso}-${p}.jpg`;
      const img = el(`<img class="page-img" loading="lazy" alt="Exam ${esc(ex.date)}, page ${p}" src="${src}">`);
      img.addEventListener("click", () => openViewer(ex.pages, n => `exams/${ex.iso}-${n}.jpg`, p));
      pages.append(img);
    }
    view.append(pages);
    const s = el(`<button class="btn">Show the worked solutions</button>`);
    s.addEventListener("click", () => { screen.mode = "sol"; render(); window.scrollTo(0, 0); });
    view.append(s);
  } else {
    if (!qs.length) view.append(el(`<div class="card empty">This exam's exercises are on topics not covered yet in Lectures 1–10.</div>`));
    const list = el(`<div class="qlist"></div>`);
    for (const q of qs) list.append(questionCard(q, false));
    view.append(list);
  }
};

SCREENS.topic = function (screen) {
  const t = data.topics.find(x => x.letter === screen.letter);
  setTitle(t.letter + ". " + t.title);
  view.append(el(`<div><div class="eyebrow">Past exam questions</div><h2 class="section">${esc(t.title)}</h2></div>`));
  const list = el(`<div class="qlist"></div>`);
  for (const q of t.questions) list.append(questionCard(q, false));
  view.append(list);
};

// Full-screen page viewer with zoom and previous/next page
// Full-screen page viewer. pages = number of pages, src(p) = image url of page p.
function openViewer(pages, src, page) {
  let zoom = 1;
  const v = el(`<div class="viewer" role="dialog" aria-label="Page viewer">
    <div class="vbar">
      <button id="prevBtn" aria-label="Previous page">‹</button>
      <span id="pageNo" class="tnum"></span>
      <button id="nextBtn" aria-label="Next page">›</button>
      <span class="grow"></span>
      <button id="zoomBtn">Zoom 2×</button>
      <button id="closeBtn">Close</button>
    </div>
    <img alt="Page">
  </div>`);
  const img = v.querySelector("img");
  const show = () => {
    img.src = src(page);
    img.style.width = (zoom * 100) + "vw";
    v.querySelector("#pageNo").textContent = page + " / " + pages;
    v.querySelector("#prevBtn").disabled = page <= 1;
    v.querySelector("#nextBtn").disabled = page >= pages;
    v.scrollTo(0, 0);
  };
  v.querySelector("#prevBtn").addEventListener("click", () => { if (page > 1) { page--; show(); } });
  v.querySelector("#nextBtn").addEventListener("click", () => { if (page < pages) { page++; show(); } });
  v.querySelector("#zoomBtn").addEventListener("click", () => {
    zoom = zoom === 1 ? 2 : 1;
    v.querySelector("#zoomBtn").textContent = zoom === 1 ? "Zoom 2×" : "Fit width";
    show();
  });
  v.querySelector("#closeBtn").addEventListener("click", () => v.remove());
  document.body.append(v);
  show();
}


// ---------------- Slides with study notes ----------------
// Slide images are shown only where they are available (private version);
// the notes are always shown.
const HAS_SLIDE_IMAGES = !!window.SLIDE_IMAGES;

function slideDeck(lecture) { return (window.SLIDES || []).find(d => d.lecture === lecture); }
function shownSlides(deck) { const a = []; for (let i = 1; i <= deck.count; i++) a.push(i); return a; }
function slideImage(deck, n) { return `slides/${deck.file}-${String(n).padStart(2, "0")}.jpg`; }
// Note for slide n: an exact key, or a range key "a-b" containing n.
function slideNote(deck, n) {
  if (deck.notes[String(n)]) return { note: deck.notes[String(n)], from: n, to: n };
  for (const key of Object.keys(deck.notes)) {
    const m = key.match(/^(\d+)-(\d+)$/);
    if (m && n >= +m[1] && n <= +m[2]) return { note: deck.notes[key], from: +m[1], to: +m[2] };
  }
  return null;
}
// Slide numbers in the bank refer to the complete Lecture 4 PDF (28 slides):
// slides 21–22 there are slides 20–21 of the "part 1" PDF used here.
function bankToShownSlide(lecture, n) {
  if (lecture === 4 && n >= 21) return n - 1;
  if (lecture === 4 && n === 20) return null;
  return n;
}
// Map lecture -> slide -> [question ids], built from the "Source:" lines of the bank.
let slideRefs = null;
function buildSlideRefs() {
  slideRefs = {};
  for (const q of allPart1().concat(allPart2())) {
    const re = /Lecture (\d+), slides? ([0-9][0-9–\-, ]*(?:and [0-9]+)?)/g;
    let m;
    while ((m = re.exec(q.source))) {
      const lec = +m[1];
      const nums = [];
      const part = /(\d+)\s*[–-]\s*(\d+)|(\d+)/g;
      let p;
      while ((p = part.exec(m[2]))) {
        if (p[1]) { for (let k = +p[1]; k <= +p[2]; k++) nums.push(k); } else nums.push(+p[3]);
      }
      for (const n0 of nums) {
        const n = bankToShownSlide(lec, n0);
        if (n == null) continue;
        slideRefs[lec] = slideRefs[lec] || {};
        slideRefs[lec][n] = slideRefs[lec][n] || [];
        if (!slideRefs[lec][n].includes(q.id)) slideRefs[lec][n].push(q.id);
      }
    }
  }
}
function questionsForSlide(lecture, n) {
  if (!slideRefs) buildSlideRefs();
  return ((slideRefs[lecture] || {})[n] || []).map(id => data.byId[id]).filter(Boolean);
}

SCREENS.slides = function () {
  setTitle("Slides");
  view.append(el(`<div><div class="eyebrow">Lectures 0–10</div><h2 class="section">Slides with study notes</h2></div>`));
  view.append(el(`<p class="lede">${HAS_SLIDE_IMAGES ? "Every slide with" : "For every slide:"} the key idea, an explanation, exam tips and the related questions from your bank.${HAS_SLIDE_IMAGES ? "" : " The slide images themselves are only in your private version (they are copyrighted), so open the PDF next to these notes."}</p>`));
  const list = el(`<div class="lect-list"></div>`);
  for (const d of window.SLIDES || []) {
    const last = (progress.slidePos || {})[d.lecture];
    const nRefs = shownSlides(d).filter(n => questionsForSlide(d.lecture, n).length).length;
    const b = el(`<button class="card lect">
      <span class="num">L${d.lecture}</span>
      <span style="min-width:0"><div class="t">${esc(d.title)}</div>
        <div class="meta"><span class="tnum">${d.count} slides</span>${nRefs ? `<span class="tnum">${nRefs} linked to questions</span>` : ""}</div></span>
      <span class="pct">${last ? "slide " + last : ""}</span>
    </button>`);
    b.addEventListener("click", () => go({ name: "deck", lecture: d.lecture, i: last ? last - 1 : 0 }));
    list.append(b);
  }
  view.append(list);
};

SCREENS.deck = function (screen) {
  const d = slideDeck(screen.lecture);
  const nums = shownSlides(d);
  screen.i = Math.max(0, Math.min(nums.length - 1, screen.i || 0));
  const n = nums[screen.i];
  progress.slidePos = progress.slidePos || {};
  progress.slidePos[d.lecture] = n; save();
  setTitle(`L${d.lecture} · slide ${n}`);

  const goTo = i => { screen.i = i; screen.scroll = 0; render(); };

  // Header: title + jump menu
  const head = el(`<div class="stackgap">
    <div class="eyebrow">Lecture ${d.lecture} · ${esc(d.title)}</div>
    <select id="jump" class="jump" aria-label="Jump to slide"></select>
  </div>`);
  const sel = head.querySelector("#jump");
  nums.forEach((k, idx) => {
    const sn = slideNote(d, k);
    const o = document.createElement("option");
    o.value = idx;
    o.textContent = `${k}. ${sn ? sn.note.k : ""}`;
    if (idx === screen.i) o.selected = true;
    sel.append(o);
  });
  sel.addEventListener("change", () => goTo(+sel.value));
  view.append(head);
  view.append(el(`<div class="bar"><i style="width:${100 * (screen.i + 1) / nums.length}%;background:var(--accent)"></i></div>`));

  // Slide image (private version only)
  if (HAS_SLIDE_IMAGES) {
    const fig = el(`<figure class="slide-fig"><img class="slide-img" alt="Lecture ${d.lecture}, slide ${n}" src="${slideImage(d, n)}"></figure>`);
    const img = fig.querySelector("img");
    img.addEventListener("click", () => openViewer(nums.length, k => slideImage(d, nums[k - 1]), screen.i + 1));
    img.addEventListener("error", () => fig.remove());
    // Swipe left/right on the slide to change slide
    let x0 = null, y0 = null;
    fig.addEventListener("touchstart", e => { x0 = e.touches[0].clientX; y0 = e.touches[0].clientY; }, { passive: true });
    fig.addEventListener("touchend", e => {
      if (x0 === null) return;
      const dx = e.changedTouches[0].clientX - x0, dy = e.changedTouches[0].clientY - y0;
      x0 = null;
      if (Math.abs(dx) > 60 && Math.abs(dx) > 2 * Math.abs(dy)) {
        if (dx < 0 && screen.i < nums.length - 1) goTo(screen.i + 1);
        if (dx > 0 && screen.i > 0) goTo(screen.i - 1);
      }
    });
    view.append(fig);
  }

  // Navigation
  const nav = el(`<div class="row">
    <button class="btn ghost grow" id="prev" ${screen.i === 0 ? "disabled" : ""}>‹ Previous</button>
    <span class="small muted tnum">${screen.i + 1} / ${nums.length}</span>
    <button class="btn grow" id="next" ${screen.i === nums.length - 1 ? "disabled" : ""}>Next ›</button>
  </div>`);
  nav.querySelector("#prev").addEventListener("click", () => goTo(screen.i - 1));
  nav.querySelector("#next").addEventListener("click", () => goTo(screen.i + 1));
  view.append(nav);

  // Study note
  const sn = slideNote(d, n);
  if (sn) {
    const star = /★/.test(sn.note.k);
    const node = el(`<section class="card note">
      <div class="row"><span class="eyebrow grow">${sn.from === sn.to ? "Slide " + n : `Slides ${sn.from}–${sn.to}`}</span>${star ? '<span class="badge-star">★ must-know</span>' : ""}</div>
      <h3>${esc(sn.note.k.replace(/\s*★\s*$/, ""))}</h3>
      <div class="md">${md(sn.note.b)}</div>
      ${sn.note.x ? `<div class="tip"><b>Exam tip</b><div class="md">${md(sn.note.x)}</div></div>` : ""}
    </section>`);
    view.append(node);
  }

  // Related questions from the bank
  const qs = questionsForSlide(d.lecture, n);
  if (qs.length) {
    view.append(el(`<div class="eyebrow">Questions on this slide · ${qs.length}</div>`));
    const list = el(`<div class="qlist"></div>`);
    for (const q of qs) list.append(questionCard(q, false));
    view.append(list);
  }

  // Bottom navigation again, after reading
  if (sn || qs.length) {
    const nav2 = el(`<button class="btn" ${screen.i === nums.length - 1 ? "disabled" : ""}>${screen.i === nums.length - 1 ? "Last slide" : "Next slide ›"}</button>`);
    nav2.addEventListener("click", () => { goTo(screen.i + 1); window.scrollTo(0, 0); });
    view.append(nav2);
  }
};

// ---------------- Boards ----------------
SCREENS.boards = function () {
  setTitle("Blackboard");
  view.append(el(`<div><div class="eyebrow">${data.boards.length} boards</div><h2 class="section">Write it from memory</h2></div>`));
  view.append(el(`<p class="lede">Copy each board on paper, then switch on cover mode and rewrite it. Tap a covered board to check yourself. (D) marks derivations to redo on paper.</p>`));
  const tg = el(`<div class="card toggle"><span class="grow"><b>Cover mode</b><div class="small muted">Blur the boards until you tap them</div></span>
    <button class="switch" role="switch" id="coverSwitch" aria-checked="${progress.cover}" aria-label="Cover mode"></button></div>`);
  tg.querySelector("#coverSwitch").addEventListener("click", () => { progress.cover = !progress.cover; save(); render(); });
  view.append(tg);
  for (const b of data.boards) {
    const node = el(`<section class="card board ${progress.cover ? "covered" : ""}"><h3>${esc(b.title)}</h3><div class="md">${md(b.body)}</div></section>`);
    node.querySelector(".md").addEventListener("click", () => node.classList.remove("covered"));
    view.append(node);
  }
};

// ---------------- Settings sheet ----------------
function openSettings() {
  const sheet = el(`<div class="sheet-wrap" role="dialog" aria-label="Settings">
    <div class="sheet card">
      <div class="row"><h3 class="grow">Settings</h3><button class="iconbtn" id="closeSheet" aria-label="Close">✕</button></div>

      <div class="field"><div class="label">Text size</div>
        <div class="seg" id="sizeSeg">
          <button data-v="s" aria-checked="${progress.size === "s"}" style="font-size:0.85rem">A</button>
          <button data-v="m" aria-checked="${progress.size === "m"}">A</button>
          <button data-v="l" aria-checked="${progress.size === "l"}" style="font-size:1.15rem">A</button>
          <button data-v="xl" aria-checked="${progress.size === "xl"}" style="font-size:1.3rem">A</button>
        </div></div>

      <div class="field"><div class="label">Theme</div>
        <div class="seg" id="themeSeg">
          <button data-v="auto" aria-checked="${progress.theme === "auto"}">Auto</button>
          <button data-v="light" aria-checked="${progress.theme === "light"}">Light</button>
          <button data-v="dark" aria-checked="${progress.theme === "dark"}">Dark</button>
        </div></div>

      <div class="field"><div class="label">Move progress to another device</div>
        <p class="small muted" style="margin:0 0 8px">Copy the code here, then paste it on your iPad (or phone) in the same box and tap Restore.</p>
        <div class="row"><button class="btn ghost grow" id="copyCode">Copy progress code</button></div>
        <textarea id="codeBox" rows="3" placeholder="Paste a progress code here"></textarea>
        <div class="row"><button class="btn grow" id="restore">Restore from code</button></div>
        <p class="small" id="codeMsg" role="status"></p>
      </div>

      <div class="field"><div class="label">Reset</div>
        <button class="btn danger" id="reset">Reset all my progress</button>
      </div>
    </div>
  </div>`);
  const close = () => sheet.remove();
  sheet.addEventListener("click", e => { if (e.target === sheet) close(); });
  sheet.querySelector("#closeSheet").addEventListener("click", close);

  const segPick = (id, key) => sheet.querySelectorAll(`#${id} [data-v]`).forEach(b => b.addEventListener("click", () => {
    progress[key] = b.dataset.v; save(); applySettings();
    sheet.querySelectorAll(`#${id} [data-v]`).forEach(x => x.setAttribute("aria-checked", String(x === b)));
  }));
  segPick("sizeSeg", "size");
  segPick("themeSeg", "theme");

  const msg = sheet.querySelector("#codeMsg");
  const box = sheet.querySelector("#codeBox");
  sheet.querySelector("#copyCode").addEventListener("click", () => {
    const code = btoa(unescape(encodeURIComponent(JSON.stringify(progress))));
    box.value = code;
    const done = () => { msg.textContent = "Code copied. Paste it on the other device."; };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(code).then(done, () => { box.select(); msg.textContent = "Select the code above and copy it."; });
    } else { box.select(); msg.textContent = "Select the code above and copy it."; }
  });
  sheet.querySelector("#restore").addEventListener("click", () => {
    try {
      const obj = JSON.parse(decodeURIComponent(escape(atob(box.value.trim()))));
      if (!obj || typeof obj.cards !== "object") throw new Error("bad");
      progress = Object.assign(JSON.parse(JSON.stringify(DEFAULTS)), obj);
      save(); applySettings();
      close(); toast("Progress restored"); switchTab("learn");
    } catch (e) {
      msg.textContent = "That code is not valid. Copy it again from the other device.";
    }
  });
  const reset = sheet.querySelector("#reset");
  reset.addEventListener("click", () => {
    if (reset.dataset.confirm !== "1") {
      reset.dataset.confirm = "1";
      reset.textContent = "Tap again to confirm: this deletes everything";
      return;
    }
    const keep = { theme: progress.theme, size: progress.size };
    progress = Object.assign(JSON.parse(JSON.stringify(DEFAULTS)), keep);
    save(); deck = null; quiz = null;
    close(); toast("Progress reset"); switchTab("learn");
  });
  document.body.append(sheet);
}

// Offline support: the service worker keeps a copy of every file on the device.
if ("serviceWorker" in navigator && location.protocol.startsWith("http")) {
  navigator.serviceWorker.register("sw.js").catch(() => { /* app still works online */ });
}

// ======================================================================
// Boot
// ======================================================================
fetch(COURSE.bank)
  .then(r => { if (!r.ok) throw new Error("HTTP " + r.status); return r.text(); })
  .then(text => {
    parseBank(text);
    try { history.replaceState({ depth: 1 }, ""); } catch (e) { /* ignore */ }
    switchTab(SCREENS[progress.lastTab] && ["learn", "slides", "cards", "quiz", "exams", "boards"].includes(progress.lastTab) ? progress.lastTab : "learn");
  })
  .catch(err => {
    view.innerHTML = "";
    view.append(el(`<div class="card pad empty">The question bank could not load (${esc(err.message)}). Reload the page to try again.</div>`));
  });
