"use strict";

/*
 * 内容配置：填写链接即可启用对应按钮，留空则显示 Soon。
 * 可用 https:// 链接，或相对于 docs/index.html 的本地路径。
 */
const PROJECT_LINKS = {
  paper: "", // 例如：assets/paper.pdf
  arxiv: "", // 例如：https://arxiv.org/abs/你的论文编号
  code: "", // 例如：https://github.com/你的用户名/仓库名
};

/*
 * 恢复前后对比图：给每个任务同时填写 before 和 after，即切换为真实结果。
 * 输入 / 输出图请使用相同尺寸。路径相对于 docs/index.html。
 * caption 为图注；alt 为无障碍图片描述。
 * 示例：before: "assets/denoising-input.png", after: "assets/denoising-ours.png"
 */
const RESTORATION_TASKS = {
  denoising: { label: "Denoising", before: "", after: "", caption: "", alt: "" },
  deraining: { label: "Deraining", before: "", after: "", caption: "", alt: "" },
  dehazing: { label: "Dehazing", before: "", after: "", caption: "", alt: "" },
  deblurring: { label: "Deblurring", before: "", after: "", caption: "", alt: "" },
  lowlight: { label: "Low-light enhancement", before: "", after: "", caption: "", alt: "" },
};

const PLACEHOLDER_IMAGE = "assets/scene-placeholder.svg";
const PLACEHOLDER_EFFECTS = {
  denoising: "Synthetic noise",
  deraining: "Synthetic rain",
  dehazing: "Synthetic haze",
  deblurring: "Synthetic blur",
  lowlight: "Synthetic low light",
};

// 只允许常规网站链接 / 本地资源路径，防止误填脚本 URL。
function safeResourceUrl(value) {
  if (typeof value !== "string" || !value.trim()) return null;
  try {
    const url = new URL(value.trim(), document.baseURI);
    if (["http:", "https:"].includes(url.protocol)) return url.href;
    if (url.protocol === "file:" && window.location.protocol === "file:") return url.href;
  } catch {
    return null;
  }
  return null;
}

document.querySelectorAll("[data-resource]").forEach((link) => {
  const url = safeResourceUrl(PROJECT_LINKS[link.dataset.resource]);
  if (!url) return;
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.removeAttribute("aria-disabled");
  link.removeAttribute("title");
  link.querySelector(".resource-status")?.remove();
});

const comparison = document.querySelector(".comparison");
const slider = document.querySelector("#comparison-slider");
const taskTabs = Array.from(document.querySelectorAll(".task-tab"));
const comparisonPanel = document.querySelector("#comparison-panel");
const beforeImage = comparison.querySelector(".comparison-before img");
const afterImage = comparison.querySelector(".comparison-after");
const figureNote = document.querySelector(".comparison-caption > span:first-child");
const taskCaption = document.querySelector("#task-caption");
const sceneNote = document.querySelector(".scene-note");
let selectedTask = "denoising";
let selectionVersion = 0;

function setSplit(value) {
  const split = Math.min(100, Math.max(0, Number(value)));
  slider.value = String(split);
  comparison.style.setProperty("--split", `${split}%`);
  slider.setAttribute("aria-valuetext", `${split}% degraded input, ${100 - split}% restored output`);
}

// 原生 range 保留键盘方向键；指针事件让鼠标 / 触屏在整个画面内直接定位分界线。
slider.addEventListener("input", () => setSplit(slider.value));
let dragging = false;
function splitAtPointer(event) {
  const bounds = comparison.getBoundingClientRect();
  setSplit(Math.round(((event.clientX - bounds.left) / bounds.width) * 100));
}
slider.addEventListener("pointerdown", (event) => {
  if (!event.isPrimary || event.button !== 0) return;
  dragging = true;
  slider.setPointerCapture(event.pointerId);
  slider.focus({ preventScroll: true });
  splitAtPointer(event);
  event.preventDefault();
});
slider.addEventListener("pointermove", (event) => {
  if (dragging) splitAtPointer(event);
});
slider.addEventListener("pointerup", (event) => {
  if (dragging) splitAtPointer(event);
  dragging = false;
});
slider.addEventListener("pointercancel", () => { dragging = false; });
slider.addEventListener("lostpointercapture", () => { dragging = false; });

function showPlaceholder(task, failed = false) {
  comparison.removeAttribute("data-real-pair");
  beforeImage.src = PLACEHOLDER_IMAGE;
  afterImage.src = PLACEHOLDER_IMAGE;
  beforeImage.alt = `Illustrative degraded input placeholder for ${task.label.toLowerCase()}`;
  afterImage.alt = "Illustrative coastal scene used as a restored output placeholder";
  figureNote.lastChild.textContent = failed
    ? " Image pair unavailable · Showing an illustrative placeholder."
    : " Illustrative placeholder · Replace with actual restoration pairs.";
  taskCaption.textContent = `${task.label} · ${PLACEHOLDER_EFFECTS[selectedTask]} for layout preview only.`;
  sceneNote.hidden = false;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = url;
  });
}

async function selectTask(key) {
  const task = RESTORATION_TASKS[key];
  if (!task) return;
  selectedTask = key;
  const version = ++selectionVersion;
  taskTabs.forEach((tab) => {
    const selected = tab.dataset.task === key;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  comparison.dataset.degradation = key;
  comparisonPanel.setAttribute("aria-labelledby", `tab-${key}`);
  comparisonPanel.removeAttribute("aria-busy");
  document.querySelector('label[for="comparison-slider"]').textContent = `Compare ${task.label.toLowerCase()}: move the divider between degraded input and restored output`;
  setSplit(50);
  showPlaceholder(task);
  const before = safeResourceUrl(task.before);
  const after = safeResourceUrl(task.after);
  if (!before || !after) return;

  // 先加载整对图再切换，防止慢加载或快速切换导致输入 / 输出来自不同任务。
  comparisonPanel.setAttribute("aria-busy", "true");
  try {
    await Promise.all([loadImage(before), loadImage(after)]);
    if (version !== selectionVersion) return;
    beforeImage.src = before;
    afterImage.src = after;
    beforeImage.alt = `${task.label} degraded input${task.alt ? `: ${task.alt}` : ""}`;
    afterImage.alt = `${task.label} restored output${task.alt ? `: ${task.alt}` : ""}`;
    comparison.dataset.realPair = "true";
    figureNote.lastChild.textContent = " Input / output comparison · Drag the divider to inspect details.";
    taskCaption.textContent = task.caption || task.label;
    sceneNote.hidden = true;
  } catch {
    if (version === selectionVersion) showPlaceholder(task, true);
  } finally {
    if (version === selectionVersion) comparisonPanel.removeAttribute("aria-busy");
  }
}

taskTabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectTask(tab.dataset.task));
  tab.addEventListener("keydown", (event) => {
    let nextIndex;
    if (event.key === "ArrowRight") nextIndex = (index + 1) % taskTabs.length;
    else if (event.key === "ArrowLeft") nextIndex = (index - 1 + taskTabs.length) % taskTabs.length;
    else if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = taskTabs.length - 1;
    else return;
    event.preventDefault();
    taskTabs[nextIndex].focus();
    selectTask(taskTabs[nextIndex].dataset.task);
  });
});

const copyButton = document.querySelector("#copy-citation");
const copyStatus = document.querySelector("#copy-status");
let copyResetTimer;
copyButton.addEventListener("click", async () => {
  const citation = document.querySelector("#bibtex").textContent.trim();
  let copied = false;
  try {
    await navigator.clipboard.writeText(citation);
    copied = true;
  } catch {
    // file:// 或受限浏览器中的兼容路径。
    const field = document.createElement("textarea");
    field.value = citation;
    field.setAttribute("readonly", "");
    field.style.cssText = "position:fixed;left:-9999px;top:0;";
    document.body.append(field);
    field.select();
    try { copied = document.execCommand("copy"); } catch { copied = false; }
    field.remove();
    copyButton.focus({ preventScroll: true });
  }
  clearTimeout(copyResetTimer);
  copyButton.querySelector("span").textContent = copied ? "Copied!" : "Select text";
  copyStatus.textContent = copied ? "BibTeX copied to clipboard." : "Clipboard unavailable. The BibTeX text has been selected for manual copying.";
  if (!copied) {
    const range = document.createRange();
    range.selectNodeContents(document.querySelector("#bibtex"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  }
  copyResetTimer = setTimeout(() => { copyButton.querySelector("span").textContent = "Copy"; }, 2200);
});

selectTask(selectedTask);
