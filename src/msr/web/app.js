const API_KEY_STORAGE = "msr_api_key";
const PAGE = document.body.dataset.page || "dashboard";
const POLL_INTERVALS = {
  dashboard: 4000,
  models: 5000,
  runtime: 3000,
  transcribe: 5000,
};

const HISTORY = {};
const STATE = {
  lastTranscribeResult: null,
};

const STATUS_LABELS = {
  queued: "排队中",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  idle: "未加载",
  loaded: "已加载",
};

const STAGE_LABELS = {
  queued: "等待入队",
  starting: "准备启动",
  normalizing: "音频归一",
  vad: "VAD 切段",
  diarization: "说话人分离",
  asr: "ASR 转写",
  postprocess: "结果整合",
  completed: "已完成",
  failed: "失败",
};

const SPEAKER_COLORS = ["#305c49", "#5f7267", "#84958b", "#aab5ae", "#c0c9c4"];

function $(selector, root = document) {
  return root.querySelector(selector);
}

function $$(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function getApiKey() {
  return localStorage.getItem(API_KEY_STORAGE) || "";
}

function setApiKey(value) {
  localStorage.setItem(API_KEY_STORAGE, value);
}

function hasApiKey() {
  return Boolean(getApiKey().trim());
}

async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const apiKey = getApiKey();
  if (apiKey) {
    headers.set("X-API-Key", apiKey);
  }
  return fetch(path, { ...options, headers });
}

async function parseResponseBody(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function extractErrorMessage(payload, status) {
  if (typeof payload === "string" && payload.trim()) {
    return payload;
  }
  if (!payload || typeof payload !== "object") {
    return `请求失败（HTTP ${status}）`;
  }
  if (payload.code === "queue_full") {
    return `队列已满：并发上限 ${payload.max_parallel_tasks}，排队上限 ${payload.max_queued_tasks}，当前运行 ${payload.active_tasks}，当前排队 ${payload.queued_tasks}`;
  }
  if (typeof payload.detail === "string") {
    return payload.detail;
  }
  if (typeof payload.error === "string") {
    return payload.error;
  }
  if (typeof payload.message === "string") {
    return payload.message;
  }
  return `请求失败（HTTP ${status}）`;
}

async function requestJson(path, options = {}) {
  const response = await apiFetch(path, options);
  const payload = await parseResponseBody(response);
  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, response.status));
  }
  return payload;
}

async function fetchBundle(requests) {
  const entries = Object.entries(requests);
  const results = await Promise.allSettled(entries.map(([, path]) => requestJson(path)));
  const data = {};
  const errors = {};

  results.forEach((result, index) => {
    const [key] = entries[index];
    if (result.status === "fulfilled") {
      data[key] = result.value;
      return;
    }
    errors[key] = result.reason;
    data[key] = null;
  });

  return { data, errors };
}

function kindLabel(kind) {
  if (kind === "asr") {
    return "语音识别";
  }
  if (kind === "diarization") {
    return "说话人分离";
  }
  return String(kind ?? "未知");
}

function statusLabel(status) {
  return STATUS_LABELS[status] || String(status ?? "未知");
}

function stageLabel(stage) {
  return STAGE_LABELS[stage] || String(stage ?? "未知阶段");
}

function toneForStatus(status) {
  if (status === "completed" || status === "running" || status === "loaded") {
    return "ok";
  }
  if (status === "queued") {
    return "warn";
  }
  if (status === "failed") {
    return "danger";
  }
  return "neutral";
}

function setStatusBadge(target, tone, text) {
  if (!target) {
    return;
  }
  target.className = `status-badge${tone && tone !== "neutral" ? ` is-${tone}` : ""}`;
  target.textContent = text;
}

function setButtonBusy(button, busy, busyText) {
  if (!button) {
    return;
  }
  if (!button.dataset.defaultLabel) {
    button.dataset.defaultLabel = button.textContent.trim();
  }
  button.disabled = busy;
  button.classList.toggle("is-busy", busy);
  button.textContent = busy ? busyText : button.dataset.defaultLabel;
}

function formatPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "0%";
  }
  return `${Math.round(numeric)}%`;
}

function formatMs(value) {
  if (value == null) {
    return "未记录";
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "未记录";
  }
  if (numeric < 1000) {
    return `${numeric} 毫秒`;
  }
  return `${(numeric / 1000).toFixed(2)} 秒`;
}

function formatDateTime(value) {
  if (!value) {
    return "未记录";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleString("zh-CN", {
    hour12: false,
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function parseDurationLabel(value) {
  if (!value) {
    return 0;
  }
  const parts = String(value).split(":").map((part) => Number(part));
  if (parts.length === 2) {
    return parts[0] * 60 + parts[1];
  }
  if (parts.length === 3) {
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  return 0;
}

function progressBar(percent, tone = "accent") {
  const clamped = Math.max(0, Math.min(100, Number(percent) || 0));
  return `
    <div class="progress-track">
      <div class="progress-fill is-${tone}" style="width:${clamped}%;"></div>
    </div>
  `;
}

function pushHistory(key, value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return;
  }
  const list = HISTORY[key] || [];
  list.push(Math.max(0, Math.min(100, numeric)));
  while (list.length > 24) {
    list.shift();
  }
  HISTORY[key] = list;
}

function sparklineSvg(values) {
  const points = Array.isArray(values) ? values : [];
  const width = 240;
  const height = 54;
  const safe = points.length ? points : [0];
  const step = safe.length > 1 ? width / (safe.length - 1) : width;
  const path = safe
    .map((value, index) => {
      const x = index * step;
      const y = height - (Math.max(0, Math.min(100, value)) / 100) * (height - 8) - 4;
      return `${x},${y}`;
    })
    .join(" ");

  return `
    <svg class="mini-trend" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <polyline fill="none" stroke="var(--accent)" stroke-width="2" points="${path}"></polyline>
    </svg>
  `;
}

function emptyState(title, note) {
  return `
    <div class="empty-state">
      <div>
        <strong>${escapeHtml(title)}</strong>
        <span>${escapeHtml(note)}</span>
      </div>
    </div>
  `;
}

function normalizePath(pathname) {
  if (!pathname || pathname === "/") {
    return "/";
  }
  return pathname.endsWith("/") ? pathname.slice(0, -1) : pathname;
}

function highlightNav() {
  const path = normalizePath(window.location.pathname);
  const map = {
    "/": "dashboard",
    "/models": "models",
    "/runtime": "runtime",
    "/transcribe": "transcribe",
  };
  const current = map[path] || "dashboard";
  $$("[data-nav]").forEach((link) => {
    link.classList.toggle("is-active", link.dataset.nav === current);
  });
}

function syncItemVisibility(item) {
  const hidden =
    item.classList.contains("is-hidden-by-search") || item.classList.contains("is-hidden-by-filter");
  item.classList.toggle("is-hidden", hidden);
}

function updateSearchEmpty() {
  const empty = $("#searchEmpty");
  const input = $("#pageSearch");
  if (!empty || !input) {
    return;
  }
  const query = input.value.trim().toLowerCase();
  if (!query) {
    empty.classList.add("is-hidden");
    return;
  }
  const items = $$("main [data-search]");
  const visibleCount = items.filter((item) => !item.classList.contains("is-hidden")).length;
  empty.classList.toggle("is-hidden", visibleCount > 0);
}

function applyPageSearch(query) {
  const normalized = String(query || "").trim().toLowerCase();
  $$("main [data-search]").forEach((item) => {
    const haystack = `${item.dataset.search || ""} ${item.textContent || ""}`.toLowerCase();
    const matched = !normalized || haystack.includes(normalized);
    item.classList.toggle("is-hidden-by-search", !matched);
    syncItemVisibility(item);
  });
  updateSearchEmpty();
}

function initSearch() {
  const input = $("#pageSearch");
  if (!input) {
    return;
  }
  input.addEventListener("input", () => applyPageSearch(input.value));
}

function setTabIndex(list, index) {
  const buttons = $$(".tab-button", list);
  const host = list.closest(".panel") || list.closest("section") || document;
  const panelsHost = $(".tab-panels", host);
  const panels = panelsHost ? Array.from(panelsHost.children).filter((node) => node.hasAttribute("data-tab-panel")) : [];
  buttons.forEach((button, buttonIndex) => {
    button.classList.toggle("is-active", buttonIndex === index);
  });
  panels.forEach((panel, panelIndex) => {
    panel.classList.toggle("is-active", panelIndex === index);
  });
}

function initTabLists() {
  $$("[data-tab-list]").forEach((list) => {
    const buttons = $$(".tab-button", list);
    const activeIndex = Math.max(
      0,
      buttons.findIndex((button) => button.classList.contains("is-active"))
    );
    if (buttons.length) {
      setTabIndex(list, activeIndex);
    }
  });
}

function applyFilterGroup(group) {
  const active = $(".chip.is-active", group) || $(".chip", group);
  const target = group.dataset.filterTarget;
  if (!active || !target) {
    return;
  }
  const value = active.dataset.filterValue || "all";
  $$(target).forEach((item) => {
    const tags = String(item.dataset.tags || "")
      .split(/\s+/)
      .filter(Boolean);
    const matched = value === "all" || tags.includes(value);
    item.classList.toggle("is-hidden-by-filter", !matched);
    syncItemVisibility(item);
  });
  updateSearchEmpty();
}

function initFilterGroups() {
  $$("[data-filter-target]").forEach((group) => {
    if (!$(".chip.is-active", group) && $(".chip", group)) {
      $(".chip", group).classList.add("is-active");
    }
    applyFilterGroup(group);
  });
}

function hydrateUiSurface() {
  highlightNav();
  initTabLists();
  initFilterGroups();
  const search = $("#pageSearch");
  if (search) {
    applyPageSearch(search.value);
  }
}

function renderAuthState(authenticated, detail) {
  const badge = $("#authStatusBadge");
  const text = $("#authStatusText");
  if (authenticated) {
    setStatusBadge(badge, "ok", "已鉴权");
    if (text) {
      text.textContent = detail || "管理接口鉴权成功，可以继续访问页面数据。";
    }
  } else if (hasApiKey()) {
    setStatusBadge(badge, "danger", "鉴权失败");
    if (text) {
      text.textContent = detail || "当前密钥不可用，请检查后重试。";
    }
  } else {
    setStatusBadge(badge, "warn", "待录入");
    if (text) {
      text.textContent = "请输入管理密钥后再校验连接。";
    }
  }

  if (PAGE === "dashboard") {
    const metricValue = $("#metricAuthValue");
    const metricNote = $("#metricAuthNote");
    const metricBadge = $("#metricAuthBadge");
    if (metricValue) {
      metricValue.textContent = authenticated ? "已连接" : hasApiKey() ? "不可用" : "待校验";
    }
    if (metricNote) {
      metricNote.textContent = authenticated ? "管理接口鉴权通过" : detail || "请先完成鉴权";
    }
    setStatusBadge(metricBadge, authenticated ? "ok" : hasApiKey() ? "danger" : "warn", authenticated ? "已连接" : hasApiKey() ? "失败" : "待校验");
  }
}

async function refreshAuth(quiet = true) {
  if (!hasApiKey()) {
    renderAuthState(false, "尚未录入管理密钥。");
    return { authenticated: false };
  }
  try {
    await requestJson("/api/v1/auth/check");
    renderAuthState(true, "管理接口鉴权通过。");
    return { authenticated: true };
  } catch (error) {
    renderAuthState(false, error.message);
    if (!quiet) {
      window.alert(`鉴权失败：${error.message}`);
    }
    return { authenticated: false, error };
  }
}

function renderResourceCards(resources) {
  const target = $("#resourceDashboard");
  if (!target) {
    return;
  }
  if (!resources) {
    target.innerHTML = emptyState("尚未获取资源数据", "保存密钥并刷新后，这里会自动轮询。");
    return;
  }

  pushHistory("cpu", resources.cpu?.percent);
  pushHistory("memory", resources.memory?.percent);
  const gpuItems = Array.isArray(resources.gpu) ? resources.gpu : [];
  gpuItems.forEach((gpu) => pushHistory(`gpu-${gpu.index}`, gpu.utilization_percent));

  const cpuCard = `
    <article class="resource-card" data-search="CPU 资源 利用率" data-tags="资源">
      <div class="section-head">
        <div>
          <h4>CPU 利用率</h4>
          <p>逻辑核心数 ${escapeHtml(resources.cpu?.count ?? 0)}</p>
        </div>
        ${renderStatusBadge(formatPercent(resources.cpu?.percent), "ok")}
      </div>
      <div class="progress-row">
        <div class="progress-meta"><span>当前水位</span><span>${formatPercent(resources.cpu?.percent)}</span></div>
        ${progressBar(resources.cpu?.percent, "accent")}
      </div>
      ${sparklineSvg(HISTORY.cpu)}
    </article>
  `;

  const memoryCard = `
    <article class="resource-card" data-search="内存 资源 占用" data-tags="资源">
      <div class="section-head">
        <div>
          <h4>内存占用</h4>
          <p>${escapeHtml(resources.memory?.used_mb ?? 0)} MB / ${escapeHtml(resources.memory?.total_mb ?? 0)} MB</p>
        </div>
        ${renderStatusBadge(formatPercent(resources.memory?.percent), resources.memory?.percent > 85 ? "warn" : "ok")}
      </div>
      <div class="progress-row">
        <div class="progress-meta"><span>当前水位</span><span>${formatPercent(resources.memory?.percent)}</span></div>
        ${progressBar(resources.memory?.percent, "accent")}
      </div>
      ${sparklineSvg(HISTORY.memory)}
    </article>
  `;

  const gpuCards = gpuItems.length
    ? gpuItems
        .map((gpu) => {
          const memoryPercent = gpu.memory_total_mb
            ? (Number(gpu.memory_used_mb) / Number(gpu.memory_total_mb)) * 100
            : 0;
          return `
            <article class="resource-card" data-search="GPU ${escapeHtml(gpu.name)} 显存 ${escapeHtml(gpu.index)}" data-tags="资源">
              <div class="section-head">
                <div>
                  <h4>GPU ${escapeHtml(gpu.index)} · ${escapeHtml(gpu.name)}</h4>
                  <p>显存 ${escapeHtml(gpu.memory_used_mb)} MB / ${escapeHtml(gpu.memory_total_mb)} MB</p>
                </div>
                ${renderStatusBadge(formatPercent(gpu.utilization_percent), gpu.utilization_percent > 85 ? "warn" : "ok")}
              </div>
              <div class="progress-row">
                <div class="progress-meta"><span>计算利用率</span><span>${formatPercent(gpu.utilization_percent)}</span></div>
                ${progressBar(gpu.utilization_percent, "accent")}
              </div>
              <div class="progress-row">
                <div class="progress-meta"><span>显存占用</span><span>${formatPercent(memoryPercent)}</span></div>
                ${progressBar(memoryPercent, memoryPercent > 85 ? "warn" : "ok")}
              </div>
              ${sparklineSvg(HISTORY[`gpu-${gpu.index}`])}
            </article>
          `;
        })
        .join("")
    : `
      <article class="resource-card" data-search="GPU 不可用" data-tags="资源">
        <div class="section-head">
          <div>
            <h4>GPU 资源</h4>
            <p>当前未检测到可用 GPU 或监控库未就绪。</p>
          </div>
          ${renderStatusBadge("未检测", "warn")}
        </div>
      </article>
    `;

  target.innerHTML = cpuCard + memoryCard + gpuCards;

  const hint = $("#resourceSampleHint");
  if (hint) {
    setStatusBadge(hint, "ok", `已采样 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`);
  }
}

function renderStatusBadge(text, tone = "neutral") {
  return `<span class="status-badge${tone !== "neutral" ? ` is-${tone}` : ""}">${escapeHtml(text)}</span>`;
}

function renderActiveModels(runtime) {
  const target = $("#activeModelCards");
  if (!target) {
    return;
  }
  const items = [
    runtime?.asr
      ? {
          title: "当前 ASR",
          note: `${runtime.asr.id} · ${runtime.asr.backend}`,
          detail: `设备 ${runtime.asr.device} · ${runtime.asr.local_path}`,
        }
      : null,
    runtime?.diarization
      ? {
          title: "当前说话人分离",
          note: `${runtime.diarization.id} · ${runtime.diarization.backend}`,
          detail: `设备 ${runtime.diarization.device} · ${runtime.diarization.local_path}`,
        }
      : null,
  ].filter(Boolean);

  if (!items.length) {
    target.innerHTML = emptyState("暂无已激活模型", "请前往模型管理页加载默认模型或备选模型。");
    return;
  }

  target.innerHTML = items
    .map(
      (item) => `
        <article class="model-card" data-search="${escapeHtml(item.title)} ${escapeHtml(item.note)} ${escapeHtml(item.detail)}" data-tags="模型">
          <div class="section-head">
            <div>
              <h4>${escapeHtml(item.title)}</h4>
              <p>${escapeHtml(item.note)}</p>
            </div>
            ${renderStatusBadge("已激活", "ok")}
          </div>
          <p>${escapeHtml(item.detail)}</p>
        </article>
      `
    )
    .join("");
}

function renderTaskItem(task) {
  const status = task.status || "queued";
  const tone = toneForStatus(status);
  const tags = [status, String(task.stage || ""), "任务"].join(" ");
  return `
    <article class="task-item" data-search="${escapeHtml(task.task_id)} ${escapeHtml(task.filename)} ${escapeHtml(statusLabel(task.status))} ${escapeHtml(stageLabel(task.stage))} ${escapeHtml(task.error || "")}" data-tags="${escapeHtml(tags)}">
      <div class="task-head">
        <div>
          <div class="task-name">${escapeHtml(task.filename || "未命名文件")}</div>
          <div class="task-subtitle mono">${escapeHtml(task.task_id)}</div>
        </div>
        ${renderStatusBadge(statusLabel(status), tone)}
      </div>
      <div class="card-meta-grid">
        <div class="meta-pair">
          <span>当前阶段</span>
          <strong>${escapeHtml(stageLabel(task.stage))}</strong>
        </div>
        <div class="meta-pair">
          <span>音频时长</span>
          <strong>${escapeHtml(task.audio_duration || "待识别")}</strong>
        </div>
        <div class="meta-pair">
          <span>排队等待</span>
          <strong>${escapeHtml(formatMs(task.queue_wait_ms))}</strong>
        </div>
        <div class="meta-pair">
          <span>运行耗时</span>
          <strong>${escapeHtml(formatMs(task.run_ms))}</strong>
        </div>
        <div class="meta-pair">
          <span>提交时间</span>
          <strong>${escapeHtml(formatDateTime(task.submitted_at))}</strong>
        </div>
        <div class="meta-pair">
          <span>完成时间</span>
          <strong>${escapeHtml(formatDateTime(task.finished_at))}</strong>
        </div>
      </div>
      ${
        task.error
          ? `<div class="split-line"></div><div class="meta-pair"><span>失败原因</span><strong>${escapeHtml(task.error)}</strong></div>`
          : ""
      }
    </article>
  `;
}

function renderTaskList(target, tasks, title, note) {
  if (!target) {
    return;
  }
  if (!Array.isArray(tasks) || !tasks.length) {
    target.innerHTML = emptyState(title, note);
    return;
  }
  target.innerHTML = tasks.map(renderTaskItem).join("");
}

function renderDashboardTasks(tasks) {
  renderTaskList(
    $("#dashboardTasksActive"),
    tasks?.active,
    "当前没有运行中的任务",
    "新任务进入并发槽后，这里会显示实时阶段和耗时。"
  );
  renderTaskList(
    $("#dashboardTasksQueued"),
    tasks?.queued,
    "当前没有排队任务",
    "等待队列为空时，说明当前吞吐还有余量。"
  );
  renderTaskList(
    $("#dashboardTasksRecent"),
    tasks?.recent,
    "还没有最近任务摘要",
    "完成或失败的任务会自动进入最近任务列表。"
  );
}

function renderDashboardMetrics(models, resources, tasks, limits) {
  const total = Array.isArray(models) ? models.length : 0;
  const loaded = Array.isArray(models) ? models.filter((model) => model.loaded).length : 0;
  const active = tasks?.counts?.active ?? 0;
  const queued = tasks?.counts?.queued ?? 0;
  const firstGpu = Array.isArray(resources?.gpu) && resources.gpu.length ? resources.gpu[0] : null;

  const modelValue = $("#metricModelValue");
  const modelNote = $("#metricModelNote");
  if (modelValue) {
    modelValue.textContent = `${loaded} / ${total}`;
  }
  if (modelNote) {
    const loadedKinds = Array.isArray(models)
      ? {
          asr: models.filter((model) => model.kind === "asr" && model.loaded).length,
          diarization: models.filter((model) => model.kind === "diarization" && model.loaded).length,
        }
      : { asr: 0, diarization: 0 };
    modelNote.textContent = `语音识别 ${loadedKinds.asr}，说话人分离 ${loadedKinds.diarization}`;
  }

  const taskValue = $("#metricTaskValue");
  const taskNote = $("#metricTaskNote");
  if (taskValue) {
    taskValue.textContent = `${active} | ${queued}`;
  }
  if (taskNote) {
    taskNote.textContent = `并发上限 ${limits?.max_parallel_tasks ?? 0}，排队上限 ${limits?.max_queued_tasks ?? 0}`;
  }

  const gpuValue = $("#metricGpuValue");
  const gpuNote = $("#metricGpuNote");
  if (gpuValue) {
    gpuValue.textContent = firstGpu ? formatPercent(firstGpu.utilization_percent) : "未检测";
  }
  if (gpuNote) {
    gpuNote.textContent = firstGpu
      ? `显存 ${firstGpu.memory_used_mb} / ${firstGpu.memory_total_mb} MB`
      : "当前未检测到可用 GPU。";
  }
}

function renderModelsMetrics(models, runtime) {
  const total = Array.isArray(models) ? models.length : 0;
  const loaded = Array.isArray(models) ? models.filter((model) => model.loaded).length : 0;
  const pathOk = Array.isArray(models) ? models.filter((model) => model.path_exists).length : 0;
  const defaults = Array.isArray(models) ? models.filter((model) => model.default) : [];
  const defaultReady =
    Boolean(runtime?.asr?.id) &&
    Boolean(runtime?.diarization?.id) &&
    defaults.some((model) => model.id === runtime?.asr?.id) &&
    defaults.some((model) => model.id === runtime?.diarization?.id);

  $("#modelsMetricTotal").textContent = String(total);
  $("#modelsMetricTotalNote").textContent = `${defaults.length} 个默认模型已注册`;
  $("#modelsMetricLoaded").textContent = String(loaded);
  $("#modelsMetricLoadedNote").textContent = loaded ? "已加载模型可直接参与主链服务" : "当前没有已加载模型";
  $("#modelsMetricPath").textContent = total ? formatPercent((pathOk / total) * 100) : "0%";
  $("#modelsMetricPathNote").textContent = `${pathOk} / ${total} 个模型路径存在`;
  $("#modelsMetricDefault").textContent = defaultReady ? "已就绪" : "未就绪";
  $("#modelsMetricDefaultNote").textContent = defaultReady
    ? "默认 ASR 与默认 diarization 均已激活"
    : "默认主链尚未完全加载";
}

function renderModelRows(models) {
  const target = $("#modelsTableBody");
  if (!target) {
    return;
  }
  if (!Array.isArray(models) || !models.length) {
    target.innerHTML = `
      <tr>
        <td colspan="8">${emptyState("尚未加载模型注册表", "保存密钥并刷新后，这里会显示可操作模型。")}</td>
      </tr>
    `;
    return;
  }

  target.innerHTML = models
    .map((model) => {
      const action = model.loaded ? "unload" : "load";
      const actionLabel = model.loaded ? "卸载模型" : "加载模型";
      const tags = [
        model.kind,
        model.loaded ? "loaded" : "idle",
        model.path_exists ? "path-ok" : "path-missing",
        model.default ? "default" : "optional",
      ].join(" ");
      return `
        <tr data-search="${escapeHtml(model.id)} ${escapeHtml(model.backend)} ${escapeHtml(model.device)} ${escapeHtml(model.local_path)} ${escapeHtml(kindLabel(model.kind))}" data-tags="${escapeHtml(tags)}">
          <td>
            <span class="table-title">${escapeHtml(model.id)}</span>
            <span class="table-note">${model.default ? "默认链模型" : "备选模型"}</span>
          </td>
          <td>${renderStatusBadge(kindLabel(model.kind), "neutral")}</td>
          <td>${escapeHtml(model.backend)}</td>
          <td>${escapeHtml(model.device)}</td>
          <td>${renderStatusBadge(model.path_exists ? "路径正常" : "路径缺失", model.path_exists ? "ok" : "danger")}</td>
          <td>${renderStatusBadge(model.loaded ? "已加载" : "未加载", model.loaded ? "ok" : "warn")}</td>
          <td class="mono">${escapeHtml(model.local_path)}</td>
          <td>
            <button
              class="button model-action ${model.loaded ? "" : "is-primary"}"
              type="button"
              data-kind="${escapeHtml(model.kind)}"
              data-id="${escapeHtml(model.id)}"
              data-action="${escapeHtml(action)}"
            >
              ${escapeHtml(actionLabel)}
            </button>
          </td>
        </tr>
      `;
    })
    .join("");
}

function renderModelsMatrix(models) {
  const target = $("#modelsMatrix");
  if (!target) {
    return;
  }
  if (!Array.isArray(models) || !models.length) {
    target.innerHTML = `<div class="empty-state grid-span-3"><div><strong>尚无模型分布数据</strong><span>刷新后将按类型、后端与设备进行聚合展示。</span></div></div>`;
    return;
  }

  const asr = models.filter((model) => model.kind === "asr");
  const diarization = models.filter((model) => model.kind === "diarization");
  const backends = Object.entries(
    models.reduce((acc, model) => {
      acc[model.backend] = (acc[model.backend] || 0) + 1;
      return acc;
    }, {})
  );
  const devices = Object.entries(
    models.reduce((acc, model) => {
      acc[model.device] = (acc[model.device] || 0) + 1;
      return acc;
    }, {})
  );

  target.innerHTML = `
    <article class="matrix-card" data-search="语音识别 asr" data-tags="asr">
      <h4>语音识别模型</h4>
      <p>注册 ${asr.length} 个，已加载 ${asr.filter((item) => item.loaded).length} 个。</p>
    </article>
    <article class="matrix-card" data-search="说话人分离 diarization" data-tags="diarization">
      <h4>说话人分离模型</h4>
      <p>注册 ${diarization.length} 个，已加载 ${diarization.filter((item) => item.loaded).length} 个。</p>
    </article>
    <article class="matrix-card" data-search="默认链 主链" data-tags="default">
      <h4>默认链模型</h4>
      <p>默认模型 ${models.filter((item) => item.default).length} 个，其中已加载 ${models.filter((item) => item.default && item.loaded).length} 个。</p>
    </article>
    <article class="matrix-card" data-search="后端 分布" data-tags="backend">
      <h4>后端分布</h4>
      <p>${backends.map(([name, count]) => `${name} ${count} 个`).join("，")}</p>
    </article>
    <article class="matrix-card" data-search="设备 分布" data-tags="device">
      <h4>设备分布</h4>
      <p>${devices.map(([name, count]) => `${name} ${count} 个`).join("，")}</p>
    </article>
    <article class="matrix-card" data-search="路径 健康" data-tags="path">
      <h4>路径健康度</h4>
      <p>路径正常 ${models.filter((item) => item.path_exists).length} 个，路径缺失 ${models.filter((item) => !item.path_exists).length} 个。</p>
    </article>
  `;
}

function renderRuntimeMetrics(tasks, limits) {
  const counts = tasks?.counts || { active: 0, queued: 0, recent: 0 };
  $("#runtimeMetricActive").textContent = String(counts.active);
  $("#runtimeMetricActiveNote").textContent = counts.active ? "当前有任务正在占用运行槽" : "当前没有运行中的任务";
  $("#runtimeMetricQueued").textContent = String(counts.queued);
  $("#runtimeMetricQueuedNote").textContent = counts.queued ? "等待队列中仍有任务" : "当前等待队列为空";
  $("#runtimeMetricParallel").textContent = String(limits?.max_parallel_tasks ?? 0);
  $("#runtimeMetricParallelNote").textContent = "只影响新入队任务";
  $("#runtimeMetricRecent").textContent = String(limits?.recent_task_limit ?? 0);
  $("#runtimeMetricRecentNote").textContent = `最近任务摘要当前记录 ${counts.recent} 条`;
}

function renderRuntimeLimits(limits) {
  if (!limits) {
    setStatusBadge($("#runtimeLimitsBadge"), "warn", "待加载");
    $("#runtimeLimitsText").textContent = "尚未获取运行限制。";
    return;
  }
  $("#maxParallelInput").value = limits.max_parallel_tasks;
  $("#maxQueuedInput").value = limits.max_queued_tasks;
  $("#recentTaskLimitInput").value = limits.recent_task_limit;
  setStatusBadge($("#runtimeLimitsBadge"), "ok", "已加载");
  $("#runtimeLimitsText").textContent = `当前并发上限 ${limits.max_parallel_tasks}，排队上限 ${limits.max_queued_tasks}，最近任务保留 ${limits.recent_task_limit} 条。`;
}

function renderRuntimePressure(tasks, limits, resources) {
  const target = $("#runtimePressurePanel");
  if (!target) {
    return;
  }
  if (!tasks || !limits) {
    target.innerHTML = emptyState("尚未获取运行压力数据", "刷新后会展示并发槽、队列槽与 GPU 当前水位。");
    return;
  }

  const active = tasks.counts?.active ?? 0;
  const queued = tasks.counts?.queued ?? 0;
  const parallelPercent = limits.max_parallel_tasks ? (active / limits.max_parallel_tasks) * 100 : 0;
  const queuedPercent = limits.max_queued_tasks ? (queued / limits.max_queued_tasks) * 100 : 0;
  const firstGpu = Array.isArray(resources?.gpu) && resources.gpu.length ? resources.gpu[0] : null;

  target.innerHTML = `
    <article class="resource-card" data-search="并发 槽 运行中" data-tags="running">
      <div class="section-head">
        <div>
          <h4>并发槽占用</h4>
          <p>${active} / ${limits.max_parallel_tasks}</p>
        </div>
        ${renderStatusBadge(formatPercent(parallelPercent), parallelPercent > 85 ? "warn" : "ok")}
      </div>
      <div class="progress-row">
        <div class="progress-meta"><span>运行中占比</span><span>${formatPercent(parallelPercent)}</span></div>
        ${progressBar(parallelPercent, parallelPercent > 85 ? "warn" : "accent")}
      </div>
    </article>
    <article class="resource-card" data-search="排队 槽 队列" data-tags="queued">
      <div class="section-head">
        <div>
          <h4>排队槽占用</h4>
          <p>${queued} / ${limits.max_queued_tasks}</p>
        </div>
        ${renderStatusBadge(formatPercent(queuedPercent), queuedPercent > 85 ? "warn" : "ok")}
      </div>
      <div class="progress-row">
        <div class="progress-meta"><span>排队占比</span><span>${formatPercent(queuedPercent)}</span></div>
        ${progressBar(queuedPercent, queuedPercent > 85 ? "warn" : "accent")}
      </div>
    </article>
    <article class="resource-card" data-search="GPU 资源 显存" data-tags="running">
      <div class="section-head">
        <div>
          <h4>GPU 联动观察</h4>
          <p>${firstGpu ? `${firstGpu.name}` : "当前未检测到 GPU"}</p>
        </div>
        ${renderStatusBadge(firstGpu ? formatPercent(firstGpu.utilization_percent) : "未检测", firstGpu ? "ok" : "warn")}
      </div>
      ${
        firstGpu
          ? `
            <div class="progress-row">
              <div class="progress-meta"><span>GPU 利用率</span><span>${formatPercent(firstGpu.utilization_percent)}</span></div>
              ${progressBar(firstGpu.utilization_percent, "accent")}
            </div>
            <div class="progress-row">
              <div class="progress-meta"><span>显存占用</span><span>${firstGpu.memory_used_mb} / ${firstGpu.memory_total_mb} MB</span></div>
              ${progressBar((firstGpu.memory_used_mb / firstGpu.memory_total_mb) * 100, "ok")}
            </div>
          `
          : ""
      }
    </article>
  `;
}

function renderRuntimeTasks(tasks) {
  renderTaskList($("#runtimeActiveTasks"), tasks?.active, "当前没有运行中的任务", "新的执行任务会在这里显示阶段与耗时。");
  renderTaskList($("#runtimeQueuedTasks"), tasks?.queued, "当前没有排队任务", "等待队列为空时，这里不会显示卡片。");
  renderTaskList($("#runtimeRecentTasks"), tasks?.recent, "还没有最近任务摘要", "完成或失败的任务会自动进入最近列表。");
}

function renderTranscribeReadiness(runtime, tasks) {
  const asr = runtime?.asr;
  const diarization = runtime?.diarization;
  const counts = tasks?.counts || { active: 0, queued: 0 };

  $("#transcribeMetricAsr").textContent = asr ? "已就绪" : "未加载";
  $("#transcribeMetricAsrNote").textContent = asr ? `${asr.id} · ${asr.backend}` : "请先加载 ASR 模型";

  $("#transcribeMetricDiarization").textContent = diarization ? "已就绪" : "未加载";
  $("#transcribeMetricDiarizationNote").textContent = diarization ? `${diarization.id} · ${diarization.backend}` : "请先加载说话人分离模型";

  $("#transcribeMetricQueue").textContent = `${counts.active} / ${counts.queued}`;
  $("#transcribeMetricQueueNote").textContent = `运行中 ${counts.active}，排队中 ${counts.queued}`;

  if (!STATE.lastTranscribeResult) {
    $("#transcribeMetricLast").textContent = "暂无";
    $("#transcribeMetricLastNote").textContent = "请先上传音频";
  }
}

function renderTranscribeSpeakerFilters(speakersInfo) {
  const group = $("#speakerFilterGroup");
  if (!group) {
    return;
  }
  const items = Array.isArray(speakersInfo) ? speakersInfo : [];
  const chips = ['<button class="chip is-active" type="button" data-filter-value="all">全部说话人</button>']
    .concat(
      items.map(
        (speaker) =>
          `<button class="chip" type="button" data-filter-value="speaker-${escapeHtml(speaker.speaker_id)}">${escapeHtml(
            speaker.speaker_label
          )}</button>`
      )
    )
    .join("");
  group.innerHTML = chips;
  applyFilterGroup(group);
}

function renderTranscribeSummary(result) {
  const target = $("#transcribeSummary");
  if (!target) {
    return;
  }
  if (!result) {
    target.innerHTML = `<div class="empty-state grid-span-3"><div><strong>尚未生成转写结果</strong><span>提交音频后，这里会显示音频时长、处理时间、处理速度和说话人数。</span></div></div>`;
    return;
  }

  target.innerHTML = `
    <article class="metric-card" data-search="音频 时长" data-tags="result">
      <div class="metric-head"><div><p class="metric-title">音频时长</p><p class="metric-subtitle">本次提交的输入时长</p></div></div>
      <div class="metric-value">${escapeHtml(result.audio_duration || "未知")}</div>
    </article>
    <article class="metric-card" data-search="处理 时间" data-tags="result">
      <div class="metric-head"><div><p class="metric-title">处理时间</p><p class="metric-subtitle">服务端实际处理耗时</p></div></div>
      <div class="metric-value">${escapeHtml(result.processing_time || "未知")}</div>
    </article>
    <article class="metric-card" data-search="处理 速度" data-tags="result">
      <div class="metric-head"><div><p class="metric-title">处理速度</p><p class="metric-subtitle">音频时长与处理时间的比值</p></div></div>
      <div class="metric-value">${escapeHtml(result.processing_speed || "未知")}</div>
    </article>
    <article class="metric-card" data-search="说话人 数量" data-tags="result">
      <div class="metric-head"><div><p class="metric-title">有效说话人</p><p class="metric-subtitle">仅统计有有效 ASR 文本的说话人</p></div></div>
      <div class="metric-value">${escapeHtml(result.total_speakers ?? 0)}</div>
    </article>
  `;
}

function renderTranscriptTimeline(transcripts) {
  const target = $("#transcriptTimeline");
  if (!target) {
    return;
  }
  if (!Array.isArray(transcripts) || !transcripts.length) {
    target.innerHTML = emptyState("暂无逐段转写内容", "提交音频后，这里会显示按说话人切分后的时间线。");
    return;
  }

  target.innerHTML = transcripts
    .map((item) => {
      const speakerIndex = Number(item.speaker_id) || 0;
      const color = SPEAKER_COLORS[speakerIndex % SPEAKER_COLORS.length];
      return `
        <article class="timeline-item" data-search="${escapeHtml(item.speaker_label)} ${escapeHtml(item.text)} ${escapeHtml(item.start_time)} ${escapeHtml(item.end_time)}" data-tags="speaker-${escapeHtml(item.speaker_id)} result">
          <div class="speaker-row">
            <div class="speaker-color-bar" style="background:${color};"></div>
            <div>
              <div class="timeline-head">
                <div>
                  <div class="timeline-label">${escapeHtml(item.speaker_label)}</div>
                  <div class="timeline-meta">${escapeHtml(item.start_time)} - ${escapeHtml(item.end_time)}</div>
                </div>
                ${renderStatusBadge(`置信度 ${Math.round((Number(item.confidence) || 1) * 100)}%`, "neutral")}
              </div>
              <div class="timeline-text">${escapeHtml(item.text)}</div>
            </div>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderSpeakerSummary(speakersInfo) {
  const target = $("#speakerSummary");
  if (!target) {
    return;
  }
  if (!Array.isArray(speakersInfo) || !speakersInfo.length) {
    target.innerHTML = `<div class="empty-state grid-span-3"><div><strong>暂无说话人统计</strong><span>提交音频后，这里会汇总每位说话人的累计时长和片段数。</span></div></div>`;
    return;
  }

  const maxDuration = Math.max(...speakersInfo.map((item) => parseDurationLabel(item.total_duration) || 1));
  target.innerHTML = speakersInfo
    .map((speaker) => {
      const index = Number(speaker.speaker_id) || 0;
      const color = SPEAKER_COLORS[index % SPEAKER_COLORS.length];
      const width = (parseDurationLabel(speaker.total_duration) / maxDuration) * 100;
      return `
        <article class="speaker-stat" data-search="${escapeHtml(speaker.speaker_label)} ${escapeHtml(speaker.total_duration)}" data-tags="speaker-${escapeHtml(speaker.speaker_id)} result">
          <div class="section-head">
            <div>
              <h4>${escapeHtml(speaker.speaker_label)}</h4>
              <p>说话人编号 ${escapeHtml(speaker.speaker_id)}</p>
            </div>
            ${renderStatusBadge(`片段 ${escapeHtml(speaker.segment_count)}`, "neutral")}
          </div>
          <div class="progress-row">
            <div class="progress-meta"><span>累计时长</span><span>${escapeHtml(speaker.total_duration)}</span></div>
            <div class="progress-track"><div class="progress-fill is-accent" style="width:${Math.max(8, width)}%; background:${color};"></div></div>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderTranscribeResult(result) {
  STATE.lastTranscribeResult = result;
  $("#transcribeMetricLast").textContent = result?.processing_speed || "未知";
  $("#transcribeMetricLastNote").textContent = result
    ? `最近一次处理 ${result.processing_time || "未知"}，有效说话人 ${result.total_speakers ?? 0}`
    : "请先上传音频";
  renderTranscribeSummary(result);
  renderTranscriptTimeline(result?.transcripts);
  renderSpeakerSummary(result?.speakers_info);
  renderTranscribeSpeakerFilters(result?.speakers_info);
  hydrateUiSurface();
}

function renderTranscribeFailure(message) {
  const summary = $("#transcribeSummary");
  if (summary) {
    summary.innerHTML = `
      <article class="metric-card grid-span-2" data-search="失败 错误">
        <div class="metric-head"><div><p class="metric-title">转写失败</p><p class="metric-subtitle">本次请求未返回成功结果</p></div>${renderStatusBadge("失败", "danger")}</div>
        <div class="panel-muted">${escapeHtml(message)}</div>
      </article>
    `;
  }
  $("#transcriptTimeline").innerHTML = emptyState("本次转写未成功", message);
  $("#speakerSummary").innerHTML = `<div class="empty-state grid-span-3"><div><strong>没有可展示的说话人统计</strong><span>${escapeHtml(message)}</span></div></div>`;
}

function updateFileMeta(file) {
  const nameTarget = $("#audioFileName");
  const sizeTarget = $("#audioFileSize");
  if (!nameTarget || !sizeTarget) {
    return;
  }
  if (!file) {
    nameTarget.textContent = "尚未选择文件";
    sizeTarget.textContent = "大小未知";
    return;
  }
  nameTarget.textContent = file.name;
  sizeTarget.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
}

async function refreshDashboardPage() {
  const auth = await refreshAuth(true);
  if (!auth.authenticated) {
    renderDashboardMetrics(null, null, null, null);
    renderResourceCards(null);
    renderActiveModels(null);
    renderDashboardTasks(null);
    return;
  }

  const { data } = await fetchBundle({
    models: "/api/v1/models",
    runtime: "/api/v1/runtime/active",
    resources: "/api/v1/system/resources",
    tasks: "/api/v1/runtime/tasks",
    limits: "/api/v1/runtime/limits",
  });

  renderDashboardMetrics(data.models, data.resources, data.tasks, data.limits);
  renderResourceCards(data.resources);
  renderActiveModels(data.runtime);
  renderDashboardTasks(data.tasks);
  hydrateUiSurface();
}

async function refreshModelsPage() {
  const auth = await refreshAuth(true);
  if (!auth.authenticated) {
    renderModelsMetrics(null, null);
    renderModelRows(null);
    renderModelsMatrix(null);
    return;
  }

  const { data } = await fetchBundle({
    models: "/api/v1/models",
    runtime: "/api/v1/runtime/active",
  });

  renderModelsMetrics(data.models, data.runtime);
  renderModelRows(data.models);
  renderModelsMatrix(data.models);
  hydrateUiSurface();
}

async function refreshRuntimePage() {
  const auth = await refreshAuth(true);
  if (!auth.authenticated) {
    renderRuntimeMetrics(null, null);
    renderRuntimeLimits(null);
    renderRuntimePressure(null, null, null);
    renderRuntimeTasks(null);
    return;
  }

  const { data } = await fetchBundle({
    tasks: "/api/v1/runtime/tasks",
    limits: "/api/v1/runtime/limits",
    resources: "/api/v1/system/resources",
  });

  renderRuntimeMetrics(data.tasks, data.limits);
  renderRuntimeLimits(data.limits);
  renderRuntimePressure(data.tasks, data.limits, data.resources);
  renderRuntimeTasks(data.tasks);
  hydrateUiSurface();
}

async function refreshTranscribePage() {
  const auth = await refreshAuth(true);
  if (!auth.authenticated) {
    renderTranscribeReadiness(null, null);
    if (!STATE.lastTranscribeResult) {
      renderTranscribeSummary(null);
      renderTranscriptTimeline(null);
      renderSpeakerSummary(null);
    }
    return;
  }

  const { data } = await fetchBundle({
    runtime: "/api/v1/runtime/active",
    tasks: "/api/v1/runtime/tasks",
  });

  renderTranscribeReadiness(data.runtime, data.tasks);
  hydrateUiSurface();
}

async function refreshCurrentPage() {
  if (PAGE === "dashboard") {
    await refreshDashboardPage();
    return;
  }
  if (PAGE === "models") {
    await refreshModelsPage();
    return;
  }
  if (PAGE === "runtime") {
    await refreshRuntimePage();
    return;
  }
  if (PAGE === "transcribe") {
    await refreshTranscribePage();
  }
}

async function handleSaveKey() {
  const input = $("#apiKey");
  const button = $("#saveKeyButton");
  if (!input) {
    return;
  }
  setButtonBusy(button, true, "保存中...");
  setApiKey(input.value.trim());
  try {
    await refreshCurrentPage();
  } finally {
    setButtonBusy(button, false, "保存密钥");
  }
}

async function handleAuthCheck() {
  const button = $("#checkAuthButton");
  setButtonBusy(button, true, "校验中...");
  try {
    await refreshAuth(false);
  } finally {
    setButtonBusy(button, false, "校验连接");
  }
}

async function handlePageRefresh() {
  const button = $("#pageRefreshButton");
  setButtonBusy(button, true, "刷新中...");
  try {
    await refreshCurrentPage();
  } finally {
    setButtonBusy(button, false, "刷新当前页");
  }
}

async function handleRuntimeLimitsSubmit(event) {
  event.preventDefault();
  const button = $("#saveRuntimeLimitsButton");
  setButtonBusy(button, true, "保存中...");
  setStatusBadge($("#runtimeLimitsBadge"), "warn", "提交中");
  $("#runtimeLimitsText").textContent = "正在提交新的运行限制，请等待接口响应。";

  const payload = {
    max_parallel_tasks: Number($("#maxParallelInput").value),
    max_queued_tasks: Number($("#maxQueuedInput").value),
    recent_task_limit: Number($("#recentTaskLimitInput").value),
  };

  try {
    await requestJson("/api/v1/runtime/limits", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    await refreshRuntimePage();
  } catch (error) {
    setStatusBadge($("#runtimeLimitsBadge"), "danger", "保存失败");
    $("#runtimeLimitsText").textContent = `保存运行控制失败：${error.message}`;
  } finally {
    setButtonBusy(button, false, "保存运行控制");
  }
}

async function handleModelAction(button) {
  const action = button.dataset.action;
  const kind = button.dataset.kind;
  const modelId = button.dataset.id;
  const busyText = action === "load" ? "加载中..." : "卸载中...";

  setButtonBusy(button, true, busyText);
  try {
    await requestJson(`/api/v1/models/${kind}/${modelId}/${action}`, { method: "POST" });
    await refreshModelsPage();
    if (PAGE === "dashboard") {
      await refreshDashboardPage();
    }
  } catch (error) {
    window.alert(`模型操作失败：${error.message}`);
  } finally {
    setButtonBusy(button, false, action === "load" ? "加载模型" : "卸载模型");
  }
}

async function handleTranscribeSubmit(event) {
  event.preventDefault();
  const input = $("#audioInput");
  const button = $("#submitTranscribeButton");
  const hint = $("#transcribeRequestHint");
  const file = input?.files?.[0];

  if (!file) {
    if (hint) {
      hint.textContent = "请先选择要上传的音频文件。";
    }
    return;
  }

  const formData = new FormData();
  formData.append("audio", file);

  setButtonBusy(button, true, "转写中...");
  if (hint) {
    hint.textContent = "正在上传音频并等待同步转写响应，请稍候。";
  }

  try {
    const response = await apiFetch("/transcribe/", {
      method: "POST",
      body: formData,
    });
    const payload = await parseResponseBody(response);
    if (!response.ok) {
      throw new Error(extractErrorMessage(payload, response.status));
    }
    renderTranscribeResult(payload);
    if (hint) {
      hint.textContent = "本次转写已完成，结果视图已同步刷新。";
    }
    await refreshTranscribePage();
  } catch (error) {
    renderTranscribeFailure(error.message);
    if (hint) {
      hint.textContent = `本次转写失败：${error.message}`;
    }
    await refreshTranscribePage();
  } finally {
    setButtonBusy(button, false, "上传并转写");
  }
}

function bindStaticEvents() {
  const keyInput = $("#apiKey");
  if (keyInput) {
    keyInput.value = getApiKey();
  }

  $("#saveKeyButton")?.addEventListener("click", handleSaveKey);
  $("#checkAuthButton")?.addEventListener("click", handleAuthCheck);
  $("#pageRefreshButton")?.addEventListener("click", handlePageRefresh);
  $("#runtimeLimitsForm")?.addEventListener("submit", handleRuntimeLimitsSubmit);
  $("#transcribeForm")?.addEventListener("submit", handleTranscribeSubmit);
  $("#audioInput")?.addEventListener("change", (event) => updateFileMeta(event.target.files?.[0]));

  document.addEventListener("click", (event) => {
    const tabButton = event.target.closest(".tab-button[data-tab-target]");
    if (tabButton) {
      const list = tabButton.closest("[data-tab-list]");
      const buttons = $$(".tab-button", list);
      const index = buttons.indexOf(tabButton);
      if (index >= 0) {
        setTabIndex(list, index);
      }
      return;
    }

    const chip = event.target.closest(".chip[data-filter-value]");
    if (chip && chip.closest("[data-filter-target]")) {
      const group = chip.closest("[data-filter-target]");
      $$(".chip", group).forEach((item) => item.classList.toggle("is-active", item === chip));
      applyFilterGroup(group);
      return;
    }

    const modelAction = event.target.closest(".model-action");
    if (modelAction) {
      handleModelAction(modelAction);
    }
  });
}

function startPolling() {
  const interval = POLL_INTERVALS[PAGE];
  if (!interval) {
    return;
  }
  setInterval(() => {
    if (!hasApiKey()) {
      return;
    }
    refreshCurrentPage();
  }, interval);
}

async function bootstrap() {
  highlightNav();
  initSearch();
  bindStaticEvents();
  hydrateUiSurface();
  updateFileMeta(null);
  await refreshCurrentPage();
  startPolling();
}

bootstrap();
