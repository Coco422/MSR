const API_KEY_STORAGE = "msr_api_key";

function getApiKey() {
  return localStorage.getItem(API_KEY_STORAGE) || "";
}

function setApiKey(value) {
  localStorage.setItem(API_KEY_STORAGE, value);
}

async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const apiKey = getApiKey();
  if (apiKey) {
    headers.set("X-API-Key", apiKey);
  }
  return fetch(path, { ...options, headers });
}

function pretty(data) {
  return JSON.stringify(data, null, 2);
}

async function refreshAuth() {
  const status = document.getElementById("authStatus");
  try {
    const response = await apiFetch("/api/v1/auth/check");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    status.textContent = "鉴权成功";
  } catch (error) {
    status.textContent = `鉴权失败: ${error.message}`;
  }
}

async function refreshRuntime() {
  const target = document.getElementById("runtimeState");
  try {
    const response = await apiFetch("/api/v1/runtime/active");
    target.textContent = pretty(await response.json());
  } catch (error) {
    target.textContent = `Failed to load runtime state: ${error.message}`;
  }
}

async function refreshResources() {
  const target = document.getElementById("resourceState");
  try {
    const response = await apiFetch("/api/v1/system/resources");
    target.textContent = pretty(await response.json());
  } catch (error) {
    target.textContent = `Failed to load resources: ${error.message}`;
  }
}

async function refreshModels() {
  const container = document.getElementById("modelsTable");
  try {
    const response = await apiFetch("/api/v1/models");
    const models = await response.json();
    container.innerHTML = renderModels(models);
    bindModelButtons();
  } catch (error) {
    container.innerHTML = `<p class="muted">加载模型列表失败: ${error.message}</p>`;
  }
}

function renderModels(models) {
  if (!Array.isArray(models) || models.length === 0) {
    return `<p class="muted">没有注册模型。</p>`;
  }

  const rows = models
    .map((model) => {
      const actionLabel = model.loaded ? "Unload" : "Load";
      const action = model.loaded ? "unload" : "load";
      return `
        <tr>
          <td><strong>${model.id}</strong><br /><span class="badge">${model.kind}</span></td>
          <td>${model.backend}</td>
          <td>${model.device}</td>
          <td>${model.path_exists ? "Yes" : "No"}</td>
          <td>${model.loaded ? "Loaded" : "Idle"}</td>
          <td>${model.local_path}</td>
          <td>
            <button class="ghost model-action" data-kind="${model.kind}" data-id="${model.id}" data-action="${action}">
              ${actionLabel}
            </button>
          </td>
        </tr>
      `;
    })
    .join("");

  return `
    <table class="models-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>Backend</th>
          <th>Device</th>
          <th>Path</th>
          <th>Status</th>
          <th>Local Path</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function bindModelButtons() {
  document.querySelectorAll(".model-action").forEach((button) => {
    button.addEventListener("click", async () => {
      const action = button.dataset.action;
      const kind = button.dataset.kind;
      const id = button.dataset.id;
      const path = `/api/v1/models/${kind}/${id}/${action}`;
      button.disabled = true;
      try {
        const response = await apiFetch(path, { method: "POST" });
        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail);
        }
      } catch (error) {
        alert(`模型操作失败: ${error.message}`);
      } finally {
        button.disabled = false;
        await Promise.all([refreshModels(), refreshRuntime()]);
      }
    });
  });
}

async function handleTranscribe(event) {
  event.preventDefault();
  const result = document.getElementById("transcribeResult");
  const input = document.getElementById("audioInput");
  const file = input.files[0];
  if (!file) {
    result.textContent = "请选择音频文件。";
    return;
  }

  const formData = new FormData();
  formData.append("audio", file);
  result.textContent = "转写中...";

  try {
    const response = await apiFetch("/transcribe/", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(pretty(payload));
    }
    result.textContent = pretty(payload);
  } catch (error) {
    result.textContent = `转写失败: ${error.message}`;
  }
}

function bootstrap() {
  const keyInput = document.getElementById("apiKey");
  keyInput.value = getApiKey();

  document.getElementById("saveKeyButton").addEventListener("click", async () => {
    setApiKey(keyInput.value.trim());
    await refreshAuth();
    await Promise.all([refreshModels(), refreshRuntime(), refreshResources()]);
  });

  document.getElementById("refreshModelsButton").addEventListener("click", refreshModels);
  document.getElementById("refreshRuntimeButton").addEventListener("click", refreshRuntime);
  document.getElementById("refreshResourcesButton").addEventListener("click", refreshResources);
  document.getElementById("transcribeForm").addEventListener("submit", handleTranscribe);

  refreshAuth();
  refreshModels();
  refreshRuntime();
  refreshResources();
  setInterval(refreshResources, 3000);
}

bootstrap();
