# MSR

MSR stands for `Multi Speaker Recognization`.

MSR is an offline-first multi-speaker transcription service rebuilt from the previous demo project. The service focuses on a stable local deployment story with `FastAPI + HTML/JS/CSS + uv + Docker`, explicit model `load/unload`, strict offline controls, and backward compatibility for the original `/transcribe/` interface.

## 1. 项目简介

MSR 的目标是把旧 demo 重构成一个真正适合长期维护的服务：

- 启动时不加载模型
- 运行时只接受本地模型路径
- 支持单密钥鉴权
- 支持模型 load / unload
- 支持资源监控
- 支持同步接口下的有界并行与有界排队
- 保留旧的 `/transcribe/` 响应结构，降低下游改造成本

## 2. 核心能力

- 离线优先：严格设置 `HF_HUB_OFFLINE`、`TRANSFORMERS_OFFLINE`、`HF_DATASETS_OFFLINE`、`MS_SDK_OFFLINE`
- 可插拔 ASR：支持 `FunASR` 和 `faster-whisper`
- 可切换 diarization：支持 `3D-Speaker` 和 `pyannote`
- KISS 的单体服务结构：不引入数据库和前端框架，运行态持久化使用本地文件
- 基础管理控制台：模型状态、资源监控、运行控制、任务观察、转写上传

## 3. 架构概览

目录结构：

- `src/msr/app/`：FastAPI 入口和 lifespan
- `src/msr/api/`：公共接口与管理接口
- `src/msr/core/`：配置、鉴权、日志、错误类型
- `src/msr/services/`：模型管理、资源监控、转写编排、音频处理
- `src/msr/backends/asr/`：FunASR / faster-whisper
- `src/msr/backends/diarization/`：3D-Speaker / pyannote
- `src/msr/backends/vad/`：WebRTC VAD
- `src/msr/web/`：控制台页面
- `config/`：应用配置和模型注册表
- `docs/`：架构与 roadmap
- `tools/doctor.py`：离线部署自检工具

更详细的设计说明见 [docs/architecture.md](docs/architecture.md)。

## 4. 文档维护约定

- 每次结束一轮开发、调试或验收，都要回看并更新 `README.md`
- `README.md` 负责记录当前可用能力、使用方式、真实环境验证结论
- `TODO.md` 负责记录当前状态、下一步事项、未完成验收和后续规划

## 5. 快速开始

### 本地开发

```bash
cd /Users/ray/Private/MCKJ-proj/MSR
uv sync --extra dev --extra default-runtime
export MSR_API_KEY="change-this"
uv run msr-api
```

说明：

- `default-runtime`：默认主链 `FunASR + 3D-Speaker + WebRTC VAD`
- `gpu-runtime`：包含默认链路和备选链路 `faster-whisper + pyannote`
- Linux + NVIDIA 环境默认锁定 `torch 2.10 / torchaudio 2.10 / torchvision 0.25`，避免 CUDA 12.8 驱动误拉到 `cu130`
- `FunASR` 默认启用 `disable_update=True`，避免离线部署时做版本检查

默认访问：

- 控制台：`http://127.0.0.1:8011/`
- 健康检查：`http://127.0.0.1:8011/health`

Linux + NVIDIA GPU 验收步骤见 [docs/linux-gpu-handoff.md](docs/linux-gpu-handoff.md)。
示例音频目录结构说明见 [samples/README.md](samples/README.md)。

### 先注册模型，再显式加载

模型注册来源于 `config/models.toml`。

如果要在联网环境先预热默认模型目录，可使用：

```bash
uv run python tools/bootstrap_models.py
uv run python tools/doctor.py
```

如果要把备选链路也纳入校验：

```bash
uv run python tools/doctor.py --include-alternates
```

启动服务后：

1. 调用 `GET /api/v1/models`
2. 调用 `POST /api/v1/models/asr/{model_id}/load`
3. 调用 `POST /api/v1/models/diarization/{model_id}/load`
4. 再调用 `POST /transcribe/`

### 最近一次真实环境验证

- 日期：`2026-04-15`
- 环境：`Linux x86_64 + NVIDIA GeForce RTX 3060 12GB + Driver 570.211.01 + CUDA 12.8`
- 主链：`FunASR + 3D-Speaker + WebRTC VAD`
- 结果：默认 ASR 与 diarization 已成功加载，`POST /transcribe/` 同步返回 `200`
- 样本：`16kHz / 单声道 / 17.19s` wav
- 性能：单条样本处理约 `4.03s`，处理速度约 `4.27x`
- 管理面：`runtime/tasks`、`runtime/active`、`system/resources` 已验证能返回真实任务与 GPU 信息
- 额外回归：`three-guys-record.mp3` 已验证按有效说话人分段返回，不再出现整段文本落到单一说话人的情况
- 离线稳定性：`FunASR` 更新检查提示已关闭，模型加载日志中不再出现版本检查提示

## 6. 配置说明

### `config/app.toml`

- `app.name`：项目名
- `app.service_name`：服务展示名
- `app.host` / `app.port`：监听地址
- `app.default_language`：默认语言
- `app.temp_dir`：临时目录
- `app.strict_offline`：是否强制离线环境变量
- `security.api_key`：默认开发密钥，可被 `MSR_API_KEY` 覆盖
- `web.resource_refresh_seconds`：前端轮询周期
- `runtime.max_parallel_tasks`：最大并行处理量
- `runtime.max_queued_tasks`：最大排队量
- `runtime.recent_task_limit`：最近任务摘要保留数
- `runtime.data_dir`：运行时覆盖配置和最近任务摘要的本地目录

### `config/models.toml`

模型注册表示例字段：

- `id`
- `kind`
- `backend`
- `local_path`
- `device`
- `enabled`
- `default`
- `options`

约束：

- `local_path` 必须是本地路径
- 禁止远程 repo id
- 禁止运行时 token

## 7. 模型准备

MSR 不负责下载模型。建议在仓库根目录下维护一个相对路径的 `models/` 目录，例如：

- `models/funasr/paraformer-zh`
- `models/faster-whisper/large-v3`
- `models/3d-speaker`
- `models/pyannote/speaker-diarization-community-1`

默认的 `config/models.toml` 也已经按这个相对目录结构配置好了。

## 8. API 说明

### `GET /health`

无鉴权，返回服务状态和当前已激活模型概览。

### `POST /transcribe/`

兼容旧 demo 的 multipart 上传接口。

请求：

- `audio`: 文件，支持 `wav/mp3/ogg/flac/m4a`

响应字段保持兼容：

- `task_id`
- `status`
- `transcripts`
- `speakers_info`
- `total_speakers`
- `audio_duration`
- `processing_time`
- `processing_speed`

当前转写响应还补了两条收敛规则：

- 只对有有效 ASR 文本片段的说话人生成 `speakers_info` 和 `total_speakers`
- 返回中的 `speaker_id` / `speaker_label` 会按有效说话人的出现顺序重新编号为 `0/A`、`1/B`、`2/C`

### 管理接口

- `GET /api/v1/auth/check`
- `GET /api/v1/models`
- `POST /api/v1/models/{kind}/{model_id}/load`
- `POST /api/v1/models/{kind}/{model_id}/unload`
- `GET /api/v1/runtime/active`
- `GET /api/v1/runtime/tasks`
- `GET /api/v1/runtime/limits`
- `POST /api/v1/runtime/limits`
- `GET /api/v1/system/resources`

除 `/health` 外，其余接口都需要 `X-API-Key`。

当服务繁忙且等待队列已满时，`POST /transcribe/` 会返回 `503`，错误体包含 `queue_full`、当前并行上限和排队占用情况。

## 9. GUI 说明

控制台页面是管理人员使用和管理接口的示例页，不是终端客户前台。当前提供：

- 当前激活模型
- 已注册模型与 load / unload
- CPU / RAM / GPU 资源监控
- 最大并行量 / 最大排队量 / 最近任务保留数
- 当前活跃任务、排队任务和最近完成任务摘要
- 音频上传转写

前端是纯 `HTML + JS + CSS`，不依赖框架，不使用 WebSocket，只做基础轮询。

## 10. 离线部署说明

运行时会设置：

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`
- `MS_SDK_OFFLINE=1`
- `AIMEETING_STRICT_OFFLINE=1`

建议的离线验收方式：

```bash
docker run --rm --gpus all --network none \
  -e MSR_API_KEY=change-this \
  -v "$(pwd)/models:/app/models" \
  -p 8011:8011 \
  msr-gpu-runtime:latest
```

说明：`Docker + --network none` 的完整离线验收仍是当前待完成事项，见 [TODO.md](TODO.md)。

## 11. 常见问题

### 为什么服务启动后不能直接转写？

因为 MSR 强制显式加载模型。先 load，再 transcribe。

### 为什么模型路径必须是本地目录？

这是为了彻底避免运行时联网和隐式下载。

### 为什么只做一个密钥鉴权？

因为 v1 目标是 KISS 和长期稳定，不引入账号系统、JWT 或数据库。

## 12. 开发 Roadmap

### Phase 0：设计冻结

- 冻结目录结构、配置格式、接口边界和 README 模板

### Phase 1：最小骨架

- 建立 FastAPI、配置系统、鉴权、模型管理和基础 GUI

### Phase 2：默认主链路

- 打通 `FunASR + 3D-Speaker + WebRTC VAD`
- 在同步 `/transcribe/` 下补齐内部并发控制、排队控制和任务观察

### Phase 3：备选后端

- 接入 `faster-whisper` 和 `pyannote`

### Phase 4：稳定性与交付

- 完成离线自检、错误治理、README 完整化和 GPU 镜像验收
- 规划 speaker registry，让跨音频复用已知说话人身份成为后续能力

详细版 roadmap 见 [docs/roadmap.md](docs/roadmap.md)。

Linux + GPU 交接步骤见 [docs/linux-gpu-handoff.md](docs/linux-gpu-handoff.md)。

## 13. 上游依赖项目链接

- FunASR：用于 ASR 后端
  GitHub: <https://github.com/modelscope/FunASR>
- faster-whisper：用于可切换的 Whisper ASR 后端
  GitHub: <https://github.com/SYSTRAN/faster-whisper>
- 3D-Speaker：用于中文友好的说话人 diarization
  GitHub: <https://github.com/modelscope/3D-Speaker>
- pyannote.audio：用于可切换的 diarization 后端
  GitHub: <https://github.com/pyannote/pyannote-audio>
- webrtcvad：用于固定的 VAD 切分
  GitHub: <https://github.com/wiseman/py-webrtcvad>
