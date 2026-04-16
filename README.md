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
- 支持异步任务提交、状态查询和结果拉取
- 保留旧的 `/transcribe/` 响应结构，降低下游改造成本

## 2. 核心能力

- 离线优先：严格设置 `HF_HUB_OFFLINE`、`TRANSFORMERS_OFFLINE`、`HF_DATASETS_OFFLINE`、`MS_SDK_OFFLINE`
- 可插拔 ASR：支持 `FunASR`、`faster-whisper`，以及实验性 `Qwen3-ASR + 本地 vLLM + ForcedAligner`
- 可切换 diarization：支持 `3D-Speaker` 和 `pyannote`
- KISS 的单体服务结构：不引入数据库和前端框架，运行态持久化使用本地文件
- 基础管理控制台：多页后台式总览、模型状态、资源监控、运行控制、任务观察、转写上传

## 3. 架构概览

目录结构：

- `src/msr/app/`：FastAPI 入口和 lifespan
- `src/msr/api/`：公共接口与管理接口
- `src/msr/core/`：配置、鉴权、日志、错误类型
- `src/msr/services/`：模型管理、资源监控、转写编排、音频处理
- `src/msr/backends/asr/`：FunASR / faster-whisper / Qwen3-ASR
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

## 5. 发布标记

- 首个上线版本使用注释标签 `v0.1.0`
- 后续版本继续沿用语义化版本标签，便于在 GitHub 与公司私有仓库之间同步发布记录

## 6. 快速开始

### 本地开发

```bash
cd /path/to/MSR
uv sync --extra dev --extra default-runtime
export MSR_API_KEY="change-me"
uv run msr-api
```

说明：

- `default-runtime`：默认主链 `faster-whisper + 3D-Speaker + WebRTC VAD`
- `gpu-runtime`：保留为全量实验依赖集，但当前不再作为默认容器或默认 venv 方案
- 当前默认链已切到 `faster-whisper + 3D-Speaker`，原因是它在现阶段的真实测试里更稳、更准，也更适合先恢复服务
- Linux + NVIDIA 环境默认锁定 `torch 2.10 / torchaudio 2.10 / torchvision 0.25`，避免 CUDA 12.8 驱动误拉到 `cu130`
- `pyannote-community-1` 备选链要求 `pyannote.audio 4.x`
- 当前建议：默认服务直接使用 `uv sync --extra dev --extra default-runtime` 或 `bash tools/runtime_env.sh setup default`
- `FunASR` 默认启用 `disable_update=True`，避免离线部署时做版本检查
- `Qwen3-ASR` 本轮作为 admin-only 备选链，不替换默认主链；建议通过独立 `qwen` profile 运行

默认访问：

- 控制台：`http://127.0.0.1:8011/`
- 模型管理：`http://127.0.0.1:8011/models`
- 运行控制：`http://127.0.0.1:8011/runtime`
- 转写示例：`http://127.0.0.1:8011/transcribe`
- 健康检查：`http://127.0.0.1:8011/health`

Linux + NVIDIA GPU 验收步骤见 [docs/linux-gpu-handoff.md](docs/linux-gpu-handoff.md)。
示例音频目录结构说明见 [samples/README.md](samples/README.md)。

代码目录采用标准 `src layout`：业务包位于 `src/msr/`。
这层目录本身不是异常点；独立 venv 启动失败通常是因为 profile 环境没有装入项目包，或启动时没有把 `src/` 纳入 Python 搜索路径。

### 独立 venv 切换脚本

如果你希望把默认链、`pyannote` 准确率优先链和 `Qwen3-ASR` 实验链彻底隔离，可直接使用：

```bash
bash tools/runtime_env.sh setup default
bash tools/runtime_env.sh setup pyannote
bash tools/runtime_env.sh setup qwen
```

说明：

- `default` 环境对应 `faster-whisper + 3D-Speaker + WebRTC VAD`
- `pyannote` 环境对应 `faster-whisper + pyannote`
- `qwen` 环境对应 `Qwen3-ASR + 3D-Speaker + 本地 vLLM + ForcedAligner`
- 之所以拆多套 venv，是因为 `speakerlab`、`pyannote 4.x` 与 `Qwen3-ASR + vLLM` 的依赖栈都偏重，分开更利于长期维护和切换
- `qwen` profile 额外固定 `qwen-asr==0.0.6` 与 `vllm==0.14.0`，并会一起安装 `speakerlab`
- `Qwen3-ASR` 的时间戳依赖 `ForcedAligner`，当前在 MSR 内部作为 Qwen backend 的必需辅助模型加载
- `Qwen3-ASR` 默认会额外约束 `max_model_len`，避免 vLLM 按模型原始 `65536` 上下文在 `12GB` 级显卡上起过大的 KV cache
- 如果要跑 `faster-whisper + pyannote`，不要直接执行 `uv run msr-api`
- 如果要跑 `Qwen3-ASR`，同样不要直接执行 `uv run msr-api`
- `uv run msr-api` 使用的是仓库默认 `.venv`，更适合当前默认链
- `Qwen3-ASR` 备选链请使用 `bash tools/runtime_env.sh run qwen`
- `tools/runtime_env.sh run/exec` 现在会显式切到对应 profile 的 Python，并补上 `PYTHONPATH=src`，不再依赖 profile 内一定存在 `msr-api` console script
- `tools/runtime_env.sh setup ...` 现在可重复执行；若 profile venv 已存在会直接复用，不再弹交互确认

常用命令：

```bash
bash tools/runtime_env.sh run default
bash tools/runtime_env.sh run pyannote
bash tools/runtime_env.sh run qwen
bash tools/runtime_env.sh exec pyannote python tools/doctor.py --include-alternates
bash tools/runtime_env.sh exec qwen python tools/doctor.py --include-qwen
```

调试建议：

- 启动日志现在会打印当前 `python`、`prefix`、`venv`、`cwd`
- 模型加载日志会打印模型类型、后端、设备、路径，以及缺失依赖时的独立环境提示
- 如果只是想确认当前服务到底跑在哪个解释器里，优先看启动时的 `Application startup ... python=... venv=...`

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

如果要把 `Qwen3-ASR` 备选链路也纳入校验：

```bash
uv run python tools/doctor.py --include-qwen
```

如果要在联网环境把备选模型也一并预热：

```bash
export HF_TOKEN="your-token"
uv run python tools/bootstrap_models.py --include-alternates --hf-token "$HF_TOKEN"
uv run python tools/bootstrap_models.py --include-qwen
```

说明：

- `faster-whisper-large-v3` 会下载到 `models/faster-whisper/large-v3`
- `pyannote-community-1` 会下载到 `models/pyannote/speaker-diarization-community-1`
- `qwen3-asr-0.6b` 会下载到 `models/qwen/qwen3-asr-0.6b`
- `qwen3-asr-1.7b` 会下载到 `models/qwen/qwen3-asr-1.7b`
- `Qwen3-ForcedAligner-0.6B` 会下载到 `models/qwen/qwen3-forced-aligner-0.6b`
- `pyannote` 需要先在 Hugging Face 接受 gated model 条件，再使用 token 下载
- 当前 `speakerlab` 与 `pyannote 4.x` 仍存在上游 `numpy` 约束冲突，因此备选链在本机是通过 `uv pip install` 方式补齐，而不是直接回写到 `uv lock`
- `Qwen3-ASR` 当前按本地目录加载，不在服务启动或模型加载时做远程拉取
- `tools/bootstrap_models.py` 会先遍历当前配置里的默认模型，再按参数补备选模型；若当前环境缺少某条链的下载依赖，会打印 `[SKIP]` 并继续处理其他模型，不再整体中断

启动服务后：

1. 调用 `GET /api/v1/models`
2. 调用 `POST /api/v1/models/asr/{model_id}/load`
3. 调用 `POST /api/v1/models/diarization/{model_id}/load`
4. 再调用 `POST /transcribe/`

### 最近一次真实环境验证

- 日期：`2026-04-16`
- 环境：`Linux x86_64 + NVIDIA GeForce RTX 3060 12GB + Driver 570.211.01 + CUDA 12.8`
- 主链：`faster-whisper + 3D-Speaker + WebRTC VAD`
- 结果：默认 ASR 与 diarization 已成功加载，`POST /transcribe/` 同步返回 `200`
- 样本：`16kHz / 单声道 / 17.19s` wav
- 性能：`32s` 样本在 `faster-whisper large-v3 + 3D-Speaker` 链路下，ASR 约 `3.2s`
- 管理面：`runtime/tasks`、`runtime/active`、`system/resources` 已验证能返回真实任务与 GPU 信息
- Docker 恢复：`docker compose -f docker-compose.gpu.yml build msr && docker compose -f docker-compose.gpu.yml up -d msr` 已在本机通过
- Docker 验证：容器内已成功加载 `faster-whisper-large-v3` 与 `3dspeaker-default`，并对 `samples/smoke/smoke_zh_3spk.mp3` 返回 `200`
- Docker 离线验收：`bash tools/docker_offline_check.sh` 已在本机通过，`--network none` 下已完成 `/health`、模型 load 和 `/transcribe/` 全链路验证
- Docker 性能：同一 `32s` mp3 在当前容器链路下总耗时约 `18.9s`
- Docker 基座：`Dockerfile.gpu` 当前暂时复用本机已有的 `aimeeting-image-offline:latest` 作为恢复期本地基座，再补装 `faster-whisper`
- 额外回归：`three-guys-record.mp3` 已验证按有效说话人分段返回，不再出现整段文本落到单一说话人的情况
- 边界修复：说话人回填已升级为“词级时间戳归属后再重新拼句”，可把跨 speaker 边界的尾字头字拆开重分配
- 管理面修复：模型 load / unload 过程已改为后台线程执行，并释放模型管理锁，避免单次加载拖住其它管理读请求
- 备选链验证：`pyannote-community-1` 已下载到本地并完成真实推理
- 备选链性能：`pyannote community-1` 加载约 `11.0s`、同样本 diarization 约 `3.1s`
- 兼容处理：`pyannote` 当前通过本地 `config.yaml` 加载，并在推理时直接喂入内存 waveform，绕开当前环境下 `torchcodec` 对系统 FFmpeg 的兼容问题

### Qwen3-ASR 当前状态

- 日期：`2026-04-16`
- 代码集成：已接入 `qwen_asr` backend，固定走 `Qwen3ASRModel.LLM(...)` 本地进程内 vLLM
- 内部能力：`ASRBackend` 已新增 `transcribe_many(...)`，Qwen 链路会对单任务内多个 VAD clip 做批量推理
- 时间戳：`ForcedAligner` 作为 Qwen backend 的必需辅助模型，在 load 阶段一并初始化；缺路径、缺依赖或加载失败会直接 load 失败
- 模型注册：已加入 `qwen3-asr-0.6b`、`qwen3-asr-1.7b` 和 `models/qwen/qwen3-forced-aligner-0.6b` 目录约定
- 启动资源：当前默认把 `qwen3-asr-0.6b` 的 `max_model_len` 收敛到 `16384`、`qwen3-asr-1.7b` 收敛到 `8192`，更贴合我们 `20s` clip 的服务场景
- 环境脚本：已新增 `tools/runtime_env.sh setup/run/exec qwen`
- 工具链：`tools/bootstrap_models.py --include-qwen` 与 `tools/doctor.py --include-qwen` 已就位
- 当前结论：代码级与测试级接入已完成，`qwen` profile 也已开始实际安装校验；`RTX 3060 12GB` 上 `0.6B` 真机转写、显存峰值和 `1.7B` 稳定性仍待补验，不把它写成已完成验收

## 7. 配置说明

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
- `runtime.recent_task_limit`：最近任务摘要和异步结果保留数，默认 `50`
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

## 8. 模型准备

MSR 不负责下载模型。建议在仓库根目录下维护一个相对路径的 `models/` 目录，例如：

- `models/funasr/paraformer-zh`
- `models/faster-whisper/large-v3`
- `models/3d-speaker`
- `models/pyannote/speaker-diarization-community-1`
- `models/qwen/qwen3-asr-0.6b`
- `models/qwen/qwen3-asr-1.7b`
- `models/qwen/qwen3-forced-aligner-0.6b`

默认的 `config/models.toml` 也已经按这个相对目录结构配置好了。

推荐的联网预热命令：

```bash
uv run python tools/bootstrap_models.py
export HF_TOKEN="your-token"
uv run python tools/bootstrap_models.py --include-alternates --hf-token "$HF_TOKEN"
uv run python tools/bootstrap_models.py --include-qwen
```

## 9. API 说明

### `GET /health`

无鉴权，返回服务状态和当前已激活模型概览。

### `POST /transcribe/`

兼容旧 demo 的 multipart 上传接口。

定位说明：

- 该接口保持同步返回，便于兼容旧上游
- 长音频或高并发场景更建议使用异步任务接口，避免被网关、浏览器或上游 SDK 超时截断

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

当前转写响应还补了几条收敛规则：

- 只对有有效 ASR 文本片段的说话人生成 `speakers_info` 和 `total_speakers`
- 返回中的 `speaker_id` / `speaker_label` 会按有效说话人的出现顺序重新编号为 `0/A`、`1/B`、`2/C`
- 当当前 ASR 后端提供词级或 token 级时间戳时，服务会先按时间戳与 diarization 边界做归属，再拼回最终文本段，减少句尾串到相邻说话人的情况

当同步转写发生显存不足时，服务会返回 `507`，错误体带 `cuda_oom` 机器码和建议说明。

### 异步任务接口

- `POST /api/v1/transcriptions/submit`
- `GET /api/v1/transcriptions/{task_id}`
- `GET /api/v1/transcriptions/{task_id}/result`

行为约定：

- `submit` 只负责接单入队，立即返回 `task_id`
- `status` 返回当前阶段、排队等待时间、运行耗时和结果是否可取
- `result` 在任务未完成时返回 `202`，失败时返回 `409`，成功时返回与 `/transcribe/` 相同的兼容结果结构
- 最近任务摘要和异步结果文件都会按 `runtime.recent_task_limit` 自动裁剪，默认只保留 `50` 条

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

## 10. GUI 说明

控制台页面是管理人员使用和管理接口的示例页，不是终端客户前台。当前提供：

- 多页后台式总览中心、模型管理、运行控制、转写示例
- 当前激活模型
- 已注册模型与 load / unload
- CPU / RAM / GPU 资源监控
- 最大并行量 / 最大排队量 / 最近任务保留数
- 当前活跃任务、排队任务和最近完成任务摘要
- 音频上传转写

当前前端特征：

- 左侧固定导航 + 主内容独立滚动，适合高信息密度管理视图
- 顶部搜索框、页面级操作按钮、导航高亮、基础筛选按钮和 Tab 切换
- 模型加载 / 卸载、运行控制保存、同步转写提交都带等待响应的忙碌态
- 模型页会显式区分 `已禁用`、`路径缺失`、`加载中`、`卸载中` 四类按钮状态；任一模型操作开始后，其它模型按钮会暂时锁定，避免重复点击造成“看起来没反应”
- 长音频生产接入建议优先走异步任务接口，控制台中的同步页仅保留为兼容联调示例
- 页面内成功 / 失败提示条与最近刷新时间会在操作完成后同步更新，不再依赖弹窗提示
- 原始 JSON 已收敛为仪表盘、状态卡片、任务列表、时间线和说话人统计

前端是纯 `HTML + JS + CSS`，不依赖框架，不使用 WebSocket，只做基础轮询。

## 11. 离线部署说明

运行时会设置：

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`
- `MS_SDK_OFFLINE=1`
- `AIMEETING_STRICT_OFFLINE=1`

建议的容器验证方式：

```bash
docker compose -f docker-compose.gpu.yml build msr
docker compose -f docker-compose.gpu.yml up -d msr
```

如果要做严格离线验收，直接运行：

```bash
bash tools/docker_offline_check.sh
```

如果要指定自己的音频样本：

```bash
bash tools/docker_offline_check.sh /absolute/path/to/sample.wav
```

说明：

- `tools/docker_offline_check.sh` 会在 `--network none` 的容器里自检 `/health`、默认模型 load 和同步 `/transcribe/`
- `--network none` 下不能直接从宿主机访问容器端口，所以离线验收不要再用宿主机 `curl` 或浏览器联调
- `Dockerfile.gpu` 当前为了先恢复服务，暂时复用本机已有的 `aimeeting-image-offline:latest` 作为本地基座，再补装 `faster-whisper`
- 这条 Docker 主链已经在当前机器上完成真实转写验收
- `Docker + --network none` 的完整离线验收已经在当前机器上通过
- 后续仍需要把这个临时基座收敛成更轻、更可复现的独立 MSR 镜像

## 12. 常见问题

### 为什么服务启动后不能直接转写？

因为 MSR 强制显式加载模型。先 load，再 transcribe。

### 为什么模型路径必须是本地目录？

这是为了彻底避免运行时联网和隐式下载。

### 为什么只做一个密钥鉴权？

因为 v1 目标是 KISS 和长期稳定，不引入账号系统、JWT 或数据库。

## 13. 开发 Roadmap

### Phase 0：设计冻结

- 冻结目录结构、配置格式、接口边界和 README 模板

### Phase 1：最小骨架

- 建立 FastAPI、配置系统、鉴权、模型管理和基础 GUI

### Phase 2：默认主链路

- 打通 `faster-whisper + 3D-Speaker + WebRTC VAD`
- 在同步 `/transcribe/` 下补齐内部并发控制、排队控制和任务观察

### Phase 3：备选后端

- 接入 `faster-whisper` 和 `pyannote`
- 接入 `Qwen3-ASR + 本地 vLLM + ForcedAligner`

### Phase 4：稳定性与交付

- 完成离线自检、错误治理、README 完整化和 GPU 镜像验收
- 规划 speaker registry，让跨音频复用已知说话人身份成为后续能力
- 继续验证 `Qwen3-ASR 0.6B/1.7B` 在 `RTX 3060 12GB` 上的加载稳定性、显存峰值和对齐效果

详细版 roadmap 见 [docs/roadmap.md](docs/roadmap.md)。

Linux + GPU 交接步骤见 [docs/linux-gpu-handoff.md](docs/linux-gpu-handoff.md)。
speaker registry 设计草案见 [docs/speaker-registry-design.md](docs/speaker-registry-design.md)。

## 14. 上游依赖项目链接

- FunASR：用于 ASR 后端
  GitHub: <https://github.com/modelscope/FunASR>
- faster-whisper：用于可切换的 Whisper ASR 后端
  GitHub: <https://github.com/SYSTRAN/faster-whisper>
- Qwen3-ASR：用于实验性本地 vLLM ASR 备选链
  GitHub: <https://github.com/QwenLM/Qwen3-ASR>
- vLLM：用于 Qwen3-ASR 的本地进程内推理引擎
  GitHub: <https://github.com/vllm-project/vllm>
- 3D-Speaker：用于中文友好的说话人 diarization
  GitHub: <https://github.com/modelscope/3D-Speaker>
- pyannote.audio：用于可切换的 diarization 后端
  GitHub: <https://github.com/pyannote/pyannote-audio>
- webrtcvad：用于固定的 VAD 切分
  GitHub: <https://github.com/wiseman/py-webrtcvad>
