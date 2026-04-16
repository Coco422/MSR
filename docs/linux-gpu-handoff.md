# Linux GPU Handoff

目标环境：`x86_64 Linux + NVIDIA GPU`

这份文档是给明天接手验证的同事用的，目标是尽快回答三个问题：

1. 默认模型是否能在真实环境离线加载
2. Docker GPU 镜像是否能在 `--network none` 下跑通
3. `/transcribe/` 在真实 GPU 上的速度和显存占用如何

## 1. 准备模型目录

在仓库根目录下准备：

- `models/funasr/paraformer-zh`
- `models/faster-whisper/large-v3`
- `models/3d-speaker`
- `models/pyannote/speaker-diarization-community-1`
- `models/qwen/qwen3-asr-0.6b`
- `models/qwen/qwen3-asr-1.7b`
- `models/qwen/qwen3-forced-aligner-0.6b`

如果实际目录不同，修改 `config/models.toml` 即可。

## 2. 本地依赖方式

```bash
cd /path/to/MSR
cp .env.example .env
export MSR_API_KEY="$(grep MSR_API_KEY .env | cut -d= -f2-)"
uv sync --extra dev --extra default-runtime
uv run python tools/bootstrap_models.py
uv run python tools/doctor.py
```

说明：

- `default-runtime` 会锁定 `torch 2.10 / torchaudio 2.10 / torchvision 0.25`，优先适配 CUDA 12.8 驱动
- `tools/doctor.py` 默认只检查主链默认模型

如果要同时验证备选链路 `faster-whisper + pyannote`，再补一轮：

```bash
uv sync --extra dev --extra gpu-runtime
uv run python tools/doctor.py --include-alternates
```

如果要验证 `Qwen3-ASR` 备选链路，使用独立 profile：

```bash
bash tools/runtime_env.sh setup qwen
bash tools/runtime_env.sh exec qwen python tools/doctor.py --include-qwen
```

## 3. 本地服务方式

```bash
uv run msr-api
```

另开终端：

```bash
curl -H "X-API-Key: $MSR_API_KEY" http://127.0.0.1:8011/api/v1/runtime/limits
curl -H "X-API-Key: $MSR_API_KEY" http://127.0.0.1:8011/api/v1/runtime/tasks

uv run python tools/smoke_api.py \
  --base-url http://127.0.0.1:8011 \
  --api-key "$MSR_API_KEY" \
  --asr-model faster-whisper-large-v3 \
  --diar-model 3dspeaker-default \
  --audio /absolute/path/to/sample.wav
```

如果要跑 `Qwen3-ASR`：

```bash
bash tools/runtime_env.sh run qwen
```

另开终端：

```bash
uv run python tools/smoke_api.py \
  --base-url http://127.0.0.1:8011 \
  --api-key "$MSR_API_KEY" \
  --asr-model qwen3-asr-0.6b \
  --diar-model 3dspeaker-default \
  --audio /absolute/path/to/sample.wav
```

## 4. Docker 方式

```bash
docker build -f Dockerfile.gpu -t msr-gpu-runtime:latest .
docker compose -f docker-compose.gpu.yml up -d msr
```

如果要从宿主机联调接口，继续执行：

```bash
uv run python tools/smoke_api.py \
  --base-url http://127.0.0.1:8011 \
  --api-key "$MSR_API_KEY" \
  --asr-model faster-whisper-large-v3 \
  --diar-model 3dspeaker-default \
  --audio /absolute/path/to/sample.wav
```

如果要做严格离线验收，不要再从宿主机直接 `curl`，而是运行：

```bash
bash tools/docker_offline_check.sh /absolute/path/to/sample.wav
```

说明：

- `tools/docker_offline_check.sh` 会在 `--network none` 容器内启动服务，并在容器内部完成 `/health`、模型 load 和 `/transcribe/` 的全链路自检
- `--network none` 下容器不会对宿主机暴露可访问网络，所以离线验收和普通联调需要分两种方式执行
- `Dockerfile.gpu` 当前为了尽快恢复服务，暂时复用本机已有的 `aimeeting-image-offline:latest` 作为恢复期基座，后续仍要收敛成独立可复现的 MSR 基础镜像

## 5. 必须记录的数据

- GPU 型号
- NVIDIA Driver 版本
- CUDA 版本
- Docker 版本
- `nvidia-smi` 输出摘要
- 默认链路是否成功
- `qwen3-asr-0.6b` 是否成功 load 并完成转写
- `qwen3-asr-1.7b` 是否能稳定 load；若失败，失败点在 load、转写还是 OOM
- 处理一段样例音频的总耗时
- 显存峰值
- `runtime/tasks` 中活跃任务数、排队任务数、最近任务摘要是否正常
- 当前 `max_parallel_tasks`、`max_queued_tasks` 是否符合预期
- 是否出现任何联网或模型校验行为

## 6. 如果默认链路失败

按顺序排查：

1. `tools/doctor.py` 是否通过
2. `config/models.toml` 路径是否和实际目录一致
3. 模型目录权限是否可读
4. 先单独尝试 `load`，不要直接测 `/transcribe/`
5. 再切换到 `faster-whisper + pyannote` 备选链路验证

## 7. Qwen3-ASR 额外说明

- 本轮只支持本地进程内 `vLLM`，不接 `vllm serve` 外部服务
- `ForcedAligner` 是 Qwen backend 的必需辅助模型，不做静默降级
- `qwen3-asr-0.6b` 是首轮必须跑通的验收模型
- `qwen3-asr-1.7b` 作为实验链路，不因为单卡 `3060 12GB` 不稳而阻塞本轮合入，但必须留下验收记录
