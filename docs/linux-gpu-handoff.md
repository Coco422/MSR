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

如果实际目录不同，修改 `config/models.toml` 即可。

## 2. 本地依赖方式

```bash
cd /path/to/MSR
cp .env.example .env
export MSR_API_KEY="$(grep MSR_API_KEY .env | cut -d= -f2-)"
uv sync --extra dev --extra gpu-runtime
uv run python tools/doctor.py
```

## 3. 本地服务方式

```bash
uv run msr-api
```

另开终端：

```bash
uv run python tools/smoke_api.py \
  --base-url http://127.0.0.1:8011 \
  --api-key "$MSR_API_KEY" \
  --asr-model funasr-paraformer-zh \
  --diar-model 3dspeaker-default \
  --audio /absolute/path/to/sample.wav
```

## 4. Docker 方式

```bash
docker build -f Dockerfile.gpu -t msr-gpu-runtime:latest .
docker run --rm --gpus all --network none \
  -e MSR_API_KEY="$MSR_API_KEY" \
  -v "$(pwd)/config:/app/config:ro" \
  -v "$(pwd)/models:/app/models:ro" \
  -p 8011:8011 \
  msr-gpu-runtime:latest
```

另开终端执行 smoke：

```bash
uv run python tools/smoke_api.py \
  --base-url http://127.0.0.1:8011 \
  --api-key "$MSR_API_KEY" \
  --asr-model funasr-paraformer-zh \
  --diar-model 3dspeaker-default \
  --audio /absolute/path/to/sample.wav
```

## 5. 必须记录的数据

- GPU 型号
- NVIDIA Driver 版本
- CUDA 版本
- Docker 版本
- `nvidia-smi` 输出摘要
- 默认链路是否成功
- 处理一段样例音频的总耗时
- 显存峰值
- 是否出现任何联网或模型校验行为

## 6. 如果默认链路失败

按顺序排查：

1. `tools/doctor.py` 是否通过
2. `config/models.toml` 路径是否和实际目录一致
3. 模型目录权限是否可读
4. 先单独尝试 `load`，不要直接测 `/transcribe/`
5. 再切换到 `faster-whisper + pyannote` 备选链路验证
