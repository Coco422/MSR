# Models Directory

将本地模型放在仓库根目录下的 `models/` 中，但不要提交真实模型文件。

建议目录结构：

- `models/funasr/paraformer-zh`
- `models/faster-whisper/large-v3`
- `models/3d-speaker`
- `models/pyannote/speaker-diarization-community-1`
- `models/qwen/qwen3-asr-0.6b`
- `models/qwen/qwen3-asr-1.7b`
- `models/qwen/qwen3-forced-aligner-0.6b`

`config/models.toml` 默认就按这套相对路径解析。

当前默认服务链优先使用：

- `models/faster-whisper/large-v3`
- `models/3d-speaker`

Docker 运行时建议：

- 将宿主机的 `./models` 挂载到容器的 `/app/models`
- 将宿主机的 `./data` 挂载到容器的 `/app/data`
- 不要把模型 bake 进 Git 仓库
