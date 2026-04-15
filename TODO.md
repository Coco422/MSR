# MSR TODO

更新时间：2026-04-15

## 当前状态

- [x] 根目录 `MSR` 工程初始化完成
- [x] `FastAPI + HTML/JS/CSS + uv + Docker` 基础骨架完成
- [x] `/transcribe/` 与 `/health` 兼容接口完成
- [x] 管理接口完成：`auth/check`、`models`、`runtime/active`、`system/resources`
- [x] ASR 后端骨架完成：`FunASR`、`faster-whisper`
- [x] diarization 后端骨架完成：`3D-Speaker`、`pyannote`
- [x] 默认主链恢复：`FunASR + 3D-Speaker + WebRTC VAD`
- [x] 同步 `/transcribe/` 已接入有界并行与有界排队
- [x] 管理接口补齐：`runtime/tasks`、`runtime/limits`
- [x] GUI 已补任务观察和运行控制面板
- [x] 模型目录切换为仓库内相对路径 `models/`
- [x] README、architecture、roadmap、doctor 已补齐
- [x] 基础测试通过：`uv run --extra dev pytest`

## 当前待推进

- [x] 补 `TODO.md`
- [x] 补 `.env.example`
- [x] 补 Linux GPU 交接文档
- [x] 补 API smoke 脚本，便于明天同事拿真实模型直接验证
- [ ] 在 README 中加入“明天 Linux 验收步骤”入口
- [ ] 准备一个最小示例音频 `samples/` 目录结构说明
- [ ] 在真实 GPU 环境验证默认链路的吞吐、显存峰值和排队表现
- [ ] 评估是否需要把当前任务摘要导出到日志或 metrics 系统

## 明天交给 Linux + NVIDIA GPU 同事

### P0：必须完成

- [ ] 准备真实模型目录到 `models/`
- [ ] 按 `config/models.toml` 对齐模型实际路径
- [ ] 在 x86 Linux + NVIDIA GPU 环境执行 `uv sync --extra gpu-runtime`
- [ ] 执行 `docker build -f Dockerfile.gpu -t msr-gpu-runtime:latest .`
- [ ] 用 `docker run --gpus all --network none` 启动服务
- [ ] 调用管理接口加载默认模型
- [ ] 用真实音频跑通 `/transcribe/`
- [ ] 记录 GPU 显存占用、处理速度、错误日志

### P1：建议完成

- [ ] 验证 `FunASR + 3D-Speaker` 默认链路
- [ ] 验证 `faster-whisper + pyannote` 备选链路
- [ ] 核对各模型是否仍有隐式联网行为
- [ ] 验证容器内模型路径、权限、挂载方式
- [ ] 补一份真实环境依赖版本记录：CUDA、驱动、Docker、Python

### P2：后续收敛

- [ ] 根据真实 GPU 环境决定是否固定 Python 3.11
- [ ] 根据真实模型准备方式决定是否保留 `gpu-runtime` 全量 extra
- [ ] 确认是否需要拆 `Dockerfile.gpu` 和 `Dockerfile.gpu.slim`
- [ ] 评估 `/transcribe/` 是否要补 `speaker_count_hint` 等可选参数
- [ ] 设计 speaker registry：保存说话人声纹特征并支持跨音频识别同一人
- [ ] 确认 speaker registry 的存储格式、人工命名流程和 UNKNOWN 自动创建策略

## 明天的最小验收标准

1. `GET /health` 正常
2. `GET /api/v1/models` 正常
3. 默认 ASR 和 diarization 能成功 `load`
4. `/transcribe/` 返回兼容旧结构的 JSON
5. 全流程在 `--network none` 下可运行
6. 同事能给出一份实际 GPU 运行数据

## 已经为明天准备好的工具

- `tools/doctor.py`
- `tools/smoke_api.py`
- `docs/linux-gpu-handoff.md`
- `.env.example`
