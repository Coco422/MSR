# MSR TODO

更新时间：2026-04-16

## 文档约定

- 每次结束一轮开发、调试或验收时，必须 review 并更新 `README.md`
- 每次结束一轮开发、调试或验收时，必须 review 并更新 `TODO.md`

## 当前状态

- [x] 根目录 `MSR` 工程初始化完成
- [x] `FastAPI + HTML/JS/CSS + uv + Docker` 基础骨架完成
- [x] `/transcribe/` 与 `/health` 兼容接口完成
- [x] 异步任务接口完成：提交、状态查询、结果拉取
- [x] 管理接口完成：`auth/check`、`models`、`runtime/active`、`system/resources`
- [x] ASR 后端骨架完成：`FunASR`、`faster-whisper`
- [x] ASR 后端已补第三套实验链：`Qwen3-ASR + 本地 vLLM + ForcedAligner`
- [x] diarization 后端骨架完成：`3D-Speaker`、`pyannote`
- [x] 默认主链已切到：`faster-whisper + 3D-Speaker + WebRTC VAD`
- [x] 同步 `/transcribe/` 已接入有界并行与有界排队
- [x] 管理接口补齐：`runtime/tasks`、`runtime/limits`
- [x] GUI 已补任务观察和运行控制面板
- [x] GUI 已升级为多页后台式管理台：总览中心、模型管理、运行控制、转写示例
- [x] GUI 已补页面内成功/失败提示条与最近刷新时间，不再依赖弹窗反馈
- [x] 修复模型页按钮反馈问题：禁用模型不可点击，加载/卸载时显式显示忙碌态，并临时锁定其它模型按钮
- [x] 模型目录切换为仓库内相对路径 `models/`
- [x] `recent_tasks` 与异步结果文件已按 `recent_task_limit` 自动裁剪，当前默认保留 `50` 条
- [x] README、architecture、roadmap、doctor 已补齐
- [x] 首个上线版本已发布规划完成，统一使用 `v0.1.0` 作为首发标签
- [x] 基础测试通过：`uv run --extra dev pytest`
- [x] 在真实 Linux + NVIDIA GPU 环境验证默认链路可用
- [x] 在真实 Linux + NVIDIA GPU 环境验证备选链 `faster-whisper + pyannote` 可加载并完成真实推理
- [x] 默认 GPU 依赖已对齐到 CUDA 12.8 兼容的 PyTorch 2.10 系列
- [x] `3D-Speaker` 默认链运行时依赖已补齐并可加载
- [x] 主链 smoke test 已通过：`17.19s` wav 同步转写约 `4.03s`
- [x] 管理接口已在真机验证：`runtime/active`、`runtime/tasks`、`system/resources`
- [x] 用户侧准确率体验反馈：`faster-whisper` 当前优于 `paraformer`
- [x] 修复 `FunASR` 词级时间戳毫秒/秒解析错误，避免出现 `9:30 -> 0:30` 这类错时间
- [x] 修复 `FunASR` 词级时间戳整段合并问题，当前可按停顿切成多个文本片段
- [x] 默认禁用 `FunASR` 更新检查，避免离线场景下出现版本探测日志和潜在外网依赖
- [x] 转写响应中的有效说话人已重新编号为连续 `0/A`、`1/B`、`2/C`
- [x] 无有效 ASR 文本的说话人已不再计入 `speakers_info` 和 `total_speakers`
- [x] `three-guys-record.mp3` 已验证不再出现“整段文本只落到单一说话人”的问题
- [x] 说话人回填已升级为词级时间戳归属，跨 speaker 边界的尾字头字可拆开重分配
- [x] `faster-whisper-large-v3` 已下载到 `models/faster-whisper/large-v3`
- [x] `pyannote-community-1` 已下载到 `models/pyannote/speaker-diarization-community-1`
- [x] `pyannote` 后端已兼容本地 `config.yaml` 加载、4.x `DiarizeOutput` 返回结构和内存 waveform 输入
- [x] 已提供多 profile venv 切换脚本：默认链环境 / `faster-whisper + pyannote` 环境 / `Qwen3-ASR + vLLM` 环境
- [x] 多 profile 的 `setup/run/exec` 已修复并验证，profile 启动不再依赖缺失的 `msr-api` console script
- [x] `runtime_env.sh setup ...` 已改为可重复执行，不再因已有 venv 弹交互确认
- [x] 启动阶段、模型加载阶段、任务阶段日志已补强，可直接看到当前解释器/venv、模型路径和阶段流转
- [x] `faster-whisper` / `pyannote` 缺依赖时会返回更明确的环境切换提示，不再只报裸 `ModuleNotFoundError`
- [x] 模型 load / unload 已改为后台线程执行，并释放长时占用的模型管理锁，避免单次加载阻塞其它管理读请求
- [x] `ASRBackend` 已新增 `transcribe_many(...)` 默认实现，现有 `FunASR` / `faster-whisper` 不受影响
- [x] 转写编排已切到 clip 批量 ASR 接口，Qwen 链路可对单任务多个 VAD clip 做批推理
- [x] Qwen backend 已接通 `ForcedAligner -> TimedToken` 映射，并复用现有 speaker token 级回填逻辑
- [x] `config/models.toml` 已加入 `qwen3-asr-0.6b`、`qwen3-asr-1.7b` 与 `forced_aligner_path`
- [x] Qwen 默认启动参数已收敛：限制 `max_model_len` 并降低默认批量，避免 `12GB` 级显卡因 vLLM KV cache 过大而在 load 阶段失败
- [x] `tools/runtime_env.sh` 已加入独立 `qwen` profile，固定 `qwen-asr==0.0.6` 与 `vllm==0.14.0`
- [x] `qwen` profile 已补齐 `speakerlab` 与 3D-Speaker 依赖，避免 Qwen 环境无法加载默认 diarization
- [x] `tools/bootstrap_models.py --include-qwen` 与 `tools/doctor.py --include-qwen` 已补齐
- [x] `bootstrap_models.py` 已改为按缺失下载依赖自动跳过，不再因无关后端缺包导致整次预热中断
- [x] README、architecture、roadmap、Linux GPU handoff 已加入 Qwen vLLM 备选链说明
- [x] 已新增 `docs/speaker-registry-design.md`，开始收敛 speaker registry 的文件存储、匹配流程和后续 API 草案
- [x] 已补 `tools/docker_offline_check.sh`，可在 `--network none` 容器内完成健康检查、模型装载和同步转写自检

## 当前待推进

- [x] 补 `TODO.md`
- [x] 补 `.env.example`
- [x] 补 Linux GPU 交接文档
- [x] 补 API smoke 脚本，便于明天同事拿真实模型直接验证
- [x] 在 README 中加入“明天 Linux 验收步骤”入口
- [x] 准备一个最小示例音频 `samples/` 目录结构说明
- [ ] 在真实 GPU 环境验证默认链路的吞吐、显存峰值和排队表现
- [x] 基于当前真实准确率体验，先把 `faster-whisper + 3D-Speaker` 定为默认链
- [ ] 在 `RTX 3060 12GB` 上完成 `qwen3-asr-0.6b + 3D-Speaker` 真机转写验收，记录加载耗时、转写耗时、显存峰值和长音频稳定性
- [ ] 把 `qwen3-asr-1.7b` 作为实验链路做一次真机验证，明确是否因显存或稳定性原因保持非推荐状态
- [ ] 基于真实 Qwen 验证结果，继续收敛 `max_model_len`、`max_inference_batch_size` 与 `gpu_memory_utilization` 的默认值
- [ ] 对比同一批样本上的 `FunASR`、`faster-whisper`、`Qwen3-ASR` 主观准确率和 speaker 对齐效果
- [ ] 评估是否需要把当前任务摘要导出到日志或 metrics 系统
- [ ] 记录默认链在真实 GPU 环境下的显存峰值，而不只是加载后常驻占用
- [ ] 补一份“本轮真机验证纪要”到 `docs/`，沉淀环境和问题修复点
- [ ] 继续优化 `faster-whisper + 3D-Speaker` 在多人短句场景下的切句质量，减少“嗯/哦”等极短片段
- [ ] 评估是否需要为 Qwen 链路单独加更严格的 clip batch 上限，避免批量 VAD 场景下 vLLM 或 aligner 触发显存抖动
- [ ] 评估是否要把同步示例页也升级成“同步/异步双模式”联调页面
- [ ] 继续收敛默认链加载日志，评估是否要压低 `modelscope` / `datasets` 的纯信息级噪声
- [ ] 评估是否要把启动日志再细分为 `INFO` / `DEBUG` 两档，避免生产期与调试期日志密度冲突
- [ ] 收敛 `pyannote.audio` 导入时的 `torchcodec` 警告，决定是补系统 FFmpeg 兼容层还是显式屏蔽无害告警
- [ ] 梳理 `speakerlab` 与 `pyannote 4.x` 的 `numpy` 依赖冲突，决定是否拆分 extra、拆容器或保留手工补装方案
- [ ] 继续优化 VAD 与 speaker 边界协同，避免整段会话被并成单个超长 VAD 段
- [ ] 继续打磨前端细节：补长列表虚拟化与更细的移动端适配
- [ ] 开始实现 speaker registry Phase 1：文件式 identity/embedding/sample 持久化
- [ ] 开始实现 speaker registry Phase 1：unknown identity 自动创建、人工 rename、人工 merge

## 已完成的真实环境验收

- [x] 准备默认模型目录到 `models/`
- [x] 按 `config/models.toml` 对齐默认模型路径
- [x] 在 x86 Linux + NVIDIA GPU 环境执行 `uv sync --extra dev --extra default-runtime`
- [x] 调用管理接口成功加载默认模型
- [x] 用真实 wav 音频跑通 `/transcribe/`
- [x] 用 Docker 容器成功加载 `faster-whisper-large-v3` 与 `3dspeaker-default`
- [x] 用 Docker 容器成功跑通 `samples/smoke/smoke_zh_3spk.mp3`
- [x] 用 `docker run --gpus all --network none` 完成默认链的 load + `/transcribe/` 离线验收
- [x] 核对 `runtime/tasks` 与 `system/resources` 返回真实任务和 GPU 数据
- [x] 记录真实环境关键版本：`Driver 570.211.01`、`CUDA 12.8`、`RTX 3060 12GB`

## 下一步重点

- [x] 执行 `docker build -f Dockerfile.gpu -t msr-gpu-runtime:latest .`
- [x] 用 `docker run --gpus all --network none` 做完整离线验收
- [ ] 验证 `faster-whisper + 3D-Speaker` 默认链路在多任务并发下的吞吐、排队和完成乱序表现
- [x] 验证 `faster-whisper + pyannote` 备选链路
- [ ] 验证 `Qwen3-ASR 0.6B` 备选链路
- [ ] 记录 `Qwen3-ASR 1.7B` 在 `3060 12GB` 上是否可接受
- [ ] 核对各模型是否仍有隐式联网行为
- [x] 验证容器内模型路径、权限、挂载方式
- [ ] 补一份真实环境依赖版本记录：CUDA、驱动、Docker、Python
- [ ] 评估是否需要把当前任务摘要导出到日志或 metrics 系统

## 后续收敛

- [ ] 根据真实 GPU 环境决定是否固定 Python 3.11
- [ ] 根据真实模型准备方式决定是否保留 `gpu-runtime` 全量 extra
- [x] 增加 `default-runtime` extra，避免默认链路首次部署就安装全量备选后端
- [x] 将默认 GPU 依赖固定到 CUDA 12 兼容的 PyTorch 2.10 系列
- [ ] 继续收敛 Docker 默认镜像，只保留 `faster-whisper + 3D-Speaker` 服务主链与必要运行数据目录挂载
- [ ] 把当前临时 Docker 基座 `aimeeting-image-offline:latest` 替换为可独立复现的 MSR 基础镜像
- [ ] 评估是否为 Qwen 链路单独维护额外的 handoff 文档或 benchmark 纪要
- [ ] 确认是否需要拆 `Dockerfile.gpu` 和 `Dockerfile.gpu.slim`
- [ ] 评估 `/transcribe/` 是否要补 `speaker_count_hint` 等可选参数
- [x] 设计 speaker registry：保存说话人声纹特征并支持跨音频识别同一人
- [x] 确认 speaker registry 的存储格式、人工命名流程和 UNKNOWN 自动创建策略

## 当前最小验收标准

1. `GET /health` 正常
2. `GET /api/v1/models` 正常
3. 默认 ASR 和 diarization 能成功 `load`
4. `/transcribe/` 返回兼容旧结构的 JSON
5. 全流程在 `--network none` 下可运行
6. 同事能给出一份实际 GPU 运行数据

## 已准备好的工具

- `tools/doctor.py`
- `tools/smoke_api.py`
- `tools/docker_offline_check.sh`
- `docs/linux-gpu-handoff.md`
- `.env.example`
