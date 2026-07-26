# 项目复盘 — MicroLM + Qwen IE 双轨实践

## 1. 关键成果

| 维度 | MicroLM | Qwen IE |
|------|------|------|
| 模型 | 自研 LLaMA 架构 38M | Qwen2.5-1.5B LoRA |
| 训练闭环 | pretrain → SFT → LoRA → DPO | 数据 → LoRA → 评测 → 部署 |
| 核心指标 | 对话评分 3.0/5 | Parse% 95.2%, Smoke 5/5 |
| 部署形态 | chat.py REPL | vLLM HTTP API + Web 前端 |
| LoRA 效率 | 0.60% 参数 | 0.14% 参数 |

**核心验证**：
1. 从零搭建完整训练链路是可行的——每一步都手写，不依赖 HF 模型层
2. LoRA 在微型模型上同样有效——0.6% 参数达到全参 95% 效果
3. 方法论可以跨生态迁移——MicroLM 的设计模式在 Qwen 线上几乎原样复用
4. 聚焦比泛化更容易出硬指标——Parse%/Strict% 替代人工打分

## 2. 关键 Bug 与教训

### Bug 1: uint16 不能直接转 torch tensor
- **现象**: `TypeError: can't convert np.ndarray of type numpy.uint16`
- **根因**: `np.memmap(dtype=np.uint16)` 存的 token 数据，torch 不接受 uint16
- **修复**: `.astype(np.int32)` 中转
- **教训**: 不同库之间的类型兼容性永远要在存储和加载两端验证

### Bug 2: view vs reshape 踩坑
- **现象**: `RuntimeError: view size is not compatible`
- **根因**: transpose 后 tensor 不连续，`.view()` 失败
- **修复**: `.contiguous().view()` 或 `.reshape()`
- **教训**: PyTorch 的 view 需要连续内存，transpose/permute 后必须 contiguous

### Bug 3: KV Cache 存了扩展后的 KV
- **现象**: Decode 时显存异常增长
- **根因**: `repeat_kv` 之后再存 KV → 缓存尺寸被放大了 n_rep 倍
- **修复**: 在 repeat_kv **之前**存 new_kv
- **教训**: 缓存的内容必须是原始尺寸，扩展只用于当前计算

### Bug 4: Pretrain 步数不够导致 SFT 崩溃
- **现象**: SFT 后 chat 输出乱码/重复，val_loss 2.83
- **根因**: Pretrain 只跑了 4000 步，基座没有语言能力
- **修复**: 重跑 50000 步 → val_loss 从 5.48 降到 3.50
- **教训**: 基座不强，后训练白费。smoke test 不能只验证"loss 降了"，要验证"模型真的会说话了"

### Bug 5: LoRA 从 Pretrain 起点训炸
- **现象**: LoRA 输出全是乱码/重复字符
- **根因**: 从 pretrain checkpoint（不会对话）开始训练，lr 还设了 1e-3
- **修复**: 从 SFT checkpoint 开始，lr 降到 3e-4
- **教训**: LoRA 的起点必须是已经会任务的状态，它只做微调不做从零学

### Bug 6: DPO 模拟数据导致模型学"长 = 好"
- **现象**: DPO 后 chat 输出全是 "A"
- **根因**: rejected = 截断回答，模型学到了"长回答 > 短回答"
- **教训**: DPO 的偏好信号必须是内容质量差异，不能是形式差异

### Bug 7: vLLM FlashInfer JIT 编译失败
- **现象**: `fatal error: curand.h: No such file or directory`
- **根因**: conda 环境缺少 CUDA 开发头文件
- **修复**: `VLLM_USE_FLASHINFER_SAMPLER=0` 降级使用原生采样
- **教训**: vLLM 依赖系统级 CUDA 环境，云实例不一定完整

### Bug 8: epoch 参数废弃警告
- **现象**: `UserWarning: The epoch parameter in scheduler.step() was not necessary`
- **根因**: PyTorch 版本差异，新版不需要传 epoch
- **修复**: 直接用 `scheduler.step()`

## 3. 方法论收获

1. **配置驱动 > 硬编码** — 17 个 JSON/Python 配置文件，每个实验都有完整记录
2. **先 smoke 再正式** — 50 步验证全链路，避免"跑了 10 小时发现 bug"
3. **评测先于优化** — 建立统一评测基线后才能判断改动是进步还是退步
4. **自研 + 开源双轨** — 自研理解原理，开源交付产品，互补
5. **协议显式化** — 数据长什么样写进文档，不隐含在代码逻辑里

## 4. 技术栈对比

| 维度 | MicroLM | Qwen | 面试说法 |
|------|------|------|------|
| 模型 | 手写 | HF AutoModel | "从零实现过，也用工业工具" |
| Tokenizer | 自训 BPE | HF AutoTokenizer | "理解 BPE 原理，用过生产级 tokenizer" |
| LoRA | 自研 lora.py | PEFT | "手写了 LoRA 理解低秩分解，生产用 PEFT" |
| 训练 | 裸 PyTorch | HF + PEFT | "能裸写训练循环，也能用工业框架" |
| 推理 | chat.py | vLLM | "从脚本到服务全做过" |

## 5. 后续方向

| 优先级 | 方向 | 工作量 |
|------|------|------|
| P0 | 更新简历 + GitHub README | 1h |
| P1 | 录 demo 视频放 GitHub | 30min |
| P2 | Function Calling 扩展 | 1d |
| P3 | INT8 量化部署 | 1d |
| 长期 | Agent 框架集成 | — |

---

*2026-07-26*
