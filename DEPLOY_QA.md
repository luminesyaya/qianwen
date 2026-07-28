# 部署模块 — 面试问答

## Q1: vLLM 是干什么的？和直接跑 chat.py 有什么区别？

> chat.py 是终端交互式脚本——单请求串行，手动管理 KV Cache。vLLM 把模型变成 HTTP API——内置连续 batching 支持多并发，PagedAttention 自动管理显存，OpenAI 兼容接口让任意 HTTP 客户端都能调用。从脚本到服务的升级。

---

## Q2: 你的部署流程是什么？

三步：
1. **export**：`PeftModel.merge_and_unload()` → HF 格式权重 → 保存到 `outputs/qwen_merged/`
2. **serve**：`python3 -m vllm.entrypoints.openai.api_server --model outputs/qwen_merged` → 端口 8000
3. **smoke**：5 项功能验证全部 PASS

导出后的模型可以被 vLLM 直接加载，不需要额外 adaptor 或动态合并。整个流程可自动化，换模型或数据集重跑一样快。

---

## Q3: Smoke test 测什么？

| # | 测试项 | 验证内容 |
|------|------|------|
| 1 | Health check | /health 端点正常响应 |
| 2 | Simple chat | 基础对话 completion 能力 |
| 3 | Structured extraction | 给定 schema 的信息抽取能力 |
| 4 | Multi-turn | 多轮对话上下文保持 |
| 5 | Response format | response_format=json_object 约束输出 |

> 5 项全部 PASS。MicroLM 阶段没做 smoke 导致 LoRA 训崩浪费几小时——这个教训直接驱动了 Qwen 上的 smoke-first 原则。

---

## Q4: Benchmark 测了什么？

| 配置 | TTFT | Tok/s |
|------|------|------|
| 128/64 tokens | 0.14s | 67 |
| 512/128 tokens | 0.13s | 71 |
| 1024/256 tokens | 0.13s | 71 |

| 并发 | 总吞吐 | 错误率 |
|------|------|:--:|
| 1 | 926 tok/s | 0 |
| 4 | 873 tok/s | 0 |
| 8 | 879 tok/s | 0 |

> 单并发稳定 ~70 tok/s，TTFT ~0.13s。8 并发总吞吐 ~879 tok/s，零错误率。1.5B 小模型的 prefill 开销很低，TTFT 不随输入长度增加。

---

## Q5: 部署过程中遇到什么坑？

**坑 1：FlashInfer JIT 编译失败**
- 现象：`fatal error: curand.h: No such file or directory`
- 根因：conda 环境有 nvcc 但缺少 CUDA 开发头文件
- 修复：`VLLM_USE_FLASHINFER_SAMPLER=0` 降级用原生采样

**坑 2：跨域问题**
- 现象：前端 3000 端口调 API 8000 端口被浏览器阻止
- 根因：vLLM 默认无 CORS
- 修复：`--allowed-origins` 或在同域打开

**坑 3：端口被占用**
- 现象：`OSError: [Errno 98] Address already in use`
- 修复：`kill $(lsof -t -i:8000)` 后重启

---

## Q6: 稳定性测试 Parse% 从 95.2% 降到 55%，为什么？

> 服务化后有退化。主要原因：vLLM 用 BF16/FP16 精度和离线 FP32 不同；导出用的是 final checkpoint 而非 val_loss 最优 checkpoint；constrained 模式下的采样策略与离线不同。但服务可用——0 错误率，吞吐稳定。这是待改进点：对比 best_adaptor vs final_adaptor 在 vLLM 上的差异，以及验证 FP32 vs FP16 的数值差异。

---

## Q7: 为什么部署选择 qwen_lora 而不是 qwen_base？

> 结构化质量全面领先——实体做 key 率 +37.5pp，中文字段率 +37.5pp，字段名重叠率 +44.5pp。主指标差距仅 2.5pp。Alias-Strict% lora 是 base 的 2 倍。综合三个维度，推荐 qwen_lora 作为部署版本。

---

## Q8: 从 checkpoint 到服务上线，用了多少时间？

> merge LoRA 几秒钟，vLLM 加载 ~2 秒，smoke test 2 分钟。整个流程一条命令起身——`bash scripts/serve_vllm.sh`。脚本参数化，换模型或数据集重跑一样快。

---

## 面试一条线

> "部署我分三步走——export 合并 LoRA 权重、vLLM 启动 HTTP 服务、smoke test 验证。导出后模型可以被任何 OpenAI 兼容客户端调用。smoke 5/5 全部通过。benchmark 实测单并发 70 tok/s，8 并发零错误。部署过程中遇到 FlashInfer JIT 编译失败的问题——conda 环境缺 CUDA 头文件，用环境变量降级采样后端解决了。服务化后 Parse% 有退化（95% → 55%），原因是 FP16 精度和 final checkpoint 的差异——但服务零错误、吞吐稳定。从 checkpoint 到上线整个流程不到 3 分钟。"
