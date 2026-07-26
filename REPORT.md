# Project Report — LLM 全链路实践

## 项目概览

两个互补项目，覆盖大模型原理、训练、微调、推理全链路：

| 项目 | MiniLLM from scratch | Qwen 结构化抽取 |
|------|------|------|
| 定位 | 从零理解原理 | 工业工具体验 |
| 模型 | 自研 LLAMA 架构 38M | Qwen2.5-1.5B LoRA |
| Tokenizer | 自训 BPE (vocab 6400) | HuggingFace AutoTokenizer |
| LoRA | 自研 lora.py | PEFT Library |
| 数据 | MiniMind SFT 90 万条对话 | InstructIE 171K 条 IE 样本 |
| 评测 | 人工评分 (3.0/5) | 全自动 Parse%/Strict%/Alias% |

---

## 一、MiniLLM 模块（核心主线）

### 1.1 模型架构
- RMSNorm、SwiGLU、GQA (8 Q-heads / 4 KV-heads)、RoPE、KV Cache
- 38M 参数, d_model=512, num_layers=8

### 1.2 训练链路
- 预训练 50K steps → loss 8.92 → 3.50
- SFT 2K steps → val_loss 2.45
- DPO (data simulated) → 代码跑通, 无真实偏好对
- LoRA (self-developed) → 0.60% trainable params, val_loss 2.58 (+5.3% gap)

### 1.3 推理系统
- top-p nucleus sampling → 延缓 repetition 崩溃
- Multi-turn chat → history management + context trimming
- KV Cache benchmark → 1.1x speedup (sample size limit)

### 1.4 能力边界
**能做到**：基础概念解释、列表/编号格式、基本对话
**做不好**：精确翻译、事实正确性
**做不到**：长文本一贯性、JSON 输出、无循环崩溃

---

## 二、Qwen 结构化抽取（迁移主线）

### 2.1 数据管线
- 6-step pipeline: Normalize → Filter → Quality Tier → Derive Tasks → Stratified Sample → ChatML
- 171K → 30K samples (28.5K train / 1.5K valid)
- 4 task types: ie_extraction (50%), relation_qa (20%), entity_verification (15%), reasoning (15%)

### 2.2 LoRA 训练
- Qwen2.5-1.5B-Instruct 基座
- LoRA r=8, alpha=16, trainable 0.14%
- Smoke test 50 steps → Full training 2K steps
- val_loss 0.41 → 0.068 (降幅 83%)

### 2.3 评测结果 + 对比

| 指标 | MicroLM 38M | Qwen 1.5B LoRA | 提升 |
|------|:--:|:--:|------|
| Parse% | 0% | **95.2%** | 从无到有 |
| Strict% | 0% | 23.8% | 突破 |
| Alias% | 0% | **95.2%** | 标准结构 |
| 可训练参数 | 38M | 2.18M (0.14%) | 省 99.86% |
| 评测方式 | 人工 | 全自动 | — |

### 2.4 关键决策
1. MicroLM Parse% = 0% → 必须换基座
2. 选 Qwen2.5-1.5B：规模适中 + 生态成熟
3. 聚焦结构化抽取：有硬指标、LoRA 擅长
4. Smoke test 前置，避免 MicroLM 未做 smoke 直接训的坑

### 2.5 已知局限
1. 关系频率倾斜（"位于" 193K + "别名" 111K = 29%）
2. cate 分布不均衡 (医学 3244 vs 地理 20000)
3. Strict% 天花板 (IE 标注不一致)
4. 非 ie_extraction 类型无 JSON 评测
5. LoRA 字段名约束弱（模型有 Qwen 基座 bias）

---

## 三、技术栈

| 组件 | MicroLM | Qwen |
|------|------|------|
| PyTorch | ✅ | ✅ |
| HuggingFace | ❌ (all custom) | ✅ |
| PEFT | ❌ (self-developed) | ✅ |
| Tokenizers | ✅ (HuggingFace standalone) | ✅ (AutoTokenizer) |
| vLLM | ❌ | ✅ (planned) |

*报告时间：2026-07-25*
