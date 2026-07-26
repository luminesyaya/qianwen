# 全项目问题汇总 — 数据 / 模型 / 训练

按发生阶段分类，标注现象、根因、修复、教训。

---

## 一、数据方面（12 个）

### 1. BPE 训练 OOM
- **阶段**: MicroLM Tokenizer 训练
- **现象**: 1.2GB `train.txt` 全量训 BPE 被 kill
- **根因**: `tokenizers` 库 Pre-processing 阶段一次性加载整个文件 + 内部索引，超内存
- **修复**: `head -50000` 取子集训练
- **教训**: 5 万行中文已覆盖所有常见字符对，6400 小词表不需要全量统计

### 2. uint16 不能直接转 torch tensor
- **阶段**: MicroLM 数据加载
- **现象**: `TypeError: can't convert np.ndarray of type numpy.uint16`
- **根因**: `np.memmap(dtype=np.uint16)` 存的 token 数据，torch 不接受 uint16
- **修复**: `.astype(np.int32)` 中转
- **教训**: 跨库类型兼容性要在存储和加载两端验证

### 3. 127 万条文本编码极慢
- **阶段**: MicroLM 编码
- **现象**: CPU 单线程逐行 encode，全量跑了几分钟
- **根因**: Python for loop + tokenizer 单行 encode
- **修复**: 可接受，未优化；后续可多进程
- **教训**: 数据管线每个环节要提前估算耗时

### 4. InstructIE 数据集字段漂移
- **阶段**: Qwen 数据探索
- **现象**: train 用 `text`，valid/test 用 `input`；relation 结构不一致；cate 名称漂移
- **根因**: 数据集三个 split 不是同一批次标注的
- **修复**: Step 1 字段标准化 + cate 归一化映射表
- **教训**: 拿到新数据集第一件事永远是 profiling——统计字段分布、检查 cross-split 一致性

### 5. InstructIE 跨集泄漏
- **阶段**: Qwen 数据过滤
- **现象**: train ∩ valid = 143 条，train ∩ test = 181 条
- **根因**: 数据集划分时未去重
- **修复**: Step 2 硬过滤剔除
- **教训**: 泄漏不剔除 = 评测虚高，浪费算力且误导决策

### 6. relation 频率严重倾斜
- **阶段**: Qwen 数据分析
- **现象**: "位于" 193K + "别名" 111K = 29% 总关系量
- **根因**: 数据自然分布
- **修复**: 未处理（已知偏差）
- **教训**: 高频关系会主导 loss，低频关系没有足够的训练信号

### 7. cate 分布不均衡
- **阶段**: Qwen 数据分析
- **现象**: 医学 3244 条 vs 地理 20000 条
- **根因**: 数据自然分布
- **修复**: Step 5 分层采样强制均衡
- **教训**: 不均衡采样 + 小数据集 = 低资源 topic 过拟合

### 8. HuggingFace 下载被墙 / SOCKS 代理
- **阶段**: 全项目多次
- **现象**: `Connection refused` / `SOCKS proxy` 报错
- **修复**: `unset http_proxy https_proxy` + `export HF_ENDPOINT=https://hf-mirror.com`
- **教训**: 国内服务器默认配置 HTTP 代理，HF 下载前必须先清代理

### 9. `datasets` 库 CastError
- **阶段**: Qwen 数据下载
- **现象**: `Couldn't cast column names don't match`
- **根因**: 数据集 schema 定义了 `input` 但 JSON 文件列名是 `text`
- **修复**: 绕过 `datasets` 库，直接 `wget` 原始 JSON
- **教训**: 数据集的 schema 定义可能和实际文件不一致，直接下原始文件最可靠

### 10. JSONL vs JSON 格式误判
- **阶段**: Qwen 数据探索
- **现象**: `json.load()` 报 `Extra data`
- **根因**: 文件是 JSONL 格式（每行一个 JSON），用了 `json.load()` 而不是逐行读
- **修复**: 逐行 `json.loads(line)`
- **教训**: 拿到文件先 `head` 看格式，不要假设

### 11. eval 数据中非 IE 任务没有 JSON gold
- **阶段**: Qwen 评测
- **现象**: 21/40 条样本 gold JSON 解析失败
- **根因**: `relation_qa`、`entity_verification`、`reasoning` 的 gold 是纯文本，不是 JSON
- **修复**: 过滤只评测 `ie_extraction` 类型的样本
- **教训**: 评测脚本要感知数据中的任务类型差异

### 12. DPO 模拟偏好数据无效
- **阶段**: MicroLM DPO
- **现象**: DPO 后 chat 输出全是 "A A A A A"
- **根因**: rejected = 截断/重复版本，模型学到的信号是"长度/流畅 > 短/重复"，不是内容质量
- **修复**: 需要真实偏好数据（chosen vs rejected 在内容上有差异）
- **教训**: DPO 的核心是偏好数据质量，不能靠形式差异模拟

---

## 二、模型方面（7 个）

### 1. view vs reshape
- **阶段**: MicroLM DecoderBlock
- **现象**: `RuntimeError: view size is not compatible`
- **根因**: `transpose` 后 tensor 不连续，`.view()` 要求连续内存
- **修复**: `.contiguous().view()` 或 `.reshape()`
- **教训**: 任何 `transpose/permute` 后想用 `view` 必须先 `contiguous()`

### 2. KV Cache 存了扩展后的 KV
- **阶段**: MicroLM gqa_attention
- **现象**: Decode 时显存异常增长
- **根因**: `repeat_kv` 之后再 `return output, (k, v)` → KV 缓存尺寸被放大了 n_rep 倍
- **修复**: 在 repeat_kv **之前**存 `new_kv`
- **教训**: 缓存的内容必须存原始尺寸，扩展只用于当前计算

### 3. LoRA A/B 在 CPU 上创建
- **阶段**: Qwen LoRA
- **现象**: `RuntimeError: Expected all tensors to be on the same device, found cuda:0 and cpu`
- **根因**: `nn.Parameter(torch.empty(...))` 默认在 CPU，模型在 GPU
- **修复**: `setattr(parent, parts[-1], lora_module.to(device))` — 替换后移到原层设备
- **教训**: 自定义模块的新参数必须显式同步 device

### 4. LoRA 从错误起点训练
- **阶段**: MicroLM LoRA SFT
- **现象**: LoRA 输出全是乱码/重复字符
- **根因**: 从 pretrain checkpoint 开始训（不会对话），lr 设了 1e-3
- **修复**: 从 SFT checkpoint 开始，lr 降到 3e-4
- **教训**: LoRA 只做微调不做从零学——起点必须是已经会任务的模型

### 5. top_p 过滤杀光所有 token
- **阶段**: MicroLM 推理
- **现象**: `probability tensor contains inf, nan or element < 0`
- **根因**: 第一个 token 概率就超过 top_p 阈值 → `cumulative_probs > top_p` 把所有 token 全 mask 为 -inf
- **修复**: 右移一位——`mask[..., 1:] = mask[..., :-1].clone(); mask[..., 0] = False`
- **教训**: top_p 过滤必须保证至少保留概率最高的 token

### 6. decoder 变量名不一致
- **阶段**: MicroLM transformer.py
- **现象**: `AttributeError: 'DecoderBlock' object has no attribute 'd_head'`
- **根因**: `__init__` 定义 `self.dim_head`，`forward` 用 `self.d_head`
- **修复**: 统一为 `self.dim_head`
- **教训**: 变量重命名后全文搜索替换

### 7. `generate()` 缺 `return` 语句
- **阶段**: MicroLM transformer.py
- **现象**: `generate()` 返回 None
- **根因**: 写了 `generated.append(...)` 但没写 `return torch.cat(...)`
- **修复**: 补充 return 语句
- **教训**: 拷贝代码时检查函数完整性

---

## 三、训练方面（9 个）

### 1. Pretrain 步数严重不足
- **阶段**: MicroLM 预训练
- **现象**: SFT 后 chat 输出乱码，val_loss 2.83（预期 < 2.5）
- **根因**: Pretrain 只跑了 4000 步，基座没有语言能力
- **修复**: 重跑 50000 步 → val_loss 5.48 → 3.50
- **教训**: 基座不强，后训练白费。D/N 比至少保证 8x 以上

### 2. Pretrain 早期 loss 不降
- **阶段**: MicroLM 预训练（第一版）
- **现象**: loss 从 7.08 到 7.03，几乎不动
- **根因**: 数据是 `torch.randint` 随机 token，不是真实数据
- **修复**: 跑通真实数据管线后 loss 8.92 → 3.50
- **教训**: 训练前确认数据是真实 token 不是随机数

### 3. Smoke test 时 scheduler 是 None
- **阶段**: Qwen LoRA smoke
- **现象**: `AttributeError: 'NoneType' object has no attribute 'get_last_lr'`
- **根因**: `sched = get_cosine_schedule(...) if not smoke else None`，但日志里调了 `sched.get_last_lr()`
- **修复**: 永远创建 scheduler，smoke 时 warmup 调整为 `min(5, steps//2)`
- **教训**: 条件分支的变量要检查所有使用点

### 4. SAVE_EVERY 被误删
- **阶段**: Qwen LoRA 训练
- **现象**: `NameError: name 'SAVE_EVERY' is not defined`
- **根因**: 改成 `LOG_EVERY = 10 if smoke else 50` 时把 `SAVE_EVERY` 一起删了
- **修复**: 单独保留 SAVE_EVERY
- **教训**: 修 bug 时检查上下文，不要误删无关变量

### 5. SFT loss plateau
- **阶段**: MicroLM SFT（第一版）
- **现象**: loss 5.79 → 2.83 后不动
- **根因**: 基座弱（pretrain 4000 步），SFT 只能教格式不能补语言
- **修复**: 补 pretrain 后再 SFT → loss 2.42
- **教训**: loss 不降先检查上游，别在 SFT 参数上浪费时间

### 6. 38M 容量天花板
- **阶段**: MicroLM 全链路
- **现象**: 64-128 token 后 repetition loop（"又又又又"、"AAAA"）
- **根因**: 8 层 × 64 维 attention head 无法维持长程连贯性
- **修复**: 不可修——容量物理限制。驱动了向 Qwen 迁移的决策
- **教训**: 小模型的能力边界要通过评测量化，而非凭感觉判断

### 7. DPO 训练 loss 下降但效果崩溃
- **阶段**: MicroLM DPO
- **现象**: loss 0.693 → 0.333 正常下降，但模型输出全是 "A"
- **根因**: 模拟 rejected 数据无真实偏好信号
- **修复**: 不可修——需要真实偏好数据
- **教训**: loss 下降 ≠ 模型变好。评测不能只看 loss

### 8. 4090 上 batch 64 反而不快
- **阶段**: MicroLM pretrain 重训
- **现象**: 预计 50min 实际 3h+
- **根因**: `get_batch` Python for-loop 逐一取 slice → CPU 瓶颈，GPU 空转
- **修复**: 未优化（接受了速度代价）
- **教训**: 增大 batch 前先确认数据加载不是瓶颈

### 9. SFT 数据格式导致 eval 不一致
- **阶段**: Qwen eval 初版
- **现象**: Parse% 72% + Strict% 0%
- **根因**: eval 用的 `SYSTEM_PROMPT` 是写死的，和训练时不一致
- **修复**: 从 ChatML 原始文本拆出训练时的 system + user prompt，一字不改
- **教训**: SFT 的推理 prompt 必须和训练时完全一致

---

## 四、基础设施方面（4 个）

### 1. vLLM FlashInfer JIT 编译失败
- **现象**: `fatal error: curand.h: No such file or directory`
- **根因**: conda 环境有 nvcc 但缺少 CUDA 开发头文件
- **修复**: `VLLM_USE_FLASHINFER_SAMPLER=0` 降级采样后端
- **教训**: vLLM 依赖系统级 CUDA 工具链

### 2. `apply_chat_template` 返回格式不一致
- **现象**: `generate()` 收到 string 不是 tensor
- **根因**: 不同版本 tokenizer 的 `return_tensors` 行为不同
- **修复**: `tokenize=False` 拿到 string → `tokenizer.encode()` → `torch.tensor([...])`
- **教训**: HuggingFace API 在不同版本间有差异，显式编码最稳妥

### 3. git 仓库嵌套
- **现象**: minimind-from-scratch 的 git 指向父目录的 cylinder-flow-dit
- **根因**: 在已有 git 仓库的子目录里 init 了新的 git
- **修复**: 删除子目录 .git，在独立目录重新 init
- **教训**: 嵌套 git 不会自动隔离，需手动管理

### 4. 远程实例间文件传输
- **现象**: 数据处理在 2080 Ti，训练在 4090，artifact 要迁移
- **根因**: 云实例间无共享存储
- **修复**: `scp -r` 传 `outputs/` 和 `data/clean/`
- **教训**: 多实例协作要提前规划 artifact 路径和传输方式

---

*汇总时间：2026-07-26*
