# GRPO 结构化抽取对齐——实验规约

## 一、实验流程

```
数据加载 → 模型合并LoRA → GRPO训练 → 评测对比

详细步骤:
1. load_ie_dataset() 从 train.jsonl 取 ie_extraction 样本
   - 拆 prompt（到 <|im_start|>assistant\n 为止）和 gold（ChatML中的JSON）
   - 只取 task=="ie_extraction"（有JSON gold可自动打分）

2. PeftModel + merge_and_unload() 合并SFT LoRA → 完整模型
   - 原因: GRPO需要vLLM生成，完整模型比LoRA快

3. GRPOTrainer.train() 自动完成:
   - 每步 2 prompt × 4 回答 = 8 completions, 2组独立比较
   - 每组内: advantage = reward - mean(reward)
   - GRPO loss: importance_sampling
   - 每步: 生成 → reward打分 → advantage → backward → update

4. eval.py 评测:
   - 对比 SFT vs GRPO 的 Parse%/Strict%/Alias%
```

## 二、关键参数

### 模型与数据

| 参数 | 调试值 | 正式实验值 | 说明 |
|------|:--:|:--:|------|
| BASE_MODEL | Qwen2.5-1.5B-Instruct | 同 | 固定 |
| MODEL_PATH | outputs/qwen_lora/final | 同 | SFT LoRA adapter |
| MAX_SAMPLES | 50 | 500-2000 | 调试/正式 |
| num_train_epochs | 1 | 3-5 | |

### GRPO 超参

| 参数 | 调试值 | 正式实验值 | 说明 |
|------|:--:|:--:|------|
| num_generations | 4 | 4-8 | group_size, 越大advantage越稳定但越慢 |
| per_device_train_batch_size | 2 | 4 | 每步prompt数 |
| max_completion_length | 1536 | 1024 | 1024够用, 1536偶尔OOM |
| learning_rate | 4e-5 | 2e-5 | GRPO比SFT更保守 |

### Reward 设计（v2，已验证）

| 维度 | 权重 | 打方式 |
|------|:--:|------|
| Parse | 0.2 | 合法JSON=0.2, 非法=0 |
| Alias | 0.1 | 有entities+relations=0.1 |
| Strict | 0.5 | 连续分: 1 - (|pred_n-gold_n|/gold_n)/2 |
| 长度惩罚 | -0.3 | 完全没有任何实体时重罚 |

## 三、监控指标

| 指标 | 含义 | 正常值 | 异常 |
|------|------|:--:|------|
| frac_reward_zero_std | 组内无差异比例 | <0.2 | >0.5 则reward无区分度 |
| reward | 平均分 | 0.6-0.8 | <0.4 SFT退化 |
| reward_std | 组内方差 | 0.03-0.15 | 0或<0.01 无训练信号 |
| clipped_ratio | 被截断比例 | 0 | >0 需要减max_completion_length |
| OOM | 显存溢出 | 偶尔1-2次可接受 | 频繁则减group_size或max_completion_length |

## 四、已知局限

1. **Strict%瓶颈不在对齐上**：标注员间一致性本身有限，GRPO无法突破人类标注上限
2. **数据量少时无法体现效果**：50条只能验证流程，500+条才可能看到趋势
3. **连续reward方差小**：和数学题(0/1)不同，IE的连续reward组内区分度天然较小
4. **GRPO训练轮数要适中**：太少未见效果，太多可能过拟合高频topic

## 五、文件清单

| 文件 | 作用 |
|------|------|
| scripts/train_grpo_trl.py | GRPO训练脚本 |
| scripts/eval.py | 评测脚本(支持full model + LoRA) |
| GRPO_EXPERIMENT.md | 实验记录 |
| outputs/grpo_ie_final/ | 最终模型 |
| outputs/qwen_lora/final/ | SFT起点(对比用) |
