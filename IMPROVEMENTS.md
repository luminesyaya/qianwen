# Pipeline 待改进点

## 1. Step 0：跨文件去重（待加）

**现状**：只用 `train_zh.json`，绕开跨集泄漏。  
**缺陷**：无法处理多文件输入，valid/test 没进 Pipeline。  
**改进**：在 Step 1 之前加 Step 0——合并所有输入文件 → SHA256 全文去重 → 再进标准化。

```python
def dedup_all_files(file_paths):
    seen = set()
    docs = []
    for path in file_paths:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                h = hashlib.sha256(d['text'].encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    docs.append(d)
    return docs
```

## 2. relation 频率倾斜（已知偏差，未处理）

**现状**："位于" + "别名" 占关系总量 29%。  
**风险**：模型可能过拟合这两个高频关系，低频关系训练信号不足。  
**改进方向**：
- 方案 A：对高频 relation 做降采样（按概率随机丢弃）
- 方案 B：训练时用 focal loss 自动降权高频样本
- 方案 C：不做模型级处理，改为评测时按 relation 类型分层报告指标

## 3. 分层采样策略（可对比实验）

**现状**：按比例分配（保留原始分布）。  
**备选**：等额配额（每个 (task, cate) 组合强制等量）。  
**改进**：做消融实验——对比两种采样策略下的 Parse%/Strict%。

## 4. 低资源 topic oversample 风险

**现状**：医学 1052 条 → 采样配额 ~625 条，数据被复用。  
**风险**：模型见过同一批医学样本多次，可能过拟合。  
**改进**：监控 per-topic 的 val_loss，如果医学的 val_loss 不降反升说明过拟合了。

## 5. 消融实验缺失

**现状**：Pipeline 每步有统计输出，但没有消融对比。  
**缺陷**：无法量化每步的贡献（"不加质量分层 Parse% 会降多少"回答不了）。  
**改进**：做 3 组对照实验——完整 Pipeline vs 去掉 Step 3 vs 去掉 Step 5。

## 6. 数据版本管理

**现状**：每次跑 Pipeline 产出固定文件名，无版本标记。  
**缺陷**：多次跑 Pipeline 后不知道哪个版本对应哪次实验结果。  
**改进**：`data/clean/` 下加时间戳目录，或每个产出文件附带 `metadata.json` 记录参数。

---

*整理时间：2026-07-26*
