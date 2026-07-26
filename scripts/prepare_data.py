"""
数据准备流水线：从原始 InstructIE 格式 → 标准化 → 过滤 → 质量分层 → 任务派生 → 采样 → SFT 格式

Usage:
    python scripts/prepare_data.py --input_dir data/raw --output_dir data/sft_candidate

Steps:
    Step 1: normalize(ds)        - 字段标准化
    Step 2: filter_data(ds)      - 硬过滤 + 软过滤（P99）
    Step 3: quality_tier(ds)     - 质量三档分层，仅保留 high
    Step 4: derive_tasks(ds)     - 四类任务派生（×4 扩增）
    Step 5: stratified_sample(ds)- 分层采样（30,000 条）
    Step 6: to_chat_jsonl(ds)    - 格式转写 + train/valid 切分
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize(ds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Step 1: 字段标准化
    
    功能：
        1. text → input(统一字段名)
        2. relation 对齐：补齐 head_type / tail_type(train 有,valid/test 可能没有）
        3. cate 分类名归一化（如 "建筑结构" → "建筑")
        4. 新增 source 字段（标记数据来源，默认 "unknown")
    
    Args:
        ds: 原始数据列表，每条包含 text, relation, cate 等字段
        
    Returns:
        标准化后的数据列表，每条包含 input, relation, cate, source 等字段
        
    Example:
        >>> raw = [{"text": "乔布斯...", "relation": [...], "cate": "人物传记"}]
        >>> normalized = normalize(raw)
        >>> normalized[0]["input"] == "乔布斯..."
        True
    """
    
    # 定义字段映射规则
    FIELD_MAP = {
        "text": "input",
        # 可扩展其他字段映射
    }
    
    # 定义分类归一化映射表
    # 根据实际数据中的分类名进行扩充
    CATE_NORMALIZE_MAP = {
        "建筑结构": "建筑",
        "建筑工程": "建筑",
        "人物传记": "人物",
        "人物生平": "人物",
        # 补充更多映射...
    }
    
    # 补齐 relation 中缺失的 head_type / tail_type 的默认值
    DEFAULT_HEAD_TYPE = "entity"  # 占位，待确认
    DEFAULT_TAIL_TYPE = "entity"  # 占位，待确认
    
    normalized_ds = []
    
    for idx, sample in enumerate(ds):
        # 深拷贝避免修改原数据
        new_sample = sample.copy()
        
        # ---- 1. 字段重命名: text -> input ----
        if "text" in new_sample:
            new_sample["input"] = new_sample.pop("text")
        # 如果已有 input 字段但没有 text，保留原样
        elif "input" not in new_sample:
            logger.warning(f"样本 {idx} 既没有 'text' 也没有 'input' 字段，跳过")
            continue
        
        # ---- 2. 补齐 relation 中的 head_type / tail_type ----
        if "relation" in new_sample and isinstance(new_sample["relation"], list):
            for rel in new_sample["relation"]:
                # 补齐 head_type
                if "head_type" not in rel or not rel["head_type"]:
                    rel["head_type"] = DEFAULT_HEAD_TYPE
                # 补齐 tail_type
                if "tail_type" not in rel or not rel["tail_type"]:
                    rel["tail_type"] = DEFAULT_TAIL_TYPE
        else:
            # 如果没有 relation 字段或格式不对，初始化为空列表
            new_sample["relation"] = []
            logger.warning(f"样本 {idx} 缺少 relation 字段或格式不正确，初始化为空列表")
            
        # ---- 3. cate 分类名归一化 ----
        if "cate" in new_sample and new_sample["cate"]:
            original_cate = new_sample["cate"]
            # 使用映射表转换，如果没有匹配则保留原值
            new_sample["cate"] = CATE_NORMALIZE_MAP.get(original_cate, original_cate)
            
            # 如果分类名被归一化了，记录日志（可选）
            if new_sample["cate"] != original_cate:
                logger.debug(f"分类归一化: '{original_cate}' → '{new_sample['cate']}'")
        else:
            # 如果没有 cate 字段，设置为 "unknown"
            new_sample["cate"] = "unknown"
            logger.warning(f"样本 {idx} 缺少 cate 字段，设置为 'unknown'")
        
        # ---- 4. 新增 source 字段 ----
        if "source" not in new_sample or not new_sample["source"]:
            new_sample["source"] = "unknown"
        
        normalized_ds.append(new_sample)
    
    # 统计分类分布（用于调试）
    if normalized_ds:
        cate_dist = Counter(s.get("cate", "unknown") for s in normalized_ds)
        logger.info(f"分类分布: {dict(cate_dist)}")
    
    logger.info(f"Step 1 完成: {len(ds)} → {len(normalized_ds)} 条")
    return normalized_ds


def filter_data(ds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Step 2: 硬过滤 + 软过滤（per-cate P99）
    
    硬过滤规则（全局固定阈值）：
        1. relation 非空：len(relation) > 0
        2. input 长度：15 <= len(input) <= 800
        3. relation 数量：len(relation) <= 25
    
    软过滤规则（per-cate P99，按各分类独立计算）：
        1. input 长度不超过该 cate 的 P99
        2. relation 数量不超过该 cate 的 P99
    
    Args:
        ds: 标准化后的数据列表
        
    Returns:
        过滤后的数据列表
    """
    # ---- 硬过滤 ----
    HARD_MIN_INPUT_LEN = 15
    HARD_MAX_INPUT_LEN = 800
    HARD_MAX_RELATION_COUNT = 25
    
    filtered = []
    hard_filter_stats = {
        "empty_relation": 0,
        "input_too_short": 0,
        "input_too_long": 0,
        "relation_too_many": 0,
        "passed_hard": 0
    }
    
    for sample in ds:
        input_text = sample.get("input", "")
        relations = sample.get("relation", [])
        rel_count = len(relations)
        
        # 1. 空 relation 过滤
        if rel_count == 0:
            hard_filter_stats["empty_relation"] += 1
            continue
        
        # 2. input 长度过滤
        input_len = len(input_text)
        if input_len < HARD_MIN_INPUT_LEN:
            hard_filter_stats["input_too_short"] += 1
            continue
        if input_len > HARD_MAX_INPUT_LEN:
            hard_filter_stats["input_too_long"] += 1
            continue
        
        # 3. relation 数量过滤
        if rel_count > HARD_MAX_RELATION_COUNT:
            hard_filter_stats["relation_too_many"] += 1
            continue
        
        hard_filter_stats["passed_hard"] += 1
        filtered.append(sample)
    
    logger.info(f"硬过滤完成: 通过 {hard_filter_stats['passed_hard']} 条")
    logger.info(f"  空 relation: {hard_filter_stats['empty_relation']}")
    logger.info(f"  input 过短 (<{HARD_MIN_INPUT_LEN}): {hard_filter_stats['input_too_short']}")
    logger.info(f"  input 过长 (>{HARD_MAX_INPUT_LEN}): {hard_filter_stats['input_too_long']}")
    logger.info(f"  relation 过多 (>{HARD_MAX_RELATION_COUNT}): {hard_filter_stats['relation_too_many']}")
    
    # ---- 软过滤：per-cate P99 ----
    if not filtered:
        logger.warning("硬过滤后无数据，跳过软过滤")
        return filtered
    
    # 按 cate 分组，计算各 cate 的 P99
    cate_groups: Dict[str, List[Dict]] = defaultdict(list)
    for sample in filtered:
        cate = sample.get("cate", "unknown")
        cate_groups[cate].append(sample)
    
    # 计算每个 cate 的阈值
    cate_thresholds: Dict[str, Dict[str, float]] = {}
    for cate, samples in cate_groups.items():
        input_lens = [len(s.get("input", "")) for s in samples]
        rel_counts = [len(s.get("relation", [])) for s in samples]
        
        # 计算 P99（至少需要 100 条才计算 P99，否则保留全部）
        if len(samples) >= 100:
            input_p99 = pd.Series(input_lens).quantile(0.99)
            rel_p99 = pd.Series(rel_counts).quantile(0.99)
        else:
            # 小分类不过滤软阈值，保留所有
            input_p99 = float('inf')
            rel_p99 = float('inf')
            logger.info(f"分类 '{cate}' 样本数 {len(samples)} < 100，跳过软过滤")
        
        cate_thresholds[cate] = {
            "input_p99": input_p99,
            "rel_p99": rel_p99,
            "count": len(samples)
        }
    
    # 执行软过滤
    soft_filtered = []
    soft_filter_stats = {
        "input_exceed_p99": 0,
        "rel_exceed_p99": 0,
        "passed_soft": 0
    }
    
    for sample in filtered:
        cate = sample.get("cate", "unknown")
        thresholds = cate_thresholds.get(cate, {"input_p99": float('inf'), "rel_p99": float('inf')})
        
        input_len = len(sample.get("input", ""))
        rel_count = len(sample.get("relation", []))
        
        # 检查是否超过 P99
        if input_len > thresholds["input_p99"]:
            soft_filter_stats["input_exceed_p99"] += 1
            continue
        
        if rel_count > thresholds["rel_p99"]:
            soft_filter_stats["rel_exceed_p99"] += 1
            continue
        
        soft_filtered.append(sample)
        soft_filter_stats["passed_soft"] += 1
    
    logger.info(f"软过滤完成: 通过 {soft_filter_stats['passed_soft']} 条")
    logger.info(f"  input 超 P99: {soft_filter_stats['input_exceed_p99']}")
    logger.info(f"  relation 超 P99: {soft_filter_stats['rel_exceed_p99']}")
    
    # 打印各分类的过滤情况（调试用）
    for cate, thresholds in cate_thresholds.items():
        logger.debug(f"  {cate}: n={thresholds['count']}, input_P99={thresholds['input_p99']:.1f}, rel_P99={thresholds['rel_p99']:.1f}")
    
    return soft_filtered


def quality_tier(ds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    质量三档分层，返回 high 质量样本
    
    评分维度：
        1. 匹配率(match_rate):relation 的 head/tail 是否在 entity 列表中
        2. 关系数(relation_count):2-20 为 high,1或21-25 为 medium,0或>25 为 low
        3. input 长度(input_len):50-500 为 high,15-50或500-800 为 medium,<15或>800 为 low
    
    最终评级：
        - high: 三个维度都是 high
        - medium: 任一维度是 medium,但没有 low
        - low: 任一维度是 low
    
    Args:
        ds: 标准化后的数据列表
        
    Returns:
        只保留 high 质量的数据列表
    """
    # 质量统计
    quality_stats = {
        "high": 0,
        "medium": 0,
        "low": 0,
        "total": len(ds)
    }
    
    # 各维度评分统计（用于调试）
    dimension_stats = {
        "match_rate_high": 0,
        "match_rate_medium": 0,
        "match_rate_low": 0,
        "relation_count_high": 0,
        "relation_count_medium": 0,
        "relation_count_low": 0,
        "input_len_high": 0,
        "input_len_medium": 0,
        "input_len_low": 0
    }
    
    high_quality_ds = []
    
    for idx, sample in enumerate(ds):
        # 检查必要字段
        if "input" not in sample or "relation" not in sample:
            logger.warning(f"样本 {idx} 缺少必要字段，标记为 low")
            quality_stats["low"] += 1
            continue
        
        relations = sample.get("relation", [])
        
        # ---- 1. 计算匹配率 ----
        # 提取所有实体名称
        entity_names = set()
        for ent in sample.get('entity', []):
            if isinstance(ent, dict) and 'entity' in ent:
                entity_names.add(ent['entity'])
            elif isinstance(ent, str):
                entity_names.add(ent)
        
        # 计算匹配数
        matches = 0
        total_relations = len(relations)
        
        if total_relations > 0:
            for rel in relations:
                head = rel.get('head', '')
                tail = rel.get('tail', '')
                if head in entity_names and tail in entity_names:
                    matches += 1
            match_rate = matches / total_relations
        else:
            match_rate = 0.0
        
        # ---- 2. 对各维度打分 ----
        
        # 2.1 匹配率评分
        if match_rate >= 0.5:  # 100% 匹配
            match_score = "high"
            dimension_stats["match_rate_high"] += 1
        elif match_rate >= 0.2:  # 50%-99% 匹配
            match_score = "medium"
            dimension_stats["match_rate_medium"] += 1
        else:  # < 50% 匹配
            match_score = "low"
            dimension_stats["match_rate_low"] += 1
        
        # 2.2 关系数评分
        rel_count = total_relations
        if 2 <= rel_count <= 20:
            rel_score = "high"
            dimension_stats["relation_count_high"] += 1
        elif rel_count == 1 or (21 <= rel_count <= 25):
            rel_score = "medium"
            dimension_stats["relation_count_medium"] += 1
        else:  # rel_count == 0 or rel_count > 25
            rel_score = "low"
            dimension_stats["relation_count_low"] += 1
        
        # 2.3 input 长度评分
        input_len = len(sample.get('input', ''))
        if 50 <= input_len <= 500:
            input_score = "high"
            dimension_stats["input_len_high"] += 1
        elif (15 <= input_len < 50) or (500 < input_len <= 800):
            input_score = "medium"
            dimension_stats["input_len_medium"] += 1
        else:  # input_len < 15 or input_len > 800
            input_score = "low"
            dimension_stats["input_len_low"] += 1
        
        # ---- 3. 综合评级 ----
        scores = [match_score, rel_score, input_score]
        
        if "low" in scores:
            quality_level = "low"
        elif "medium" in scores:
            quality_level = "medium"
        else:
            quality_level = "high"
        
        # 记录质量等级到样本中（便于后续分析）
        sample["quality_level"] = quality_level
        sample["quality_scores"] = {
            "match_rate": round(match_rate, 3),
            "match_score": match_score,
            "relation_count": rel_count,
            "rel_score": rel_score,
            "input_len": input_len,
            "input_score": input_score
        }
        
        # 更新统计
        quality_stats[quality_level] += 1
        
        # ---- 4. 只保留 high 质量 ----
        if quality_level == "high":
            high_quality_ds.append(sample)
    
    # ---- 5. 打印统计信息 ----
    logger.info(f"质量分层统计 (总数: {quality_stats['total']}):")
    logger.info(f"  High:   {quality_stats['high']} ({quality_stats['high']/quality_stats['total']*100:.1f}%)")
    logger.info(f"  Medium: {quality_stats['medium']} ({quality_stats['medium']/quality_stats['total']*100:.1f}%)")
    logger.info(f"  Low:    {quality_stats['low']} ({quality_stats['low']/quality_stats['total']*100:.1f}%)")
    
    # 打印各维度评分统计
    logger.info(f"各维度评分分布:")
    logger.info(f"  匹配率 - High: {dimension_stats['match_rate_high']}, Medium: {dimension_stats['match_rate_medium']}, Low: {dimension_stats['match_rate_low']}")
    logger.info(f"  关系数 - High: {dimension_stats['relation_count_high']}, Medium: {dimension_stats['relation_count_medium']}, Low: {dimension_stats['relation_count_low']}")
    logger.info(f"  Input长度 - High: {dimension_stats['input_len_high']}, Medium: {dimension_stats['input_len_medium']}, Low: {dimension_stats['input_len_low']}")
    
    logger.info(f"Step 3 完成: {len(ds)} → {len(high_quality_ds)} 条 high 质量数据")
    return high_quality_ds


def derive_ie_extraction(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    从一条 IE 样本构造 ie_extraction 类型的 SFT 训练样本
    
    Args:
        sample: 标准化后的数据样本，包含 input, relation, entity, cate 等字段
        
    Returns:
        构造好的 SFT 样本，格式为 {"task": "ie_extraction", "messages": [...]}
        
    Example:
        >>> sample = {"input": "圣尼古拉斯岛是美国的岛屿...", "relation": [...], "entity": [...]}
        >>> sft_sample = derive_ie_extraction(sample)
        >>> sft_sample["task"] == "ie_extraction"
        True
    """
    # ---- 1. 提取数据 ----
    input_text = sample.get("input", "")
    cate = sample.get("cate", "未知")
    relations = sample.get("relation", [])
    entities = sample.get("entity", [])
    
    # ---- 2. 构建 entity 列表和 relation 列表 ----
    # 提取实体名称列表（用于显示）
    entity_names = []
    for ent in entities:
        if isinstance(ent, dict):
            entity_names.append(ent.get("entity", ""))
        elif isinstance(ent, str):
            entity_names.append(ent)
    
    # 提取关系类型列表（用于 schema）
    relation_types = set()
    for rel in relations:
        if isinstance(rel, dict):
            rel_type = rel.get("relation", "")
            if rel_type:
                relation_types.add(rel_type)
    
    # ---- 3. 构建 System Prompt ----
    system_prompt = """你是一个严格遵循 schema 的信息抽取助手。请从给定文本中抽取所有实体和关系三元组，以 JSON 格式输出。

    输出格式要求：
    {
    "entities": [
        {"entity": "实体名称", "entity_type": "实体类型"}
    ],
    "relations": [
        {"head": "头实体", "relation": "关系类型", "tail": "尾实体"}
    ]
    }

    注意事项：
    1. 只抽取文本中明确提到的实体和关系
    2. 实体类型从给定 schema 中选择
    3. 不要添加额外信息或解释
    4. 确保 JSON 格式合法"""
    
    # ---- 4. 构建 User Prompt ----
    # 4.1 构建 schema 说明
    schema_text = f"分类: {cate}\n"
    
    if relation_types:
        schema_text += f"关系类型: {', '.join(sorted(relation_types))}\n"
    else:
        schema_text += "关系类型: 无\n"
    
    if entity_names:
        # 取前10个实体作为示例
        example_entities = entity_names[:10]
        schema_text += f"实体示例: {', '.join(example_entities)}"
        if len(entity_names) > 10:
            schema_text += f" 等{len(entity_names)}个实体"
        schema_text += "\n"
    
    # 4.2 构建完整 user prompt
    user_prompt = f"""从以下文本中抽取关系三元组和实体。

    文本: {input_text}

    Schema:
    {schema_text}

    请输出 JSON 格式的结果。"""
    
    # ---- 5. 构建 Assistant 的 JSON 输出 ----
    # 5.1 构建 entities（保持原始格式）
    output_entities = []
    for ent in entities:
        if isinstance(ent, dict):
            output_entities.append({
                "entity": ent.get("entity", ""),
                "entity_type": ent.get("entity_type", "未知")
            })
        elif isinstance(ent, str):
            output_entities.append({
                "entity": ent,
                "entity_type": "未知"
            })
    
    # 5.2 构建 relations
    output_relations = []
    for rel in relations:
        if isinstance(rel, dict):
            output_relations.append({
                "head": rel.get("head", ""),
                "relation": rel.get("relation", ""),
                "tail": rel.get("tail", "")
            })
    
    # 5.3 构建最终 JSON
    assistant_json = {
        "entities": output_entities,
        "relations": output_relations
    }
    
    # 转为 JSON 字符串
    assistant_content = json.dumps(assistant_json, ensure_ascii=False, indent=2)
    
    # ---- 6. 构造 SFT 样本 ----
    sft_sample = {
        "task": "ie_extraction",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content}
        ],
        # 额外信息用于溯源
        "_meta": {
            "cate": cate,
            "relation_count": len(relations),
            "entity_count": len(entities),
            "input_len": len(input_text)
        }
    }
    
    return sft_sample


def derive_relation_qa(sample: Dict[str, Any]) -> Dict[str, Any]:
    """构造关系问答 SFT 样本"""
    input_text = sample.get("input", "")
    relations = sample.get("relation", [])
    
    if not relations:
        # 没有关系，无法构造问答
        return derive_ie_extraction(sample)  # fallback
    
    # 随机选一个关系
    rel = random.choice(relations)
    head = rel.get("head", "")
    tail = rel.get("tail", "")
    relation = rel.get("relation", "")
    
    # 构造问题模板
    question_templates = [
        f"{head}的{relation}是什么？",
        f"哪个实体{relation}于{head}？",
        f"{head}和{tail}是什么关系？",
    ]
    question = random.choice(question_templates)
    
    # 构造 system prompt
    system_prompt = "你是一个基于文本的关系问答助手。请根据给定文本简洁准确地回答问题。"
    
    # 构造 user prompt
    user_prompt = f"""根据以下文本回答问题。

文本: {input_text}

问题: {question}

请直接给出答案，不要添加额外解释。"""
    
    # assistant 回答（简洁答案）
    assistant_content = tail
    
    return {
        "task": "relation_qa",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content}
        ],
        "_meta": {
            "cate": sample.get("cate", "未知"),
            "relation_count": len(relations),
            "input_len": len(input_text)
        }
    }
    

def derive_entity_verification(sample: Dict[str, Any]) -> Dict[str, Any]:
    """构实体验证 SFT 样本"""
    
    input_text = sample.get("input", "")
    relations = sample.get("relation", [])
    entities = sample.get("entity", [])
    
    if not relations or not entities:
        return derive_ie_extraction(sample)  # fallback
    
    # 随机选一个关系
    rel = random.choice(relations)
    head = rel.get("head", "")
    tail = rel.get("tail", "")
    relation = rel.get("relation", "")
    
    # 50% 概率构造正确声明，50% 构造错误声明
    if random.random() < 0.5:
        # 正确声明
        statement = f"{head} {relation} {tail}"
        answer = f"成立。文本中明确提到：\"{statement}\"，与原文一致。"
    else:
        # 错误声明：随机选一个不相关的实体
        other_entities = [e.get("entity", "") for e in entities if e.get("entity", "") != tail]
        if other_entities:
            fake_tail = random.choice(other_entities)
            statement = f"{head} {relation} {fake_tail}"
            answer = f"不成立。文本中 {head} 的 {relation} 是 {tail}，而不是 {fake_tail}。"
        else:
            # 没有其他实体，fallback
            statement = f"{head} {relation} {tail}"
            answer = f"成立。文本中明确提到：\"{statement}\"，与原文一致。"
    
    system_prompt = "你是一个事实核查助手。请根据给定文本判断声明是否成立，并给出简短理由。"
    
    user_prompt = f"""根据以下文本判断声明是否成立。

文本: {input_text}

声明: {statement}

请回答"成立"或"不成立"，并给出理由。"""
    
    return {
        "task": "entity_verification",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer}
        ],
        "_meta": {
            "cate": sample.get("cate", "未知"),
            "relation_count": len(relations),
            "input_len": len(input_text)
        }
    }
    
    
def derive_relation_reasoning(sample: Dict[str, Any]) -> Dict[str, Any]:
    """构造关系推理 SFT 样本"""
    input_text = sample.get("input", "")
    relations = sample.get("relation", [])
    entities = sample.get("entity", [])
    
    # 需要至少 3 个实体才能做链式推理
    if len(entities) < 3 or len(relations) < 2:
        return derive_ie_extraction(sample)  # fallback
    
    # 尝试找到链式关系：A → B → C
    # 先建立 head → tail 的映射
    head_to_tail = {}
    for rel in relations:
        head = rel.get("head", "")
        tail = rel.get("tail", "")
        if head and tail:
            if head not in head_to_tail:
                head_to_tail[head] = []
            head_to_tail[head].append(tail)
    
    # 找一个 A → B 的关系
    chain_found = False
    for rel1 in relations:
        head1 = rel1.get("head", "")
        tail1 = rel1.get("tail", "")
        relation1 = rel1.get("relation", "")
        
        # 看看 tail1 是否作为 head 出现在其他关系中
        if tail1 in head_to_tail and head_to_tail[tail1]:
            tail2 = head_to_tail[tail1][0]  # 取第一个
            # 找 tail1 → tail2 的关系
            for rel2 in relations:
                if rel2.get("head", "") == tail1 and rel2.get("tail", "") == tail2:
                    relation2 = rel2.get("relation", "")
                    chain_found = True
                    break
            if chain_found:
                break
    
    if not chain_found:
        return derive_ie_extraction(sample)  # fallback
    
    # 构造推理
    system_prompt = "你是一个关系推理助手。请基于已知关系事实进行逻辑推理，得出结论。"
    
    user_prompt = f"""基于已知关系事实进行推理。

文本: {input_text}

已知关系:
1. {head1} → {relation1} → {tail1}
2. {tail1} → {relation2} → {tail2}

请推理: {head1} 和 {tail2} 之间可能是什么关系？

请给出推理过程和结论。"""
    
    assistant_content = f"""推理过程:
1. 已知 {head1} {relation1} {tail1}
2. 已知 {tail1} {relation2} {tail2}
3. 通过链式传递，{head1} 通过 {tail1} 间接关联到 {tail2}

结论: {head1} 和 {tail2} 存在间接关系。"""
    
    return {
        "task": "relation_reasoning",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content}
        ],
        "_meta": {
            "cate": sample.get("cate", "未知"),
            "relation_count": len(relations),
            "input_len": len(input_text)
        }
    }


def derive_tasks(ds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Step 4: 四类任务派生(×4 扩增）
    
    任务类型：
        1. ie_extraction (50%) - 信息抽取核心任务
        2. relation_qa (20%) - 关系问答
        3. entity_verification (15%) - 实体验证
        4. relation_reasoning (15%) - 关系推理
    """
    
    random.seed(42)
    
    all_tasks = []
    task_counts = {
        "ie_extraction": 0,
        "relation_qa": 0,
        "entity_verification": 0,
        "relation_reasoning": 0
    }
    
    for idx, sample in enumerate(ds):
        if "input" not in sample or "relation" not in sample:
            logger.warning(f"样本 {idx} 缺少必要字段，跳过")
            continue
        
        # 随机分配任务类型
        rand_val = random.random()
        
        if rand_val < 0.5:
            task_type = "ie_extraction"
            task_sample = derive_ie_extraction(sample)  # ✅ 调用真正的函数
        elif rand_val < 0.7:
            task_type = "relation_qa"
            task_sample = derive_relation_qa(sample)    # ✅ 调用真正的函数
        elif rand_val < 0.85:
            task_type = "entity_verification"
            task_sample = derive_entity_verification(sample)  # ✅ 调用真正的函数
        else:
            task_type = "relation_reasoning"
            task_sample = derive_relation_reasoning(sample)   # ✅ 调用真正的函数
        
        # 确保 task 字段正确
        task_sample["task"] = task_type
        
        all_tasks.append(task_sample)
        task_counts[task_type] += 1
    
    # 打印统计信息
    logger.info(f"Step 4 完成: {len(ds)} → {len(all_tasks)} 条任务")
    logger.info(f"任务分布:")
    for task_type, count in task_counts.items():
        if count > 0:
            percentage = (count / len(all_tasks)) * 100
            logger.info(f"  {task_type}: {count} ({percentage:.1f}%)")
    
    return all_tasks


def stratified_sample(tasks: List[Dict], target_size: int = 30000) -> List[Dict]:
    """
    Step 5: 分层采样 — 按 task_type + cate 均衡采样到 target_size
    每个 (task_type, cate) 组合至少保留 1 条
    """
    import random
    from collections import defaultdict
    
    # ---- 1. 按 (task, cate) 分组 ----
    groups = defaultdict(list)
    for t in tasks:
        key = (t.get('task', 'unknown'), t.get('_meta', {}).get('cate', 'unknown'))
        groups[key].append(t)
    
    n_groups = len(groups)
    logger.info(f"共有 {n_groups} 个 (task, cate) 组合")
    
    # ---- 2. 检查每组至少1条 ----
    # 每组合至少保留 1 条
    min_per_group = 1
    total_min = n_groups * min_per_group
    
    if total_min > target_size:
        logger.warning(f"目标采样数 {target_size} 小于最少保留数 {total_min}，调整为 {total_min}")
        target_size = total_min
    
    # ---- 3. 计算配额 ----
    # 方案A: 按比例分配（推荐）
    # 先给每组分配 1 条保底
    remaining = target_size - total_min
    
    # 计算每组额外的配额（按组大小比例）
    group_sizes = {key: len(items) for key, items in groups.items()}
    total_size = len(tasks)
    
    extra_per_group = {}
    for key, size in group_sizes.items():
        # 按比例分配剩余名额
        extra = int(remaining * (size / total_size))
        extra_per_group[key] = extra
    
    # 处理剩余的零头（从大到小分配）
    allocated = sum(extra_per_group.values())
    leftover = remaining - allocated
    
    # 按组大小排序，从大到小分配零头
    sorted_groups = sorted(group_sizes.items(), key=lambda x: x[1], reverse=True)
    for i in range(leftover):
        if i < len(sorted_groups):
            key = sorted_groups[i][0]
            extra_per_group[key] += 1
    
    # 最终配额 = 保底1条 + 额外配额
    quotas = {key: min_per_group + extra_per_group.get(key, 0) for key in groups.keys()}
    
    # ---- 4. 采样 ----
    sampled = []
    for key, items in groups.items():
        quota = quotas[key]
        if len(items) <= quota:
            sampled.extend(items)
        else:
            sampled.extend(random.sample(items, quota))
    
    # ---- 5. 打乱 ----
    random.shuffle(sampled)
    
    logger.info(f"Step 5 完成: {len(tasks)} → {len(sampled)} 条 (target={target_size})")
    return sampled


def to_chat_jsonl(tasks: List[Dict], train_ratio: float = 0.95):
    """
    Step 6: ChatML 格式转写 + train/valid 切分
    把 messages 列表转成 Qwen ChatML 文本，一条写一行
    """
    # ---- 1. 创建输出目录 ----
    output_dir = Path('data/clean')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 2. 切分训练/验证集 ----
    n = len(tasks)
    split = int(n * train_ratio)  # 95% 训练，5% 验证
    random.shuffle(tasks)  # 打乱数据

    # ---- 3. 转换并保存 ----
    for tag, subset in [('train', tasks[:split]), ('valid', tasks[split:])]:
        path = output_dir / f'{tag}.jsonl'
        with open(path, 'w', encoding='utf-8') as f:
            for t in subset:
                # 3.1 渲染 ChatML 格式
                text = ""
                for msg in t['messages']:
                    role = msg['role']
                    content = msg['content']
                    text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                
                # 3.2 写入 JSONL
                f.write(json.dumps({"text": text, "task": t['task']}, ensure_ascii=False) + '\n')

        logger.info(f"  {tag}: {len(subset)} 条 → {path}")

    logger.info(f"Step 6 完成: train {len(tasks[:split])} / valid {len(tasks[split:])}")


def main():
    # ---- 1. 读取原始数据 ----
    input_file = 'data/raw/train_zh.json'
    logger.info(f"读取原始数据: {input_file}")
    
    ds = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    ds.append(json.loads(line))
        logger.info(f"读取原始数据: {len(ds)} 条")
    except FileNotFoundError:
        logger.error(f"文件不存在: {input_file}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {e}")
        return
    
    # ---- 2. 执行数据处理流水线 ----
    logger.info("=" * 60)
    logger.info("开始数据处理流水线")
    logger.info("=" * 60)
    
    # Step 1: 标准化
    ds = normalize(ds)
    logger.info(f"Step 1 完成: {len(ds)} 条")
    
    # Step 2: 过滤
    ds = filter_data(ds)
    logger.info(f"Step 2 完成: {len(ds)} 条")
    
    # Step 3: 质量分层
    ds = quality_tier(ds)
    logger.info(f"Step 3 完成: {len(ds)} 条")
    
    # 检查是否还有数据
    if not ds:
        logger.warning("没有数据通过质量筛选，流程结束")
        return
    
    # 打印最终 cate 分布
    cate_dist = Counter(d.get('cate', 'unknown') for d in ds)
    logger.info(f"最终保留: {len(ds)} 条")
    logger.info(f"cate 分布: {dict(cate_dist)}")
    
    # ---- 3. Step 4: 任务派生 ----
    logger.info("=" * 60)
    logger.info("开始任务派生 (Step 4)")
    logger.info("=" * 60)
    
    # 派生任务
    tasks = derive_tasks(ds)
    logger.info(f"Step 4 完成: 生成 {len(tasks)} 条任务")
    
    # ---- 4. Step 5: 分层采样 ----
    logger.info("=" * 60)
    logger.info("开始分层采样 (Step 5)")
    logger.info("=" * 60)
    
    tasks = stratified_sample(tasks, target_size=30000)
    
    # ---- 5. Step 6: 格式转换 ----
    logger.info("=" * 60)
    logger.info("开始格式转换 (Step 6)")
    logger.info("=" * 60)
    
    to_chat_jsonl(tasks, train_ratio=0.95)
    
    # ---- 6. 保存中间结果 ----
    # 创建输出目录
    output_dir = Path('data/sft_candidate')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 6.1 保存采样后的任务数据
    output_file = output_dir / 'sft_data_sampled.jsonl'
    logger.info(f"保存采样后的任务数据到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')
    
    # 6.2 按任务类型分别保存
    task_types = ['ie_extraction', 'relation_qa', 'entity_verification', 'relation_reasoning']
    for task_type in task_types:
        type_tasks = [t for t in tasks if t.get('task') == task_type]
        if type_tasks:
            type_file = output_dir / f'sft_{task_type}.jsonl'
            with open(type_file, 'w', encoding='utf-8') as f:
                for task in type_tasks:
                    f.write(json.dumps(task, ensure_ascii=False) + '\n')
            logger.info(f"  {task_type}: {len(type_tasks)} 条 -> {type_file}")
    
    # 6.3 保存统计信息
    stats_file = output_dir / 'stats.json'
    stats = {
        "原始数据": len(ds),
        "派生任务总数": len(tasks),
        "任务分布": {
            task_type: len([t for t in tasks if t.get('task') == task_type])
            for task_type in task_types
        },
        "cate_分布": dict(cate_dist)
    }
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"统计信息保存到: {stats_file}")
    
    # ---- 7. 打印最终总结 ----
    logger.info("=" * 60)
    logger.info("数据处理完成!")
    logger.info("=" * 60)
    logger.info(f"输出目录: data/clean/")
    logger.info(f"总任务数: {len(tasks)}")
    for task_type in task_types:
        count = len([t for t in tasks if t.get('task') == task_type])
        if count > 0:
            pct = count / len(tasks) * 100
            logger.info(f"  {task_type}: {count} ({pct:.1f}%)")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()