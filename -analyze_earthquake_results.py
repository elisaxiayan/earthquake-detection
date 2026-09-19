#!/usr/bin/env python3
"""
地震幻觉检测分析脚本
将 question JSON + result JSON 整理成对比分析 CSV。

用法:
    python3 /home/caokexin/analyze_earthquake_results.py <question.json> <result.json> [选项]

选项:
    -n, --name      数据集名称 (默认从文件名推断)
    -o, --output    输出 CSV 路径 (默认: <result>_analysis_<时间戳>.csv)
    --second        第二组 question + result (合并到同一 CSV)
    --second-name   第二组数据集名称
    --no-summary    不打印汇总统计

示例:
    # 单个数据集
    python3 /home/caokexin/analyze_earthquake_results.py /home/caokexin/earthquakes_true.json /home/caokexin/earthquakes_true_result.json -n 真实地震

    # 合并两个数据集
    python3 /home/caokexin/analyze_earthquake_results.py /home/caokexin/earthquakes_true.json /home/caokexin/earthquakes_true_result.json -n 真实地震 --second /home/caokexin/false_domestic_earthquake_batch_50.json /home/caokexin/false_domestic_result.json --second-name 伪造地震
"""

import argparse
import csv
import json
import os
import re
import sys


# ── 从 input_text 中提取地震信息 ──────────────────────────────────────

def parse_input_text(text: str) -> dict:
    """从标准格式的地震文本中提取地点、震级、时间、经纬度。"""
    info = {
        "地点": "",
        "震级(声称)": "",
        "时间(北京)": "",
        "纬度": "",
        "经度": "",
    }

    # 匹配: 北京时间2026年6月15日11时24分(在)内蒙古...（北纬40.72度，东经107.35度）发生3.0级地震
    m = re.search(
        r"北京时间\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*"
        r"(\d{1,2})\s*[时:：]\s*(\d{1,2})\s*分\s*"
        r"(?:在)?\s*(.+?)\s*"
        r"[（(]\s*北纬\s*([\d.]+)\s*度[，,]\s*东经\s*([\d.]+)\s*度\s*[）)]\s*"
        r"发生\s*([\d.]+)\s*级地震",
        text,
    )
    if m:
        year, month, day, hour, minute = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        info["时间(北京)"] = f"{year}-{int(month):02d}-{int(day):02d} {int(hour):02d}:{int(minute):02d}"
        info["地点"] = m.group(6).strip()
        info["纬度"] = m.group(7)
        info["经度"] = m.group(8)
        info["震级(声称)"] = m.group(9)
    else:
        # 宽松回退：分别提取
        time_m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2})时(\d{1,2})分", text)
        if time_m:
            y, mo, d, h, mi = time_m.groups()
            info["时间(北京)"] = f"{y}-{int(mo):02d}-{int(d):02d} {int(h):02d}:{int(mi):02d}"

        lat_m = re.search(r"北纬\s*([\d.]+)\s*度", text)
        lon_m = re.search(r"东经\s*([\d.]+)\s*度", text)
        mag_m = re.search(r"发生\s*([\d.]+)\s*级地震", text)
        if lat_m:
            info["纬度"] = lat_m.group(1)
        if lon_m:
            info["经度"] = lon_m.group(1)
        if mag_m:
            info["震级(声称)"] = mag_m.group(1)

        # 地点提取（回退）
        loc_match = re.search(r"分\s*(?:在)?\s*(.+?)\s*[（(]", text)
        if loc_match:
            info["地点"] = loc_match.group(1).strip()

    return info


# ── 从 result item 中提取验证信息 ─────────────────────────────────────

def extract_result_info(item: dict, original_location: str) -> dict:
    """从单个 result item 中提取 occurrence/magnitude 判定、USGS 数据等。"""
    results = item.get("results", [])
    extraction = item.get("extraction", {})
    claims = extraction.get("claims", [])

    info = {
        "occurrence判定": "missing",
        "occurrence原因": "",
        "magnitude判定": "unknown",
        "震级容差": "-",
        "USGS事件数": 0,
        "USGS匹配震级": "-",
        "震级差值": "-",
        "输入标准地点": "",
        "USGS匹配地点": "",
        "事件距离(km)": "-",
        "地点匹配等级": "",
        "地点核验提示": "",
        "查询时间窗口": "",
        "提取含occurrence": "否",
        "提取含magnitude": "否",
    }

    def apply_sample(samples):
        if not samples or not isinstance(samples, list) or not isinstance(samples[0], dict):
            return
        first = samples[0]
        if first.get("magnitude") is not None:
            info["USGS匹配震级"] = str(first["magnitude"])
        info["USGS匹配地点"] = first.get("place_zh") or ""
        if first.get("distance_km") is not None:
            info["事件距离(km)"] = first["distance_km"]
        info["地点匹配等级"] = first.get("place_match") or ""
        info["地点核验提示"] = first.get("place_warning") or ""

    # ── 从 results 提取 occurrence / magnitude 判定 ──
    for r in results:
        ft = r.get("fact_type", "")
        status = r.get("status", "unknown")
        claim = r.get("claim", {})

        if ft == "occurrence":
            info["occurrence判定"] = status
            info["occurrence原因"] = r.get("reason", "")
            # USGS 事件数
            info["USGS事件数"] = r.get("event_count", 0)
            apply_sample(r.get("sample_events", []))
            info["地点核验提示"] = info["地点核验提示"] or "; ".join(r.get("warnings", []))
            info["输入标准地点"] = claim.get("standard_location") or ""
            # 查询时间窗口
            ti = claim.get("time_interval", {})
            start = ti.get("start_time", "")
            end = ti.get("end_time", "")
            if start and end:
                info["查询时间窗口"] = f"{start}~{end}"

        elif ft == "magnitude":
            info["magnitude判定"] = status
            if r.get("tolerance") is not None:
                info["震级容差"] = r["tolerance"]
            # 如果 magnitude result 也有 sample_events 但 occurrence 没有
            if info["USGS匹配震级"] == "-":
                apply_sample(r.get("sample_events", []))
            info["输入标准地点"] = info["输入标准地点"] or claim.get("standard_location") or ""
            # 如果 occurrence 没有时间窗口，从 magnitude 取
            if not info["查询时间窗口"]:
                ti = claim.get("time_interval", {})
                start = ti.get("start_time", "")
                end = ti.get("end_time", "")
                if start and end:
                    info["查询时间窗口"] = f"{start}~{end}"

    # ── 计算震级差值 ──
    if info["USGS匹配震级"] != "-":
        try:
            claimed = None
            # 从 results 中找 magnitude claim 的 value
            for r in results:
                if r.get("fact_type") == "magnitude":
                    claimed = r.get("claim", {}).get("value")
                    break
            # 回退：从 input_text 解析
            if claimed is None:
                mag_m = re.search(r"发生\s*([\d.]+)\s*级地震", item.get("input_text", ""))
                claimed = float(mag_m.group(1)) if mag_m else None

            if claimed is not None:
                usgs_mag = float(info["USGS匹配震级"])
                diff = abs(float(claimed) - usgs_mag)
                info["震级差值"] = f"{diff:.1f}"
        except (ValueError, TypeError, KeyError):
            pass

    # ── 从 extraction.claims 提取信息 ──
    for c in claims:
        ft = c.get("fact_type", "")
        if ft == "occurrence":
            info["提取含occurrence"] = "是"
        elif ft == "magnitude":
            info["提取含magnitude"] = "是"

    if not info["输入标准地点"]:
        info["输入标准地点"] = original_location

    return info


# ── 综合判定逻辑 ──────────────────────────────────────────────────────

def determine_verdict(dataset_name: str, row: dict) -> tuple:
    """根据各字段推断综合判定、错误分类、根因分析、改进建议。"""
    occ = row["occurrence判定"]
    occ_reason = row.get("occurrence原因", "")
    mag = row["magnitude判定"]
    has_occ = row["提取含occurrence"]
    has_mag = row["提取含magnitude"]
    is_true = "真实" in dataset_name

    verdict = ""
    error_type = ""
    root_cause = ""
    suggestion = ""

    # ── 缺少 occurrence 提取 ──
    if has_occ == "否":
        if is_true:
            verdict = "错误-缺少occurrence验证"
            error_type = "提取不完整"
            root_cause = "LLM仅提取了magnitude claim, 未提取occurrence claim"
            suggestion = "优化prompt强制同时提取occurrence+magnitude; 添加后处理补全逻辑"
        else:
            # 伪造地震：根据 magnitude 是否有结果来区分
            if mag in ("supported", "contradicted"):
                verdict = "不完整-缺少occurrence"
                error_type = "提取不完整"
                root_cause = "occurrence claim缺失"
                suggestion = "同上"
            else:
                verdict = "不完整-仅查询震级"
                error_type = "提取不完整"
                root_cause = "LLM仅提取magnitude, 未提取occurrence; 无法判断地震是否发生"
                suggestion = "优化prompt确保occurrence+magnitude同时提取; 添加fallback逻辑"
        return verdict, error_type, root_cause, suggestion

    # ── occurrence = contradicted ──
    if occ == "contradicted":
        if is_true:
            verdict = "错误(FN)-真实地震被否认"
            claimed_mag = float(row["震级(声称)"]) if row["震级(声称)"] else 0
            error_type = "USGS查询未命中"
            root_cause = f"M{claimed_mag}事件在±3分钟、100km范围内未找到可信USGS记录"
            suggestion = "检查输入坐标与时间；保持USGS结果并进行人工复核"
        else:
            verdict = "正确(TN)-伪造被否认"
            error_type = "无错误"
            root_cause = "-"
            suggestion = "-"
        return verdict, error_type, root_cause, suggestion

    # ── occurrence = supported ──
    if occ == "supported":
        if is_true:
            if mag == "supported":
                verdict = "正确(TP)"
                error_type = "无错误"
                root_cause = "-"
                suggestion = "-"
            elif mag == "contradicted":
                verdict = "部分正确-震级矛盾"
                error_type = "震级不匹配"
                claimed = row["震级(声称)"]
                usgs_mag = row["USGS匹配震级"]
                diff = row["震级差值"]
                tolerance = row.get("震级容差", 0.5)
                root_cause = f"声称震级{claimed}, USGS匹配震级{usgs_mag}, 差值{diff}, 当前容差±{tolerance}"
                suggestion = "检查候选事件距离和地点匹配等级; 必要时人工核对震级标度"
            else:
                # magnitude unknown 但 occurrence supported
                verdict = "部分正确-仅occurrence确认"
                error_type = "震级数据缺失"
                root_cause = "occurrence被USGS确认但震级无法匹配"
                suggestion = "检查USGS震级数据完整性; 增加备选震级来源"
        else:
            verdict = "严重错误(FP)-伪造被确认"
            error_type = "假阳性"
            root_cause = "伪造地震在当前时空范围内匹配到USGS事件"
            suggestion = "检查事件距离、中文地点匹配等级和原始USGS证据"
        return verdict, error_type, root_cause, suggestion

    # ── occurrence = unknown / missing (但有 extraction) ──
    if occ_reason == "below_usgs_reliable_coverage":
        verdict = "无法核实-USGS覆盖不足"
        error_type = "USGS覆盖不足"
        root_cause = "声称震级低于USGS可靠覆盖阈值，未命中不能作为未发生的证据"
        suggestion = "保留unknown，等待国内区域地震源接入"
    else:
        verdict = "无法核实"
        error_type = "证据不足"
        root_cause = occ_reason or "USGS未返回可用判定"
        suggestion = "检查查询参数和外部服务状态"

    return verdict, error_type, root_cause, suggestion


# ── 主处理逻辑 ────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "序号", "数据集", "输入提问", "输入原始地点", "输入标准地点", "震级(声称)", "时间(北京)", "纬度", "经度",
    "occurrence判定", "occurrence原因", "magnitude判定", "震级容差", "USGS事件数", "USGS匹配震级", "震级差值",
    "USGS匹配地点", "事件距离(km)", "地点匹配等级", "地点核验提示", "查询时间窗口",
    "提取含occurrence", "提取含magnitude",
    "综合判定", "错误分类", "根因分析", "改进建议",
]


def process_pair(question_path: str, result_path: str, dataset_name: str) -> list:
    """处理一对 question + result，返回行列表。"""
    with open(question_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    texts = questions.get("texts", [])
    items = results.get("items", [])

    rows = []
    for i, item in enumerate(items):
        input_text = item.get("input_text", "")
        if i < len(texts) and not input_text:
            input_text = texts[i]

        # 从文本提取基本信息
        parsed = parse_input_text(input_text)

        # 从 result 提取验证信息
        result_info = extract_result_info(item, parsed["地点"])

        row = {
            "数据集": dataset_name,
            "输入提问": input_text,
            "输入原始地点": parsed["地点"],
            "震级(声称)": parsed["震级(声称)"],
            "时间(北京)": parsed["时间(北京)"],
            "纬度": parsed["纬度"],
            "经度": parsed["经度"],
            **result_info,
        }

        # 综合判定
        verdict, error_type, root_cause, suggestion = determine_verdict(dataset_name, row)
        row["综合判定"] = verdict
        row["错误分类"] = error_type
        row["根因分析"] = root_cause
        row["改进建议"] = suggestion

        rows.append(row)

    return rows


def infer_dataset_name(path: str) -> str:
    """从文件名推断数据集名称。"""
    basename = os.path.basename(path).lower()
    if "true" in basename or "真实" in basename:
        return "真实地震"
    elif "false" in basename or "伪造" in basename or "fake" in basename:
        return "伪造地震"
    return "未知数据集"


def write_csv(rows: list, output_path: str):
    """写入 CSV 文件。"""
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for idx, row in enumerate(rows, 1):
            row["序号"] = idx
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


def calculate_accuracy(rows: list) -> dict:
    """按严格口径和原宽松口径统计准确率。"""
    total = len(rows)
    strict_correct = sum(r.get("错误分类") == "无错误" for r in rows)
    complete_errors = sum(
        r.get("错误分类") in {"假阴性", "假阳性"}
        or str(r.get("综合判定", "")).startswith("严重错误")
        for r in rows
    )
    partial = sum(str(r.get("综合判定", "")).startswith("部分正确") for r in rows)
    unknown = sum(str(r.get("综合判定", "")).startswith("无法核实") for r in rows)
    return {
        "total": total,
        "strict_correct": strict_correct,
        "loose_correct": total - complete_errors,
        "complete_errors": complete_errors,
        "partial": partial,
        "unknown": unknown,
    }


def print_accuracy(label: str, rows: list):
    stats = calculate_accuracy(rows)
    total = stats["total"]
    if not total:
        return
    print(f"  {label}严格准确率: {stats['strict_correct']}/{total} "
          f"({stats['strict_correct']/total*100:.1f}%)")
    print(f"  {label}宽松准确率: {stats['loose_correct']}/{total} "
          f"({stats['loose_correct']/total*100:.1f}%)，原口径：只排除FN/FP")


def print_summary(rows: list):
    """打印汇总统计（按真实/伪造分组）。"""
    from collections import Counter

    true_rows = [r for r in rows if "真实" in r["数据集"]]
    false_rows = [r for r in rows if "伪造" in r["数据集"]]

    print(f"\n{'='*60}")
    print(f"总计: {len(rows)} 条 (真实 {len(true_rows)} / 伪造 {len(false_rows)})")
    print(f"{'='*60}")

    overall = calculate_accuracy(rows)
    print("\n【准确率】")
    print_accuracy("总体", rows)
    print(f"  其中部分正确: {overall['partial']}/{overall['total']} "
          f"({overall['partial']/overall['total']*100:.1f}%)")
    print(f"  其中无法核实: {overall['unknown']}/{overall['total']} "
          f"({overall['unknown']/overall['total']*100:.1f}%)，不代表完全正确")
    print(f"  完全错误(FN/FP): {overall['complete_errors']}/{overall['total']} "
          f"({overall['complete_errors']/overall['total']*100:.1f}%)")

    if true_rows:
        print(f"\n【真实地震】共 {len(true_rows)} 条")
        print_accuracy("", true_rows)
        print(f"  综合判定:")
        for k, v in Counter(r["综合判定"] for r in true_rows).most_common():
            print(f"    {k}: {v} ({v/len(true_rows)*100:.1f}%)")
        print(f"  错误分类:")
        for k, v in Counter(r["错误分类"] for r in true_rows).most_common():
            print(f"    {k}: {v}")


    if false_rows:
        print(f"\n【伪造地震】共 {len(false_rows)} 条")
        print_accuracy("", false_rows)
        print(f"  综合判定:")
        for k, v in Counter(r["综合判定"] for r in false_rows).most_common():
            print(f"    {k}: {v} ({v/len(false_rows)*100:.1f}%)")
        print(f"  错误分类:")
        for k, v in Counter(r["错误分类"] for r in false_rows).most_common():
            print(f"    {k}: {v}")


    print()


# ── CLI 入口 ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="地震幻觉检测结果分析 → CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("question", help="提问 JSON 文件路径 (含 texts 数组)")
    parser.add_argument("result", help="结果 JSON 文件路径 (含 items 数组)")
    parser.add_argument("-o", "--output", default=None, help="输出 CSV 路径 (默认: <result>_analysis.csv)")
    parser.add_argument("-n", "--name", default=None, help="数据集名称 (默认从文件名推断)")
    parser.add_argument("--second", nargs=2, metavar=("QUESTION2", "RESULT2"),
                        help="第二组 question + result (合并到同一 CSV)")
    parser.add_argument("--second-name", default=None, help="第二组数据集名称")
    parser.add_argument("--no-summary", action="store_true", help="不打印汇总统计")

    args = parser.parse_args()

    # 处理第一组
    dataset_name = args.name or infer_dataset_name(args.question)
    rows = process_pair(args.question, args.result, dataset_name)

    # 处理第二组（可选）
    if args.second:
        q2, r2 = args.second
        name2 = args.second_name or infer_dataset_name(q2)
        rows2 = process_pair(q2, r2, name2)
        rows.extend(rows2)

    # 输出路径
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.result))[0]
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{base}_analysis_{ts}.csv"

    write_csv(rows, output_path)
    print(f"已写入: {output_path} ({len(rows)} 行)")

    if not args.no_summary:
        print_summary(rows)


if __name__ == "__main__":
    main()
