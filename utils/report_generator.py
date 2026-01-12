# 报告生成工具
# utils/report_generator.py
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from core.schema import ProcessContext, QuestionType, GTStructureMap
from core.analyzer import GTAnalyzer

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, output_dir: Path):
        """
        初始化报告生成器
        :param output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.segments_dir = self.output_dir / "segments"
        self.segments_dir.mkdir(exist_ok=True)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
    
    def save_segments(self, ctx: ProcessContext):
        """
        保存切割下来的 pred 题目和 gt 题目
        """
        filename_base = Path(ctx.filename).stem
        
        # 保存 GT 题目
        if ctx.gt_clean:
            gt_file = self.segments_dir / f"{filename_base}_gt.md"
            gt_file.write_text(ctx.gt_clean, encoding='utf-8')
            logger.debug(f"Saved GT segment: {gt_file}")
        
        # 保存 Pred 题目片段（所有题目）
        if ctx.all_question_segments:
            # 保存所有题目的 segments
            for i, question in enumerate(ctx.all_question_segments):
                question_id = question.get("question_id", str(i + 1))
                pred_segment = question.get("pred_segment")
                if pred_segment:
                    pred_file = self.segments_dir / f"{filename_base}_pred_q{question_id}.md"
                    pred_file.write_text(pred_segment, encoding='utf-8')
                    logger.debug(f"Saved Pred segment for question {question_id}: {pred_file}")
        elif ctx.pred_segment:
            # 向后兼容：如果没有 all_question_segments，保存第一个题目
            pred_file = self.segments_dir / f"{filename_base}_pred.md"
            pred_file.write_text(ctx.pred_segment, encoding='utf-8')
            logger.debug(f"Saved Pred segment: {pred_file}")
        
        # 保存原始 GT 和 Pred（用于对比）
        if ctx.gt_raw:
            gt_raw_file = self.segments_dir / f"{filename_base}_gt_raw.md"
            gt_raw_file.write_text(ctx.gt_raw, encoding='utf-8')
        
        if ctx.pred_raw:
            pred_raw_file = self.segments_dir / f"{filename_base}_pred_raw.md"
            pred_raw_file.write_text(ctx.pred_raw, encoding='utf-8')
    
    def generate_single_report(self, ctx: ProcessContext) -> dict:
        """
        生成单个题目的对比报告，包含详细的匹配信息
        """
        # 构建详细的匹配内容
        matched_content = {
            "gt_question_full": ctx.gt_clean,  # GT 题目的完整内容（清理后）
            "pred_question_segment": ctx.pred_segment,  # Pred 题目的匹配片段（第一个题目）
            "gt_structure_map": {k: v.value for k, v in ctx.gt_structure_map.items()},  # 题号-题型映射
        }
        # 添加所有题目的对齐信息（如果存在）
        if ctx.all_question_segments:
            matched_content["all_questions"] = ctx.all_question_segments
        # 添加第一个题目的对齐锚点信息（如果存在）
        if ctx.alignment_anchors:
            matched_content["alignment_anchors"] = ctx.alignment_anchors
        
        report = {
            "filename": ctx.filename,
            "question_type": ctx.gt_type.value if ctx.gt_type else "Unknown",
            "metrics": ctx.metrics.copy(),
            "confusion_matrix": {
                "tp": ctx.metrics.get("tp", 0.0),
                "fp": ctx.metrics.get("fp", 0.0),
                "tn": ctx.metrics.get("tn", 0.0),
                "fn": ctx.metrics.get("fn", 0.0),
            },
            "alignment": {
                "found": ctx.alignment_found,
                "is_rejected": ctx.pred_is_rejected,
                "stats": ctx.alignment_stats if ctx.alignment_stats else {
                    "total_aligned": 0,
                    "exact_match_count": 0,
                    "fuzzy_match_count": 0,
                    "fallback_next_question_count": 0,
                    "fallback_text_end_count": 0,
                    "failed_alignment_count": 0,
                    "total_questions": 0
                },
            },
            "answers": {
                "gt_answer_truth": ctx.gt_answer_truth,
                "pred_raw_handwritten": ctx.pred_raw_handwritten,
                "gt_extracted_answers": ctx.gt_extracted_answers,
                "pred_extracted_answers": ctx.pred_extracted_answers,
            },
            "images": {
                "gt_img_paths": ctx.gt_img_paths,
                "pred_img_paths": ctx.pred_img_paths,
            },
            "matched_content": matched_content  # 详细的匹配内容
        }
        return report
    
    def _is_question_rejected(self, ctx: ProcessContext, q_num: str) -> bool:
        """
        检查某个题目是否在GT中被标记为"无法识别"
        :param ctx: 处理上下文
        :param q_num: 题号
        :return: 是否被拒绝
        """
        # 检查整个文件是否被拒绝
        if ctx.metrics.get("gt_rejected_flag", 0) == 1.0:
            return True
        
        if not ctx.gt_clean:
            return False
        
        # 优先从all_question_segments中获取信息
        gt_start_snippet = ""
        gt_end_snippet = ""
        if ctx.all_question_segments:
            for question in ctx.all_question_segments:
                if question.get("question_id") == q_num:
                    gt_start_snippet = question.get("gt_start_snippet", "")
                    gt_end_snippet = question.get("gt_end_snippet", "")
                    break
        
        # 如果有snippet，使用snippet来检查
        if gt_start_snippet:
            import re
            start_idx = ctx.gt_clean.find(gt_start_snippet)
            if start_idx != -1:
                # 找到结尾
                end_idx = ctx.gt_clean.find(gt_end_snippet, start_idx) if gt_end_snippet else -1
                if end_idx == -1:
                    # 尝试找到下一个题目的开头
                    try:
                        next_q_num = str(int(q_num) + 1)
                        next_question_pattern = f"\n{next_q_num}."
                        next_idx = ctx.gt_clean.find(next_question_pattern, start_idx)
                        if next_idx != -1:
                            end_idx = next_idx
                        else:
                            end_idx = len(ctx.gt_clean)
                    except ValueError:
                        end_idx = len(ctx.gt_clean)
                else:
                    end_idx = end_idx + len(gt_end_snippet)
                
                # 提取题目内容
                question_content = ctx.gt_clean[start_idx:end_idx]
                
                # 检查是否包含拒答标记
                rejection_keywords = ["[无法识别]", "无法识别", "无法辨认", "无法读取"]
                if any(keyword in question_content for keyword in rejection_keywords):
                    return True
        else:
            # 如果没有snippet，尝试从gt_clean中查找题目
            import re
            question_patterns = [
                f"{q_num}.",
                f"{q_num}、",
                f"**{q_num}.**",
                f"**{q_num}.",
                f"{q_num}.**",
            ]
            question_start = -1
            for pattern in question_patterns:
                question_start = ctx.gt_clean.find(pattern)
                if question_start != -1:
                    break
            
            if question_start == -1:
                pattern = re.compile(rf'\*?{re.escape(q_num)}\*?[\.。、]')
                match = pattern.search(ctx.gt_clean)
                if match:
                    question_start = match.start()
            
            if question_start != -1:
                # 找到下一个题目的开头或文本末尾
                try:
                    next_q_num = str(int(q_num) + 1)
                    next_patterns = [
                        f"{next_q_num}.",
                        f"{next_q_num}、",
                        f"**{next_q_num}.**",
                    ]
                    question_end = -1
                    for pattern in next_patterns:
                        question_end = ctx.gt_clean.find(pattern, question_start)
                        if question_end != -1:
                            break
                    
                    if question_end == -1:
                        pattern = re.compile(rf'\*?{re.escape(next_q_num)}\*?[\.。、]')
                        match = pattern.search(ctx.gt_clean, question_start)
                        if match:
                            question_end = match.start()
                    
                    if question_end == -1:
                        question_end = len(ctx.gt_clean)
                except ValueError:
                    question_end = len(ctx.gt_clean)
                
                # 提取题目内容
                question_content = ctx.gt_clean[question_start:question_end]
                
                # 检查是否包含拒答标记
                rejection_keywords = ["[无法识别]", "无法识别", "无法辨认", "无法读取"]
                if any(keyword in question_content for keyword in rejection_keywords):
                    return True
        
        return False
    
    def _count_questions_from_gt_files(self, all_results: List[ProcessContext], failed_files_list: Optional[List[Dict]] = None) -> int:
        """
        从GT文件直接统计所有题目数（包括处理失败的文件）
        
        重要：无论对齐是否成功，所有GT中的题目都会被计入total_count。
        gt_structure_map是在对齐之前（Step 2）通过analyzer.analyze()设置的，不依赖于对齐结果。
        因此，即使对齐失败（没有匹配成功），GT中的题目也会被正确统计。
        
        :param all_results: 成功处理的文件结果列表（包括对齐成功和失败的文件）
        :param failed_files_list: 失败的文件信息列表（处理异常的文件）
        :return: 总题目数（所有GT文件中的题目数，无论对齐是否成功）
        """
        analyzer = GTAnalyzer()
        total_count = 0
        processed_files = set()
        
        # 从成功处理的文件中统计（无论对齐是否成功）
        # 注意：gt_structure_map在对齐之前就已经设置，所以即使对齐失败，也能正确统计题目数
        for ctx in all_results:
            if ctx.gt_structure_map:
                # gt_structure_map包含该文件中所有GT题目的题号和题型映射
                # 无论对齐是否成功，这些题目都应该被计入total_count
                file_question_count = len(ctx.gt_structure_map)
                total_count += file_question_count
                processed_files.add(ctx.filename)
                logger.debug(f"从成功处理的文件统计题目数: {ctx.filename}, 题目数: {file_question_count} (对齐状态: {ctx.alignment_found})")
            elif ctx.gt_raw:
                # 如果没有 structure_map，尝试从GT文本中分析
                # 这种情况不应该发生，因为gt_structure_map应该在Step 2就设置了
                try:
                    structure_map = analyzer.analyze(ctx.gt_raw)
                    file_question_count = len(structure_map)
                    total_count += file_question_count
                    processed_files.add(ctx.filename)
                    logger.warning(f"文件 {ctx.filename} 没有gt_structure_map，从GT文本重新分析，题目数: {file_question_count}")
                except Exception as e:
                    logger.warning(f"无法从GT文本分析题目数: {ctx.filename}, 错误: {e}")
        
        # 从失败的文件中统计（如果提供了GT文件路径）
        # 这些是处理异常的文件（如文件读取失败、分析失败等）
        if failed_files_list:
            for failure_info in failed_files_list:
                filename = failure_info.get("filename", "")
                if filename in processed_files:
                    continue  # 已经统计过了
                
                gt_file_path = failure_info.get("gt_file_path")
                if gt_file_path:
                    try:
                        gt_path = Path(gt_file_path)
                        if gt_path.exists():
                            gt_raw = gt_path.read_text(encoding='utf-8')
                            structure_map = analyzer.analyze(gt_raw)
                            file_question_count = len(structure_map)
                            total_count += file_question_count
                            processed_files.add(filename)
                            logger.debug(f"从失败文件中统计题目数: {filename}, 题目数: {file_question_count}")
                        else:
                            logger.warning(f"失败文件的GT文件路径不存在: {gt_file_path}")
                    except Exception as e:
                        logger.warning(f"无法从失败文件的GT文件统计题目数: {filename}, 错误: {e}")
                else:
                    logger.debug(f"失败文件 {filename} 没有提供GT文件路径，无法统计题目数")
        
        logger.info(f"总题目数统计完成: {total_count} 道题目（包括所有GT文件，无论对齐是否成功）")
        return total_count
    
    def generate_summary_report(self, all_results: List[ProcessContext], failed_files_list: Optional[List[Dict]] = None) -> dict:
        """
        生成汇总报告，按题型分组统计
        注意：统计的是题目数，不是文件数
        
        :param all_results: 成功处理的文件结果列表
        :param failed_files_list: 失败的文件信息列表（用于统计总题目数）
        """
        # 初始化容器，用于收集所有有效题目的分数（排除GT中无法识别的题目）
        overall_stats = {
            "stem_sims": [],      # 题目文本相似度
            "img_sims": [],       # 图片相似度 (按题目加权)
            "ans_accs": [],       # 答案准确率
            "ext_accs": []        # 答案提取准确率 (按题目加权)
        }
        
        # 收集对齐统计信息
        alignment_stats_total = {
            "total_aligned": 0,
            "exact_match_count": 0,
            "fuzzy_match_count": 0,
            "fallback_next_question_count": 0,
            "fallback_text_end_count": 0,
            "failed_alignment_count": 0,
            "total_questions": 0
        }
        
        for ctx in all_results:
            # 收集对齐统计信息
            if ctx.alignment_stats:
                alignment_stats_total["total_aligned"] += ctx.alignment_stats.get("total_aligned", 0)
                alignment_stats_total["exact_match_count"] += ctx.alignment_stats.get("exact_match_count", 0)
                alignment_stats_total["fuzzy_match_count"] += ctx.alignment_stats.get("fuzzy_match_count", 0)
                alignment_stats_total["fallback_next_question_count"] += ctx.alignment_stats.get("fallback_next_question_count", 0)
                alignment_stats_total["fallback_text_end_count"] += ctx.alignment_stats.get("fallback_text_end_count", 0)
                alignment_stats_total["failed_alignment_count"] += ctx.alignment_stats.get("failed_alignment_count", 0)
                alignment_stats_total["total_questions"] += ctx.alignment_stats.get("total_questions", 0)
            
            # 1. 如果整个文件被标记为拒答/无效，跳过该文件（这些题目不参与相似度计算）
            if ctx.metrics.get("gt_rejected_flag", 0) == 1.0:
                continue
            
            # 获取文件级别的指标 (用于分配给每个题目)
            file_img_sim = ctx.metrics.get("img_sim", -1.0)
            file_ext_acc = ctx.metrics.get("answer_extraction_acc", -1.0)
            
            # 2. 遍历该文件下的所有题目片段
            if ctx.all_question_segments:
                for question in ctx.all_question_segments:
                    question_id = question.get("question_id", "1")
                    
                    # 检查该题目是否在GT中被标记为"无法识别"
                    is_gt_rejected = self._is_question_rejected(ctx, question_id)
                    
                    # 如果GT被拒绝，跳过该题目的相似度统计
                    if is_gt_rejected:
                        continue
                    
                    # [A] 收集文本相似度 (Question Level) - 只收集GT正常的题目
                    if "stem_sim" in question:
                        overall_stats["stem_sims"].append(question["stem_sim"])
                    
                    # [B] 收集答案准确率 (Question Level) - 只收集GT正常的题目
                    # 只有当该题计算了答案准确率（非-1）时才统计
                    if "ans_acc" in question and question["ans_acc"] != -1.0:
                        overall_stats["ans_accs"].append(question["ans_acc"])
                        
                    # [C] 收集图片相似度 (File Level -> Weighted by Question)
                    # 如果文件有图片相似度，视为该文件的每个题目都继承这个相似度
                    if file_img_sim != -1.0:
                        overall_stats["img_sims"].append(file_img_sim)
                        
                    # [D] 收集答案提取准确率 (File Level -> Weighted by Question)
                    if file_ext_acc != -1.0:
                        overall_stats["ext_accs"].append(file_ext_acc)
            else:
                # 兼容性处理：如果 old version 没有 all_question_segments，回退到文件级
                # 假设该文件只有 1 道题
                if ctx.metrics.get("stem_sim", -1) != -1:
                    overall_stats["stem_sims"].append(ctx.metrics["stem_sim"])
                if ctx.metrics.get("ans_acc", -1) != -1:
                    overall_stats["ans_accs"].append(ctx.metrics["ans_acc"])
                if file_img_sim != -1.0:
                    overall_stats["img_sims"].append(file_img_sim)
                if file_ext_acc != -1.0:
                    overall_stats["ext_accs"].append(file_ext_acc)

        # 计算算术平均值
        def calc_avg(values):
            return sum(values) / len(values) if values else 0.0

        # 如果列表为空（比如所有都是纯文本无图），img_sim 返回 -1.0 以示不适用
        avg_img_sim = calc_avg(overall_stats["img_sims"]) if overall_stats["img_sims"] else -1.0
        avg_ans_acc = calc_avg(overall_stats["ans_accs"]) if overall_stats["ans_accs"] else -1.0
        avg_ext_acc = calc_avg(overall_stats["ext_accs"]) if overall_stats["ext_accs"] else -1.0
        
        # 按每个题目的实际类型分组（而不是按文件的主要题型）
        # 收集所有文件，后续按题目级别分组
        all_valid_contexts = []
        
        rejected_question_count = 0
        # 从GT文件直接统计所有题目数（包括处理失败的文件）
        # 注意：处理失败的文件也要计入混淆矩阵，因为它们的GT是有效的
        total_question_count = self._count_questions_from_gt_files(all_results, failed_files_list)
        
        # 统计成功处理的文件中的题目数（用于验证）
        processed_question_count = 0
        for ctx in all_results:
            # 统计该文件中的题目数
            if ctx.gt_structure_map:
                file_question_count = len(ctx.gt_structure_map)
            else:
                # 如果没有 structure_map，尝试从 extracted_answers 统计
                file_question_count = len(ctx.gt_extracted_answers) if ctx.gt_extracted_answers else 1
            
            processed_question_count += file_question_count
            
            if ctx.metrics.get("gt_rejected_flag", 0) == 1.0:
                # 如果整个文件被拒绝，所有题目都算作拒绝
                rejected_question_count += file_question_count
                continue
            
            # 统计单个题目级别的拒绝数量（GT标记为'[无法识别]'的题目）
            if ctx.gt_structure_map:
                for q_num in ctx.gt_structure_map.keys():
                    if self._is_question_rejected(ctx, q_num):
                        rejected_question_count += 1
            else:
                # 如果没有gt_structure_map，使用原来的逻辑（向后兼容）
                if ctx.all_question_segments:
                    for question in ctx.all_question_segments:
                        gt_start_snippet = question.get("gt_start_snippet", "")
                        gt_end_snippet = question.get("gt_end_snippet", "")
                        # 检查GT中该题目是否被拒答
                        if "[无法识别]" in gt_start_snippet or "[无法识别]" in gt_end_snippet:
                            rejected_question_count += 1
            
            # 收集所有有效文件（不管主要题型是什么）
            all_valid_contexts.append(ctx)
        
        # 汇总混淆矩阵（整体）- 先统计成功处理的文件
        total_tp = sum(ctx.metrics.get("tp", 0.0) for ctx in all_results)
        total_fp = sum(ctx.metrics.get("fp", 0.0) for ctx in all_results)
        total_tn = sum(ctx.metrics.get("tn", 0.0) for ctx in all_results)
        total_fn = sum(ctx.metrics.get("fn", 0.0) for ctx in all_results)
        
        # 按题型汇总混淆矩阵
        confusion_matrix_by_type_sum = {
            "Choice": {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0},
            "Fill": {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0},
            "Subjective": {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0},
        }
        
        # 统计每个文件按题型的混淆矩阵，并验证总数
        for ctx in all_results:
            cm_by_type = ctx.metrics.get("confusion_matrix_by_type", {})
            # 计算该文件按题型统计的总数
            file_cm_by_type_total = {}
            for type_name, cm in cm_by_type.items():
                if type_name in confusion_matrix_by_type_sum:
                    type_total = cm.get("tp", 0.0) + cm.get("fp", 0.0) + cm.get("tn", 0.0) + cm.get("fn", 0.0)
                    file_cm_by_type_total[type_name] = type_total
                    confusion_matrix_by_type_sum[type_name]["tp"] += cm.get("tp", 0.0)
                    confusion_matrix_by_type_sum[type_name]["fp"] += cm.get("fp", 0.0)
                    confusion_matrix_by_type_sum[type_name]["tn"] += cm.get("tn", 0.0)
                    confusion_matrix_by_type_sum[type_name]["fn"] += cm.get("fn", 0.0)
            
            # 注意：evaluator.py已经遍历了gt_structure_map中的所有题目，
            # 包括对齐失败的题目，所以混淆矩阵应该已经完整了，不需要在这里补充
            # 如果发现不一致，可能是代码逻辑有问题，记录警告即可
            if ctx.gt_structure_map:
                # 统计该文件中各题型的题目数
                file_questions_by_type = {}
                for q_id, q_type in ctx.gt_structure_map.items():
                    type_name = q_type.value if hasattr(q_type, 'value') else str(q_type)
                    file_questions_by_type[type_name] = file_questions_by_type.get(type_name, 0) + 1
                
                # 检查每个题型的统计是否一致
                for type_name, question_count in file_questions_by_type.items():
                    cm_total = file_cm_by_type_total.get(type_name, 0)
                    if question_count != cm_total:
                        logger.warning(
                            f"文件 {ctx.filename}: 题型 {type_name} 的题目数 ({question_count}) "
                            f"与混淆矩阵统计数 ({cm_total}) 不一致，差异 {question_count - cm_total} 道题目。"
                            f"这可能表明evaluator.py中的混淆矩阵统计逻辑有问题。"
                        )
                        
        # 为处理失败的文件也创建混淆矩阵条目
        # 注意：需要排除已经在all_results中的文件（对齐失败但仍返回了context的文件）
        # 只为真正处理失败（异常、未返回context）的文件补充混淆矩阵
        failed_files_question_count = 0  # 统计失败文件中的题目数
        if failed_files_list:
            # 收集all_results中的文件名，避免重复统计
            processed_filenames = set(ctx.filename for ctx in all_results if ctx.filename)
            
            logger.info(f"开始处理 {len(failed_files_list)} 个失败文件，检查是否需要补充混淆矩阵条目...")
            analyzer = GTAnalyzer()
            actually_failed_count = 0  # 真正需要补充的失败文件数
            
            for failure_info in failed_files_list:
                filename = failure_info.get("filename")
                
                # 如果该文件已经在all_results中，跳过（避免重复统计）
                if filename in processed_filenames:
                    logger.debug(f"文件 {filename} 已在 all_results 中，跳过重复统计")
                    continue
                            
                actually_failed_count += 1
                gt_file_path = failure_info.get("gt_file_path")
                if not gt_file_path:
                    continue
                
                try:
                    gt_path = Path(gt_file_path)
                    if not gt_path.exists():
                        logger.warning(f"失败文件的GT文件路径不存在: {gt_file_path}")
                        continue
                    
                    # 读取GT文件内容
                    gt_raw = gt_path.read_text(encoding='utf-8')
                    gt_clean = gt_raw  # 简化处理，使用原始文本
                    
                    # 分析GT文件，获取题目结构
                    structure_map = analyzer.analyze(gt_raw)
                    if not structure_map:
                        logger.debug(f"失败文件 {filename} 的GT文件中没有识别到题目")
                        continue
                    
                    # 为每个题目创建混淆矩阵条目
                    for question_id, q_type in structure_map.items():
                        # 判断GT中该题目是否被拒答
                        from core.schema import ProcessContext
                        temp_ctx = ProcessContext(
                            filename=filename,
                            gt_raw=gt_raw,
                            pred_raw="",  # 失败的文件没有pred_raw，提供空字符串
                            gt_clean=gt_clean,
                            gt_structure_map=structure_map
                        )
                        is_gt_q_rejected = self._is_question_rejected(temp_ctx, question_id)
                        
                        # 对于失败的文件，Pred总是拒答（因为处理失败）
                        # 所以：GT无法识别 -> TP, GT正常 -> FP
                        if is_gt_q_rejected:
                            total_tp += 1
                        else:
                            total_fp += 1
                        
                        failed_files_question_count += 1
                        
                        # 按题型统计
                        type_name = q_type.value if hasattr(q_type, 'value') else str(q_type)
                        if type_name in confusion_matrix_by_type_sum:
                            if is_gt_q_rejected:
                                confusion_matrix_by_type_sum[type_name]["tp"] += 1
                            else:
                                confusion_matrix_by_type_sum[type_name]["fp"] += 1
                        else:
                            logger.debug(f"失败文件中的题目类型 {type_name} 不在预定义类型中，跳过按题型统计")
                    
                    logger.debug(f"为失败文件 {filename} 添加了 {len(structure_map)} 道题目的混淆矩阵条目")
                    
                except Exception as e:
                    logger.warning(f"无法为失败文件 {filename} 创建混淆矩阵条目: {e}")
            
            if actually_failed_count > 0:
                logger.info(f"为 {actually_failed_count} 个真正失败的文件补充了混淆矩阵条目，共 {failed_files_question_count} 道题目")
            else:
                logger.info(f"所有失败文件都已在 all_results 中，无需补充混淆矩阵条目")
        
        # 按题型计算Precision, Recall, F1
        precision_by_type = {}
        recall_by_type = {}
        f1_by_type = {}
        
        for type_name, cm in confusion_matrix_by_type_sum.items():
            type_tp = cm["tp"]
            type_fp = cm["fp"]
            type_fn = cm["fn"]
            
            type_precision = type_tp / (type_tp + type_fp) if (type_tp + type_fp) > 0 else 0.0
            type_recall = type_tp / (type_tp + type_fn) if (type_tp + type_fn) > 0 else 0.0
            type_f1 = 2 * (type_precision * type_recall) / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0.0
            
            precision_by_type[type_name] = type_precision
            recall_by_type[type_name] = type_recall
            f1_by_type[type_name] = type_f1
        
        # 使用混淆矩阵的统计结果，确保数据一致性
        # 无法识别的题目数 = TP + FN（GT无法识别）
        # 有效题目数 = FP + TN（GT正常）
        rejected_count_from_cm = int(total_tp + total_fn)
        valid_count_from_cm = int(total_fp + total_tn)
        
        # 验证：总题目数应该等于无法识别题目数 + 有效题目数
        total_from_cm = rejected_count_from_cm + valid_count_from_cm
        
        # 验证：混淆矩阵总数应该等于从GT文件统计的总题目数
        # 现在混淆矩阵已经包含了所有文件（包括失败的文件），所以应该一致
        if total_question_count != total_from_cm:
            logger.error(
                f"混淆矩阵总数不一致: "
                f"从GT文件统计的题目数 ({total_question_count}) "
                f"与混淆矩阵统计的题目数 ({total_from_cm} = TP:{total_tp} + FP:{total_fp} + TN:{total_tn} + FN:{total_fn}) 不一致。"
                f"差异: {total_question_count - total_from_cm} 道题目"
            )
            # 使用混淆矩阵的结果作为总题目数（确保一致性）
            final_total_count = total_from_cm
            logger.warning(f"使用混淆矩阵统计的题目数作为总题目数: {final_total_count}")
        else:
            # 验证通过，使用从GT文件统计的结果
            final_total_count = total_question_count
            logger.info(f"✓ 验证通过: 混淆矩阵总数 ({total_from_cm}) = 从GT文件统计的题目数 ({total_question_count})")
        
        # 如果计算不一致，记录警告（但使用混淆矩阵的结果）
        if rejected_question_count != rejected_count_from_cm:
            logger.warning(
                f"整体统计: rejected_question_count ({rejected_question_count}) "
                f"与混淆矩阵统计的无法识别题目数 ({rejected_count_from_cm}) 不一致。"
                f"使用混淆矩阵的结果：无法识别题目数 = {rejected_count_from_cm}, 有效题目数 = {valid_count_from_cm}"
            )
        
        # 初始化未分类题目列表（将在后续循环中填充）
        unclassified_questions = []
        
        summary = {
            "total_count": final_total_count,  # 总题目数（使用混淆矩阵统计的结果，确保一致性：TP+FP+TN+FN）
            "total_count_from_gt_files": total_question_count,  # 从GT文件统计的题目数（应该等于total_count）
            "processed_count": processed_question_count,  # 成功处理的文件中的题目数
            "rejected_count": rejected_count_from_cm,  # 无法识别的题目数（使用混淆矩阵的统计结果：TP+FN）
            "valid_count": valid_count_from_cm,  # 有效题目数（使用混淆矩阵的统计结果：FP+TN）
            "unclassified_count": 0,  # 未分类的题目数（将在后续循环中统计）
            "alignment_stats": alignment_stats_total,  # 对齐统计信息
            "confusion_matrix": {
                "tp": float(total_tp),  # True Positive: GT无法识别，Pred也拒答
                "fp": float(total_fp),  # False Positive: GT正常，但Pred拒答或未匹配
                "tn": float(total_tn),  # True Negative: GT正常，Pred也正常
                "fn": float(total_fn),  # False Negative: GT无法识别，但Pred正常
            },
            "confusion_matrix_by_type": confusion_matrix_by_type_sum,
            "precision_by_type": precision_by_type,
            "recall_by_type": recall_by_type,
            "f1_by_type": f1_by_type,
            # 整体平均值（所有文件，不按题型分组，排除GT无法识别的题目）
            "overall": {
                "avg_stem_sim": calc_avg(overall_stats["stem_sims"]), 
                "avg_img_sim": avg_img_sim,
                "avg_ans_acc": avg_ans_acc,
                "avg_answer_extraction_acc": avg_ext_acc,
            },
            "by_type": {}
        }
        
        # 计算Precision, Recall, F1（基于混淆矩阵）
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        summary["precision"] = precision
        summary["recall"] = recall
        summary["f1"] = f1
        
        # 按每个题目的实际类型分组计算指标（遍历所有文件，按题目类型分组）
        questions_by_type = {
            QuestionType.MCQ: {"stem_sims": [], "ans_accs": [], "img_sims": [], "answer_extraction_accs": [], "all_question_ids": [], "rejected_question_ids": []},
            QuestionType.FIB: {"stem_sims": [], "ans_accs": [], "img_sims": [], "answer_extraction_accs": [], "all_question_ids": [], "rejected_question_ids": []},
            QuestionType.SUB: {"stem_sims": [], "ans_accs": [], "img_sims": [], "all_question_ids": [], "rejected_question_ids": []},
        }
        
        # 遍历所有成功处理的文件（包括整个文件被拒绝的情况），按每个题目的实际类型分组
        # unclassified_questions 已在前面初始化
        for ctx in all_results:
            # 检查整个文件是否被拒绝
            is_file_rejected = ctx.metrics.get("gt_rejected_flag", 0) == 1.0
            
            if ctx.gt_structure_map:
                # 遍历该文件中的每个题目
                for q_num, q_type_in_map in ctx.gt_structure_map.items():
                    # 检查题目类型是否有效
                    if q_type_in_map not in questions_by_type:
                        # 记录未分类的题目
                        unclassified_questions.append({
                            "filename": ctx.filename,
                            "question_id": q_num,
                            "question_type": str(q_type_in_map) if q_type_in_map else "None"
                        })
                        logger.warning(
                            f"文件 {ctx.filename} 中的题目 {q_num} 的类型 {q_type_in_map} 不在预期的类型中 "
                            f"(MCQ/FIB/SUB)。该题目将被忽略。"
                        )
                        continue  # 跳过未分类的题目
                    
                    # 记录所有题目（包括[无法识别]的）
                    questions_by_type[q_type_in_map]["all_question_ids"].append(q_num)
                    
                    # 检查该题目是否在GT中被标记为[无法识别]
                    is_gt_rejected = self._is_question_rejected(ctx, q_num)
                    
                    # 如果题目被标记为"无法识别"，添加到 rejected_question_ids
                    if is_gt_rejected:
                        questions_by_type[q_type_in_map]["rejected_question_ids"].append(q_num)
                        continue  # 跳过该题目的相似度统计
                    
                    # 从 all_question_segments 中提取该题目的指标（只统计GT正常的题目）
                    if ctx.all_question_segments:
                        for question in ctx.all_question_segments:
                            if question.get("question_id") == q_num:
                                # 收集该题目的相似度
                                if "stem_sim" in question:
                                    questions_by_type[q_type_in_map]["stem_sims"].append(question["stem_sim"])
                                # 收集该题目的答案准确率
                                if "ans_acc" in question and question["ans_acc"] != -1.0:
                                    questions_by_type[q_type_in_map]["ans_accs"].append(question["ans_acc"])
            else:
                # 如果没有 structure_map，使用文件的主要题型
                q_type = ctx.gt_type if ctx.gt_type else QuestionType.MCQ
                # 统计该文件的所有题目数（如果没有structure_map，假设只有1个题目）
                questions_by_type[q_type]["all_question_ids"].append(ctx.filename)
                
                # 检查该文件是否在GT中被标记为[无法识别]
                if ctx.metrics.get("gt_rejected_flag", 0) == 1.0:
                    questions_by_type[q_type]["rejected_question_ids"].append(ctx.filename)
                else:
                    # 只有GT正常的题目才统计相似度
                    if ctx.all_question_segments:
                        for question in ctx.all_question_segments:
                            # 收集该题目的相似度
                            if "stem_sim" in question:
                                questions_by_type[q_type]["stem_sims"].append(question["stem_sim"])
                            if "ans_acc" in question and question["ans_acc"] != -1.0:
                                questions_by_type[q_type]["ans_accs"].append(question["ans_acc"])
            
            # 图片相似度和答案提取准确率按文件计算（因为它们是文件级别的指标）
            # 需要判断该文件是否包含该题型，并且只有GT正常的题目才统计
            if not is_file_rejected:
                if ctx.gt_structure_map:
                    file_has_types = set(ctx.gt_structure_map.values())
                    # 过滤掉未分类的类型
                    file_has_types = {q_type for q_type in file_has_types if q_type in questions_by_type}
                else:
                    file_has_types = {ctx.gt_type} if ctx.gt_type and ctx.gt_type in questions_by_type else {QuestionType.MCQ}
                
                for q_type in file_has_types:
                    if ctx.metrics.get("img_sim", -1) != -1.0:
                        questions_by_type[q_type]["img_sims"].append(ctx.metrics.get("img_sim"))
                    if q_type in [QuestionType.MCQ, QuestionType.FIB]:
                        if ctx.metrics.get("answer_extraction_acc", -1) != -1.0:
                            questions_by_type[q_type]["answer_extraction_accs"].append(ctx.metrics.get("answer_extraction_acc"))
        
        # 处理失败文件：将真正失败的文件的题目也加入到all_question_ids中
        # 注意：只统计不在all_results中的失败文件（避免重复）
        if failed_files_list:
            # 收集all_results中的文件名
            processed_filenames = set(ctx.filename for ctx in all_results if ctx.filename)
            
            analyzer = GTAnalyzer()
            for failure_info in failed_files_list:
                filename = failure_info.get("filename")
                
                # 如果该文件已经在all_results中，跳过（避免重复统计）
                if filename in processed_filenames:
                    continue
                
                gt_file_path = failure_info.get("gt_file_path")
                if not gt_file_path:
                    continue
                
                try:
                    gt_path = Path(gt_file_path)
                    if not gt_path.exists():
                        continue
                    
                    # 读取GT文件内容并分析题目结构
                    gt_raw = gt_path.read_text(encoding='utf-8')
                    structure_map = analyzer.analyze(gt_raw)
                    
                    if structure_map:
                        # 将失败文件的题目加入到all_question_ids中
                        for q_num, q_type_in_map in structure_map.items():
                            # 如果题目类型在预定义类型中，添加到对应的列表
                            if q_type_in_map in questions_by_type:
                                questions_by_type[q_type_in_map]["all_question_ids"].append(f"{filename}_{q_num}")
                                
                                # 判断该题目是否在GT中被标记为[无法识别]
                                from core.schema import ProcessContext
                                temp_ctx = ProcessContext(
                                    filename=filename,
                                    gt_raw=gt_raw,
                                    pred_raw="",
                                    gt_clean=gt_raw,
                                    gt_structure_map=structure_map
                                )
                                is_gt_rejected = self._is_question_rejected(temp_ctx, q_num)
                                if is_gt_rejected:
                                    questions_by_type[q_type_in_map]["rejected_question_ids"].append(f"{filename}_{q_num}")
                            else:
                                # 未分类的题目类型
                                type_name = q_type_in_map.value if hasattr(q_type_in_map, 'value') else str(q_type_in_map)
                                unclassified_questions.append({
                                    "filename": filename,
                                    "question_id": q_num,
                                    "question_type": type_name
                                })
                except Exception as e:
                    logger.warning(f"无法处理失败文件 {filename}: {e}")
        
        # 更新summary中的未分类题目数
        summary["unclassified_count"] = len(unclassified_questions)
        
        # 如果有未分类的题目，记录警告
        if unclassified_questions:
            logger.warning(
                f"发现 {len(unclassified_questions)} 道未分类的题目。"
                f"这些题目在总题目数中被统计，但在按题型统计时被忽略。"
                f"未分类题目详情: {unclassified_questions[:10]}"  # 只显示前10个
            )
        
        # 计算各题型的平均值
        for q_type in [QuestionType.MCQ, QuestionType.FIB, QuestionType.SUB]:
            type_name = q_type.value
            type_data = questions_by_type[q_type]
            
            # 统计题目数：所有题目（包括[无法识别]的）
            # 这个统计源：遍历所有文件的 gt_structure_map，将 q_type_in_map == q_type 的题目添加到 all_question_ids
            type_question_count = len(type_data.get("all_question_ids", []))
            
            # 有效题目数：使用混淆矩阵的统计结果
            # 这个统计源：从所有文件的 ctx.metrics["confusion_matrix_by_type"][type_name] 中汇总
            cm = confusion_matrix_by_type_sum.get(type_name, {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0})
            tp_count = cm.get("tp", 0.0)
            fp_count = cm.get("fp", 0.0)
            tn_count = cm.get("tn", 0.0)
            fn_count = cm.get("fn", 0.0)
            valid_question_count = int(fp_count + tn_count)
            rejected_count_from_cm = int(tp_count + fn_count)
            total_from_cm = int(tp_count + fp_count + tn_count + fn_count)
            
            # 验证：混淆矩阵的总数应该等于题目数
            # 这两个统计应该使用相同的统计源：都基于 gt_structure_map
            if type_question_count != total_from_cm:
                logger.error(
                    f"题型 {type_name}: 统计不一致！\n"
                    f"  - 从 gt_structure_map 统计的题目数: {type_question_count}\n"
                    f"  - 从混淆矩阵统计的题目数: {total_from_cm} (TP:{tp_count} + FP:{fp_count} + TN:{tn_count} + FN:{fn_count})\n"
                    f"  - 差值: {type_question_count - total_from_cm} 道题目\n"
                    f"可能的原因：\n"
                    f"  1. 有些文件在 evaluator 中计算混淆矩阵时，题目类型不在 confusion_matrix_by_type 中，被跳过了统计\n"
                    f"  2. 有些文件处理失败，这些文件的题目没有被统计到混淆矩阵中\n"
                    f"  3. 有些题目的类型识别错误，在 evaluator 中被识别为其他类型\n"
                    f"  4. 统计逻辑存在bug，导致某些题目被重复统计或遗漏"
                )
                
                # 详细诊断：检查每个文件的统计情况
                logger.info(f"开始诊断 {type_name} 题型的统计不一致问题...")
                type_total_from_files = 0
                type_total_from_cm = 0
                for ctx in all_results:
                    if ctx.gt_structure_map:
                        # 统计该文件中该题型的题目数
                        file_type_count = sum(1 for q_id, q_type_in_map in ctx.gt_structure_map.items() if q_type_in_map == q_type)
                        type_total_from_files += file_type_count
                        
                        # 统计该文件中该题型的混淆矩阵数
                        cm_by_type = ctx.metrics.get("confusion_matrix_by_type", {})
                        file_cm = cm_by_type.get(type_name, {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0})
                        file_cm_total = file_cm.get("tp", 0.0) + file_cm.get("fp", 0.0) + file_cm.get("tn", 0.0) + file_cm.get("fn", 0.0)
                        type_total_from_cm += file_cm_total
                        
                        if file_type_count != file_cm_total:
                            logger.warning(
                                f"  文件 {ctx.filename}: gt_structure_map中有 {file_type_count} 道{type_name}题目，"
                                f"但混淆矩阵中只有 {file_cm_total} 道"
                            )
                
                logger.info(
                    f"诊断结果: 所有文件的{type_name}题目总数 = {type_total_from_files}, "
                    f"混淆矩阵总数 = {type_total_from_cm}, 差值 = {type_total_from_files - type_total_from_cm}"
                )
            
            # 如果计算不一致，记录警告
            rejected_count = len(type_data.get("rejected_question_ids", []))
            if rejected_count != rejected_count_from_cm:
                logger.warning(
                    f"题型 {type_name}: rejected_question_ids 数量 ({rejected_count}) "
                    f"与混淆矩阵统计的无法识别题目数 ({rejected_count_from_cm}) 不一致。"
                    f"使用混淆矩阵的结果：有效题目数 = {valid_question_count}"
                )
            
            # 计算平均值（只计算GT正常题目的相似度）
            stem_sim_sum = sum(type_data["stem_sims"])
            avg_stem_sim = stem_sim_sum / valid_question_count if valid_question_count > 0 else 0.0
            avg_ans_acc = sum(type_data["ans_accs"]) / len(type_data["ans_accs"]) if type_data["ans_accs"] else (-1.0 if type_name in ["Choice", "Fill"] else -1.0)
            avg_img_sim = sum(type_data["img_sims"]) / len(type_data["img_sims"]) if type_data["img_sims"] else -1.0
            # 答案提取准确率：只有选择题和填空题才有，解答题不适用
            if "answer_extraction_accs" in type_data:
                avg_answer_extraction_acc = sum(type_data["answer_extraction_accs"]) / len(type_data["answer_extraction_accs"]) if type_data["answer_extraction_accs"] else (-1.0 if type_name in ["Choice", "Fill"] else -1.0)
            else:
                avg_answer_extraction_acc = -1.0  # 解答题不适用
            
            summary["by_type"][type_name] = {
                "count": type_question_count,  # 该题型的题目数
                "valid_count": valid_question_count,  # 该题型的有效题目数
                "avg_stem_sim": avg_stem_sim,  # 按题目级别计算的平均相似度（排除GT无法识别的题目）
                "avg_img_sim": avg_img_sim,  # 按文件级别计算的平均图片相似度
                "avg_ans_acc": avg_ans_acc,  # 按题目级别计算的平均答案准确率
                "avg_answer_extraction_acc": avg_answer_extraction_acc,  # 按文件级别计算的平均答案提取准确率
                # 按题型计算的混淆矩阵和分类指标
                "confusion_matrix": confusion_matrix_by_type_sum.get(type_name, {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}),
                "precision": precision_by_type.get(type_name, 0.0),
                "recall": recall_by_type.get(type_name, 0.0),
                "f1": f1_by_type.get(type_name, 0.0),
            }
        
        return summary
    
    def save_reports(self, all_results: List[ProcessContext], failed_files_list=None):
        """
        保存所有报告
        
        :param all_results: 成功处理的文件结果列表
        :param failed_files_list: 失败的文件信息列表
        """
        # 1. 保存每个题目的详细报告
        detailed_reports = []
        for ctx in all_results:
            # 保存切割的题目
            self.save_segments(ctx)
            
            # 生成详细报告
            report = self.generate_single_report(ctx)
            detailed_reports.append(report)
        
        # 保存详细报告 JSON
        detailed_file = self.reports_dir / "detailed_reports.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_reports, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved detailed reports: {detailed_file}")
        
        # 2. 生成并保存汇总报告（传入失败文件列表以便统计总题目数）
        summary = self.generate_summary_report(all_results, failed_files_list)
        summary_file = self.reports_dir / "summary_report.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved summary report: {summary_file}")
        
        # 3. 生成文本格式的汇总报告
        self._save_text_summary(summary, all_results)
        
        # 4. 生成失败文件报告
        if failed_files_list:
            self._save_failed_files_report(failed_files_list, summary)
    
    def _save_text_summary(self, summary: dict, all_results: List[ProcessContext]):
        """
        生成文本格式的汇总报告
        """
        text_report = []
        text_report.append("=" * 80)
        text_report.append("评估报告汇总")
        text_report.append("=" * 80)
        
        # 显示总题目数
        total_count = summary['total_count']
        total_from_gt = summary.get('total_count_from_gt_files', total_count)
        rejected_count = summary['rejected_count']
        valid_count = summary['valid_count']
        calculated_total = rejected_count + valid_count
        
        # 验证：混淆矩阵总数应该等于总题目数
        text_report.append(f"\n总题目数: {total_count} (基于混淆矩阵统计，确保一致性：TP+FP+TN+FN)")
        if total_count != total_from_gt:
            text_report.append(f"  - 从GT文件统计的题目数: {total_from_gt} (仅供参考)")
            text_report.append(f"  - 说明: 两者不一致可能是因为有些文件处理失败或第二轮匹配后统计更新")
        else:
            text_report.append(f"  - 从GT文件统计的题目数: {total_from_gt} (与总题目数一致)")
        
        if 'processed_count' in summary:
            text_report.append(f"成功处理的文件中的题目数: {summary['processed_count']}")
        
        text_report.append(f"无法识别题目数: {rejected_count}")
        text_report.append(f"有效题目数: {valid_count}")
        
        # 验证：混淆矩阵统计的题目数 = 无法识别题目数 + 有效题目数 = 总题目数
        if calculated_total != (rejected_count + valid_count):
            text_report.append(f"\n⚠️ 警告: 混淆矩阵统计的题目数 ({calculated_total}) ≠ 无法识别题目数 ({rejected_count}) + 有效题目数 ({valid_count}) = {rejected_count + valid_count}")
            text_report.append(f"   这可能表示统计逻辑存在问题，请检查日志中的警告信息")
        else:
            text_report.append(f"\n✓ 验证通过: 总题目数 ({total_count}) = 无法识别题目数 ({rejected_count}) + 有效题目数 ({valid_count})")
        
        # 说明：混淆矩阵现在包含了所有文件（包括处理失败的文件）
        if total_count == calculated_total:
            text_report.append(f"\n注意: 混淆矩阵统计包含了所有GT文件中的题目（包括处理失败的文件），确保总数一致")
        
        # 如果有未分类的题目，添加说明
        unclassified_count = summary.get('unclassified_count', 0)
        if unclassified_count > 0:
            text_report.append(f"未分类题目数: {unclassified_count} (不在选择题/填空题/解答题中，已从按题型统计中排除)")
            # 计算各题型题目数之和
            type_counts_sum = sum(stats.get('count', 0) for stats in summary.get('by_type', {}).values())
            text_report.append(f"说明: 总题目数 ({summary['total_count']}) = 各题型题目数之和 ({type_counts_sum}) + 未分类题目数 ({unclassified_count})")
        
        # 提示查看失败文件报告
        text_report.append(f"\n注意: 如果有文件处理失败，请查看 failed_files_report.txt 了解详情")
        
        # 添加对齐统计信息
        if "alignment_stats" in summary:
            alignment_stats = summary["alignment_stats"]
            text_report.append("\n" + "=" * 80)
            text_report.append("题目对齐统计")
            text_report.append("=" * 80)
            total_questions_aligned = alignment_stats.get('total_questions', 0)
            total_aligned = alignment_stats.get('total_aligned', 0)
            failed_count = alignment_stats.get('failed_alignment_count', 0)
            
            text_report.append(f"总对齐题目数: {total_questions_aligned}")
            text_report.append(f"成功对齐题目数: {total_aligned}")
            text_report.append(f"  - 直接匹配成功: {alignment_stats.get('exact_match_count', 0)} 道题目")
            text_report.append(f"  - 模糊匹配成功: {alignment_stats.get('fuzzy_match_count', 0)} 道题目")
            text_report.append(f"  - Fallback匹配（使用下一个题目开头）: {alignment_stats.get('fallback_next_question_count', 0)} 道题目")
            text_report.append(f"  - Fallback匹配（使用文本末尾）: {alignment_stats.get('fallback_text_end_count', 0)} 道题目")
            
            # 验证对齐统计的一致性
            if total_aligned + failed_count != total_questions_aligned:
                text_report.append(f"\n⚠️ 警告: 对齐统计不一致: 成功对齐({total_aligned}) + 失败({failed_count}) = {total_aligned + failed_count} ≠ 总对齐题目数({total_questions_aligned})")
            
            # 如果总对齐题目数小于总题目数，说明有题目没有被LLM识别到
            total_from_gt = summary.get('total_count_from_gt_files', summary.get('total_count', 0))
            if total_questions_aligned < total_from_gt:
                missing_count = total_from_gt - total_questions_aligned
                text_report.append(f"\n说明: 总对齐题目数({total_questions_aligned}) < 总题目数({total_from_gt})，")
                text_report.append(f"  有 {missing_count} 道题目没有被LLM识别到，这些题目在混淆矩阵中被统计为：")
                text_report.append(f"  - 如果GT无法识别：TP（GT无法识别，Pred也拒答）")
                text_report.append(f"  - 如果GT正常：FP（GT正常，但Pred拒答或未匹配）")
        
        # 整体平均值统计（排除GT无法识别的题目）
        if "overall" in summary:
            overall = summary["overall"]
            text_report.append("\n" + "=" * 80)
            text_report.append("整体平均值（排除GT无法识别的题目）")
            text_report.append("=" * 80)
            text_report.append(f"整体文本相似度: {overall['avg_stem_sim']:.4f}")
            if overall['avg_img_sim'] != -1.0:
                text_report.append(f"整体图片相似度: {overall['avg_img_sim']:.4f}")
            else:
                text_report.append(f"整体图片相似度: 不适用（所有文件GT无图）")
            # if overall['avg_ans_acc'] != -1.0:
            #     text_report.append(f"整体答案准确率: {overall['avg_ans_acc']:.4f}")
            # if overall['avg_answer_extraction_acc'] != -1.0:
            #     text_report.append(f"整体答案提取准确率: {overall['avg_answer_extraction_acc']:.4f}")
        
        # 混淆矩阵统计
        if "confusion_matrix" in summary:
            cm = summary["confusion_matrix"]
            text_report.append("\n" + "=" * 80)
            text_report.append("混淆矩阵统计")
            text_report.append("=" * 80)
            text_report.append(f"TP (True Positive): {cm['tp']:.0f} - GT无法识别，Pred也拒答")
            text_report.append(f"FP (False Positive): {cm['fp']:.0f} - GT正常，但Pred拒答或未匹配")
            text_report.append(f"TN (True Negative): {cm['tn']:.0f} - GT正常，Pred也正常")
            text_report.append(f"FN (False Negative): {cm['fn']:.0f} - GT无法识别，但Pred正常")
            
            # 添加Precision, Recall, F1
            if "precision" in summary:
                text_report.append("")
                text_report.append("分类指标（针对所有题型的整体评估）")
                text_report.append("=" * 80)
                text_report.append(f"Precision: {summary['precision']:.4f} - 在所有Pred拒答的题目中，有多少是GT确实无法识别的")
                text_report.append(f"Recall: {summary['recall']:.4f} - 在所有GT无法识别的题目中，有多少被Pred正确拒答")
                text_report.append(f"F1 Score: {summary['f1']:.4f} - Precision和Recall的调和平均")
        
        text_report.append("\n" + "=" * 80)
        text_report.append("按题型统计（相似度仅统计GT正常的题目）")
        text_report.append("=" * 80)
        
        for type_name, stats in summary['by_type'].items():
            text_report.append(f"\n【{type_name}】")
            text_report.append(f"  题目数: {stats['count']}")
            text_report.append(f"  有效题目数: {stats['valid_count']}")
            text_report.append(f"  平均题目主干相似度: {stats['avg_stem_sim']:.4f}")
            if stats['avg_img_sim'] != -1.0:
                text_report.append(f"  平均图片相似度: {stats['avg_img_sim']:.4f}")
            else:
                text_report.append(f"  平均图片相似度: 不适用（GT无图）")
            # if stats['avg_ans_acc'] != -1.0:
            #     text_report.append(f"  平均答案准确率: {stats['avg_ans_acc']:.4f}")
            # if stats['avg_answer_extraction_acc'] != -1.0:
            #     text_report.append(f"  平均答案提取准确率: {stats['avg_answer_extraction_acc']:.4f}")
            
            # 添加按题型的混淆矩阵和分类指标
            if "confusion_matrix" in stats:
                cm = stats["confusion_matrix"]
                text_report.append("")
                text_report.append(f"  混淆矩阵（{type_name}）:")
                text_report.append(f"    TP: {cm['tp']:.0f} - GT无法识别，Pred也拒答")
                text_report.append(f"    FP: {cm['fp']:.0f} - GT正常，但Pred拒答或未匹配")
                text_report.append(f"    TN: {cm['tn']:.0f} - GT正常，Pred也正常")
                text_report.append(f"    FN: {cm['fn']:.0f} - GT无法识别，但Pred正常")
            
            if "precision" in stats:
                text_report.append("")
                text_report.append(f"  分类指标（{type_name}）:")
                text_report.append(f"    Precision: {stats['precision']:.4f}")
                text_report.append(f"    Recall: {stats['recall']:.4f}")
                text_report.append(f"    F1 Score: {stats['f1']:.4f}")
        
        text_report.append("\n" + "=" * 80)
        text_report.append("详细结果")
        text_report.append("=" * 80)
        
        for ctx in all_results:
            text_report.append(f"\n文件名: {ctx.filename}")
            text_report.append(f"  题型: {ctx.gt_type.value if ctx.gt_type else 'Unknown'}")
            # 添加对齐统计信息
            if ctx.alignment_stats:
                alignment_stats = ctx.alignment_stats
                text_report.append(f"  对齐统计: 总题目数={alignment_stats.get('total_questions', 0)}, "
                                 f"成功对齐={alignment_stats.get('total_aligned', 0)}, "
                                 f"直接匹配={alignment_stats.get('exact_match_count', 0)}, "
                                 f"模糊匹配={alignment_stats.get('fuzzy_match_count', 0)}, "
                                 f"Fallback(下一题)={alignment_stats.get('fallback_next_question_count', 0)}, "
                                 f"Fallback(文本末尾)={alignment_stats.get('fallback_text_end_count', 0)}")
            text_report.append(f"  题目主干相似度: {ctx.metrics.get('stem_sim', 0):.4f}")
            img_sim = ctx.metrics.get('img_sim', 0)
            if img_sim != -1.0:
                text_report.append(f"  图片相似度: {img_sim:.4f}")
            else:
                text_report.append(f"  图片相似度: 不适用（GT无图）")
            # ans_acc = ctx.metrics.get('ans_acc', -1)
            # if ans_acc != -1.0:
            #     text_report.append(f"  答案准确率: {ans_acc:.4f}")
            # answer_extraction_acc = ctx.metrics.get('answer_extraction_acc', -1)
            # if answer_extraction_acc != -1.0:
            #     text_report.append(f"  答案提取准确率: {answer_extraction_acc:.4f}")
        
        # 保存文本报告
        text_file = self.reports_dir / "summary_report.txt"
        text_file.write_text('\n'.join(text_report), encoding='utf-8')
        logger.info(f"Saved text summary report: {text_file}")
    
    def _save_failed_files_report(self, failed_files_list: List[dict], summary: dict):
        """
        生成并保存失败文件报告
        
        :param failed_files_list: 失败的文件信息列表
        :param summary: 汇总报告数据
        """
        if not failed_files_list:
            return
        
        # 统计失败类型
        failure_types = {}
        for failure in failed_files_list:
            failure_type = failure.get("failure_type", "unknown")
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
        
        # 生成JSON报告
        failed_report_json = {
            "total_failed_files": len(failed_files_list),
            "failure_type_summary": failure_types,
            "failed_files": failed_files_list,
            "summary": {
                "total_processed_files": summary.get("total_count", 0) + len(failed_files_list),
                "successful_files": summary.get("total_count", 0),
                "failed_files": len(failed_files_list)
            }
        }
        
        json_file = self.reports_dir / "failed_files_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(failed_report_json, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved failed files report (JSON): {json_file}")
        
        # 生成文本报告
        text_report = []
        text_report.append("=" * 80)
        text_report.append("失败文件报告")
        text_report.append("=" * 80)
        text_report.append(f"\n总失败文件数: {len(failed_files_list)}")
        text_report.append(f"\n失败类型统计:")
        for failure_type, count in sorted(failure_types.items()):
            type_name_map = {
                "alignment_failed": "对齐失败",
                "exception": "处理异常",
                "unknown": "未知原因"
            }
            text_report.append(f"  {type_name_map.get(failure_type, failure_type)}: {count} 个文件")
        
        text_report.append("\n" + "=" * 80)
        text_report.append("失败文件详情")
        text_report.append("=" * 80)
        
        # 按失败类型分组
        by_type = {}
        for failure in failed_files_list:
            failure_type = failure.get("failure_type", "unknown")
            if failure_type not in by_type:
                by_type[failure_type] = []
            by_type[failure_type].append(failure)
        
        for failure_type, failures in sorted(by_type.items()):
            type_name_map = {
                "alignment_failed": "对齐失败",
                "exception": "处理异常",
                "unknown": "未知原因"
            }
            text_report.append(f"\n【{type_name_map.get(failure_type, failure_type)}】 ({len(failures)} 个文件)")
            
            for i, failure in enumerate(failures, 1):
                text_report.append(f"\n{i}. {failure['filename']}")
                text_report.append(f"   失败原因: {failure.get('reason', '未知')}")
                
                if failure_type == "alignment_failed":
                    text_report.append(f"   gt_structure_map 存在: {failure.get('gt_structure_map_exists', False)}")
                    text_report.append(f"   gt_structure_map 大小: {failure.get('gt_structure_map_size', 0)}")
                
                if failure_type == "exception":
                    text_report.append(f"   异常类型: {failure.get('exception_type', 'Unknown')}")
                
                if 'gt_file_path' in failure:
                    text_report.append(f"   GT文件路径: {failure['gt_file_path']}")
                if 'pred_file_path' in failure:
                    text_report.append(f"   Pred文件路径: {failure['pred_file_path']}")
        
        # 添加统计信息
        text_report.append("\n" + "=" * 80)
        text_report.append("统计信息")
        text_report.append("=" * 80)
        total_processed = summary.get('total_count', 0) + len(failed_files_list)
        text_report.append(f"总处理文件数: {total_processed}")
        text_report.append(f"成功处理文件数: {summary.get('total_count', 0)}")
        text_report.append(f"失败文件数: {len(failed_files_list)}")
        if total_processed > 0:
            success_rate = (summary.get('total_count', 0) / total_processed * 100)
            text_report.append(f"成功率: {success_rate:.2f}%")
        
        # 保存文本报告
        text_file = self.reports_dir / "failed_files_report.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_report))
        logger.info(f"Saved failed files report (TXT): {text_file}")