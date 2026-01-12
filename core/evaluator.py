# 指标计算
# core/evaluator.py

import logging
import re
from core.schema import ProcessContext, QuestionType
from utils.llm_client import LLMClient
from utils.image_sim import ImageComparator
from utils.text_metric import TextMetricsCalculator

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, vision_client: LLMClient):
        """
        初始化评估器
        :param vision_client: 用于图片对比的视觉模型客户端
        """
        # 初始化所有需要的工具
        self.image_comparator = ImageComparator(vision_client=vision_client)
        self.text_metric_calc = TextMetricsCalculator() # 文本相似度计算器

    def _is_gt_question_rejected(self, question_id: str, gt_clean: str, gt_start_snippet: str, gt_end_snippet: str) -> bool:
        """
        判断GT中的某个题目是否被拒答（无法识别）
        
        :param question_id: 题目编号
        :param gt_clean: GT清理后的文本
        :param gt_start_snippet: GT题目开头片段
        :param gt_end_snippet: GT题目结尾片段
        :return: 是否被拒答
        """
        # 注意：不能直接检查 gt_start_snippet 或 gt_end_snippet 是否包含"无法识别"标记
        # 因为 snippet 可能包含了下一题的内容，而下一题可能有"无法识别"标记，会误判当前题目
        # 必须精确定位当前题目的内容范围，然后在该范围内检查拒答标记
        rejection_keywords = ["[无法识别]", "无法识别", "无法辨认", "无法读取"]
        
        # 从GT文本中精确查找当前题目的内容
        start_idx = gt_clean.find(gt_start_snippet)
        if start_idx == -1:
            # 如果找不到 gt_start_snippet，尝试使用题号来查找
            question_num_pattern = re.compile(rf'{re.escape(question_id)}\s*[\.。、]')
            match = question_num_pattern.search(gt_clean)
            if match:
                question_num_pos = match.start()
                after_question_num = gt_clean[question_num_pos:]
                if "[无法识别" in after_question_num or "无法识别" in after_question_num:
                    start_idx = question_num_pos
                else:
                    return False
            else:
                return False
        
        # 找到结尾（下一个题目的开头或文本末尾）
        end_idx = gt_clean.find(gt_end_snippet, start_idx)
        if end_idx == -1:
            # 尝试找到下一个题目的开头
            next_q_id = str(int(question_id) + 1)
            next_patterns = [
                f"\n{next_q_id}.",
                f"\n{next_q_id}、",
                f"{next_q_id}.",
                f"{next_q_id}、",
            ]
            next_idx = -1
            for pattern in next_patterns:
                next_idx = gt_clean.find(pattern, start_idx)
                if next_idx != -1:
                    break
            if next_idx != -1:
                end_idx = next_idx
            else:
                end_idx = len(gt_clean)
        else:
            # 如果找到了 gt_end_snippet，检查它是否包含了下一题的内容
            # 对于"无法识别"的题目，gt_end_snippet 可能错误地包含了下一题的内容
            # 需要截断到下一个题目的开头之前
            potential_end = end_idx + len(gt_end_snippet)
            
            # 检查 potential_end 之后是否包含下一个题目的开头
            next_q_id = str(int(question_id) + 1)
            next_patterns = [
                f"\n{next_q_id}.",
                f"\n{next_q_id}、",
                f"{next_q_id}.",
                f"{next_q_id}、",
            ]
            next_idx = -1
            for pattern in next_patterns:
                next_idx = gt_clean.find(pattern, start_idx)
                if next_idx != -1 and next_idx < potential_end:
                    # 找到了下一个题目，且它在 potential_end 之前
                    # 说明 gt_end_snippet 包含了下一题的内容，需要截断
                    potential_end = next_idx
                    break
            
            # 如果 potential_end 之后还有下一个题目，使用下一个题目的位置
            if next_idx == -1 or next_idx >= potential_end:
                # 如果还没找到下一个题目，再尝试在 potential_end 之后查找
                for pattern in next_patterns:
                    next_idx = gt_clean.find(pattern, potential_end)
                    if next_idx != -1:
                        # 如果找到了，说明 gt_end_snippet 确实包含了下一题的内容
                        # 但我们已经使用了 potential_end，所以不需要再调整
                        # 不过，为了更准确，我们应该使用 next_idx
                        if next_idx < len(gt_clean):
                            potential_end = next_idx
                        break
            
            end_idx = potential_end
        
        # 提取题目内容
        question_content = gt_clean[start_idx:end_idx]
        
        # 检查是否包含拒答标记
        rejection_keywords = ["[无法识别]", "无法识别", "无法辨认", "无法读取"]
        return any(keyword in question_content for keyword in rejection_keywords)

    def evaluate(self, ctx: ProcessContext) -> ProcessContext:
        """
        对单个 ProcessContext 对象进行全面评估并填充 metrics 字段
        评估所有题目，排除GT中被拒答的题目
        """
        # --- 1. 整体拒答状态判断（用于向后兼容）---
        is_gt_rejected = "[无法识别]" in ctx.gt_raw
        is_pred_invalid = ctx.pred_is_rejected or not ctx.alignment_found
        
        # 如果没有题目对齐结果，但仍然需要统计混淆矩阵（基于gt_structure_map）
        # 注意：即使all_question_segments为空，gt_structure_map中可能仍有题目需要统计
        # 这些题目应该被统计到混淆矩阵中（作为未匹配的题目）
        if not ctx.all_question_segments:
            # 如果gt_structure_map存在，仍然需要遍历其中的题目来统计混淆矩阵
            # 这种情况下，所有题目都视为未匹配（alignment_found = False）
            if ctx.gt_structure_map:
                # 继续执行下面的逻辑，遍历gt_structure_map统计混淆矩阵
                # 但跳过相似度计算（因为没有对齐结果）
                pass
            else:
                # 如果没有gt_structure_map也没有all_question_segments，使用旧的逻辑
                if is_gt_rejected:
                    ctx.metrics = {
                        "stem_sim": 0.0,
                        "img_sim": 0.0,
                        "ans_acc": -1.0,
                        "answer_extraction_acc": -1.0,
                        "gt_rejected_flag": 1.0,
                        "pred_rejected_flag": 1.0 if is_pred_invalid else 0.0,
                        "tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0,
                    }
                    return ctx
                
                if is_pred_invalid:
                    ctx.metrics = {
                    "stem_sim": 0.0,
                    "img_sim": 0.0,
                    "ans_acc": 0.0,
                        "answer_extraction_acc": 0.0,
                        "gt_rejected_flag": 0.0,
                        "pred_rejected_flag": 1.0,
                        "tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0,
                }
                    return ctx

        # --- 2. 对所有题目进行评估 ---
        # 使用无图文本进行相似度计算
        gt_text_for_sim = ctx.gt_clean_for_alignment if ctx.gt_clean_for_alignment else ctx.gt_clean
        
        stem_sims = []  # 所有题目的相似度（只包括有效题目，即GT正常的题目）
        ans_accs = []  # 所有题目的答案准确率（只包括有效题目中的选择题和填空题）
        valid_question_count = 0  # 有效题目数（GT正常的题目数，用于计算相似度分母）
        valid_mcq_fib_count = 0  # 有效题目数中的选择题和填空题数（用于计算答案准确率分母）
        # 按题型分组的混淆矩阵
        confusion_matrix_by_type = {
            QuestionType.MCQ: {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            QuestionType.FIB: {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            QuestionType.SUB: {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        }
        # 整体混淆矩阵（用于向后兼容）
        tp_count = 0  # True Positive: GT无法识别，Pred也拒答
        fp_count = 0  # False Positive: GT正常，但Pred拒答或未匹配
        tn_count = 0  # True Negative: GT正常，Pred也正常
        fn_count = 0  # False Negative: GT无法识别，但Pred正常
        
        # 遍历所有在gt_structure_map中的题目，确保混淆矩阵统计的范围与题目数统计的范围一致
        if ctx.gt_structure_map:
            # 遍历gt_structure_map中的所有题目
            for question_id, q_type in ctx.gt_structure_map.items():
                # 在all_question_segments中查找该题目
                question_data = None
                if ctx.all_question_segments:
                    for q in ctx.all_question_segments:
                        if q.get("question_id") == question_id:
                            question_data = q
                            break
                
                # 如果找到了对齐结果，使用对齐结果的信息
                if question_data:
                    alignment_found = question_data.get("alignment_found", False)
                    pred_is_rejected = question_data.get("is_rejected", False)
                    pred_segment = question_data.get("pred_segment")
                    gt_start_snippet = question_data.get("gt_start_snippet", "")
                    gt_end_snippet = question_data.get("gt_end_snippet", "")
                else:
                    # 如果没有找到对齐结果，说明对齐失败
                    # 对齐失败算作拒答（因为aligner会返回is_rejected=True）
                    alignment_found = False
                    pred_is_rejected = True
                    pred_segment = None
                    gt_start_snippet = ""
                    gt_end_snippet = ""
                
                # 判断GT中该题目是否被拒答
                is_gt_q_rejected = self._is_gt_question_rejected(
                    question_id, ctx.gt_clean, gt_start_snippet, gt_end_snippet
                )
                
                # 检查题目类型是否在混淆矩阵中（只统计MCQ、FIB、SUB三种类型）
                if q_type not in confusion_matrix_by_type:
                    # 如果题目类型不在预定义的类型中，记录警告但继续统计整体混淆矩阵
                    logger.warning(f"题目 {question_id} 的类型 {q_type} 不在预定义的类型中（MCQ/FIB/SUB），跳过按题型统计，但仍计入整体统计")
                    # 只统计整体混淆矩阵，不统计按题型
                    if is_gt_q_rejected:
                        # GT无法识别
                        if pred_is_rejected or not alignment_found:
                            tp_count += 1  # 整体统计
                            logger.debug(f"题目 {question_id}: TP (GT无法识别，Pred也拒答) [未分类类型]")
                        else:
                            fn_count += 1  # 整体统计
                            logger.debug(f"题目 {question_id}: FN (GT无法识别，但Pred正常) [未分类类型]")
                    else:
                        # GT正常
                        if pred_is_rejected or not alignment_found:
                            fp_count += 1  # 整体统计
                            logger.debug(f"题目 {question_id}: FP (GT正常，但Pred拒答或未匹配) [未分类类型]")
                        else:
                            tn_count += 1  # 整体统计
                            logger.debug(f"题目 {question_id}: TN (GT正常，Pred也正常) [未分类类型]")
                    # 跳过按题型统计和相似度计算
                    continue
                
                # 计算混淆矩阵（按题型分组）
                if is_gt_q_rejected:
                    # GT无法识别
                    if pred_is_rejected or not alignment_found:
                        tp_count += 1  # 整体统计
                        confusion_matrix_by_type[q_type]["tp"] += 1  # 按题型统计
                        logger.debug(f"题目 {question_id}: TP (GT无法识别，Pred也拒答)")
                    else:
                        fn_count += 1  # 整体统计
                        confusion_matrix_by_type[q_type]["fn"] += 1  # 按题型统计
                        logger.debug(f"题目 {question_id}: FN (GT无法识别，但Pred正常)")
                else:
                    # GT正常
                    if pred_is_rejected or not alignment_found:
                        fp_count += 1  # 整体统计
                        confusion_matrix_by_type[q_type]["fp"] += 1  # 按题型统计
                        logger.debug(f"题目 {question_id}: FP (GT正常，但Pred拒答或未匹配)")
                    else:
                        tn_count += 1  # 整体统计
                        confusion_matrix_by_type[q_type]["tn"] += 1  # 按题型统计
                        logger.debug(f"题目 {question_id}: TN (GT正常，Pred也正常)")
                
                # 如果GT被拒答（[无法识别]），跳过相似度计算（不计入分母）
                if is_gt_q_rejected:
                    continue
                
                # GT正常的题目计入有效题目数（分母）
                valid_question_count += 1
                
                # 计算答案准确率（只针对选择题和填空题）
                # 注意：答案准确率的计算应该在continue之前，确保所有有效题目都计入分母
                if q_type in [QuestionType.MCQ, QuestionType.FIB]:
                    # 有效题目数中的选择题和填空题计入答案准确率分母
                    valid_mcq_fib_count += 1
                    
                    # 答案优先级：aligner (LLM) > raw_handwritten_answer > preprocessor (正则)
                    # 优先使用从对齐结果中提取的答案（LLM 可以识别跨行答案，更准确）
                    pred_ans = str(question_data.get("pred_answer", "")).strip().upper() if question_data else ""
                    if not pred_ans and question_data:
                        # 如果没有 pred_answer（来自 aligner），使用 raw_handwritten_answer
                        pred_ans = str(question_data.get("raw_handwritten_answer", "")).strip().upper()
                    if not pred_ans:
                        # 如果对齐环节（LLM）没有提取到答案，使用 preprocessor 的答案作为后备
                        pred_ans = str(ctx.pred_extracted_answers.get(question_id, "")).strip().upper()
                    
                    # 优先使用从对齐结果中提取的GT答案（已经去除了题干中的答案）
                    gt_ans = str(question_data.get("gt_answer", "")).strip().upper() if question_data else ""
                    if not gt_ans:
                        # 如果没有gt_answer，使用ctx.gt_extracted_answers
                        gt_ans = str(ctx.gt_extracted_answers.get(question_id, "")).strip().upper()
                    
                    if gt_ans:  # 如果有GT答案
                        # 如果Pred未对齐或被拒答，答案准确率为0.0
                        if not alignment_found or not pred_segment:
                            q_ans_acc = 0.0
                        else:
                            # 标准化答案：去除所有分隔符（逗号、空格、分号、顿号等），然后比较
                            # 这样可以处理 "A,C" vs "AC" 或 "A C" vs "AC" 等情况
                            gt_ans_normalized = re.sub(r'[,\s;，、；\s]+', '', gt_ans)
                            pred_ans_normalized = re.sub(r'[,\s;，、；\s]+', '', pred_ans) if pred_ans else ""
                            q_ans_acc = 1.0 if (pred_ans_normalized and pred_ans_normalized == gt_ans_normalized) else 0.0
                        ans_accs.append(q_ans_acc)
                        # 保存每个题目的答案准确率到 question_data 字典中（用于按题型分组）
                        if question_data:
                            question_data["ans_acc"] = q_ans_acc
                    else:
                        # 没有GT答案，不适用（不计入分母）
                        valid_mcq_fib_count -= 1  # 回退，因为不计入分母
                        if question_data:
                            question_data["ans_acc"] = -1.0  # 没有GT答案，不适用
                
                # 如果Pred未对齐或被拒答，相似度设为0.0（但仍计入分母）
                if not alignment_found or not pred_segment:
                    q_stem_sim = 0.0
                    stem_sims.append(q_stem_sim)
                    # 保存每个题目的相似度到 question_data 字典中（用于按题型分组）
                    if question_data:
                        question_data["stem_sim"] = q_stem_sim
                    continue
                
                # 获取题目类型（已经在上面获取过了）
                
                # 计算题目主干相似度
                # 从GT文本中提取该题目的内容（使用锚点）
                # 注意：gt_start_snippet可能不包含Markdown标记（**），但gt_text_for_sim可能包含
                # 所以先尝试完全匹配，如果失败，尝试移除Markdown标记后匹配
                # 另外，对于"无法识别"的题目，gt_start_snippet可能错误地包含了下一题的内容
                # 所以需要先尝试找到题号，然后截断到题号之后的内容
                gt_question_start = gt_text_for_sim.find(gt_start_snippet)
                
                # 如果完全匹配失败，尝试移除Markdown标记（**）后匹配
                if gt_question_start == -1:
                    # gt_text_for_sim可能包含**标记，但gt_start_snippet没有
                    # 尝试在gt_text_for_sim中移除**后查找
                    gt_text_normalized = re.sub(r'\*\*', '', gt_text_for_sim)
                    gt_question_start = gt_text_normalized.find(gt_start_snippet)
                    if gt_question_start != -1:
                        logger.debug(f"Question {question_id}: Found GT start after removing Markdown markers")
                    else:
                        # 如果还是找不到，尝试使用题号进行部分匹配
                        # 对于"无法识别"的题目，gt_start_snippet可能包含下一题的内容
                        # 所以先尝试找到题号，然后检查题号之后的内容是否匹配
                        question_num_pattern = re.compile(rf'{re.escape(question_id)}\s*[\.。、]')
                        match = question_num_pattern.search(gt_text_normalized)
                        if match:
                            # 找到题号位置
                            question_num_pos = match.start()
                            # 检查题号之后的内容是否与 gt_start_snippet 的开头部分匹配
                            # 对于"无法识别"的题目，gt_start_snippet 可能是 "18. [无法识别的解答题设函数..."
                            # 但实际 GT 中只有 "18. [无法识别的解答题]"
                            # 所以我们需要检查题号之后是否包含"无法识别"标记
                            after_question_num = gt_text_normalized[question_num_pos:]
                            if "[无法识别" in after_question_num or "无法识别" in after_question_num:
                                gt_question_start = question_num_pos
                                logger.debug(f"Question {question_id}: Using partial match for GT start (found question number with rejection marker)")
                            else:
                                # 如果题号之后没有"无法识别"标记，尝试匹配 gt_start_snippet 的开头部分
                                # gt_start_snippet 的开头应该是题号，后面可能跟着一些内容
                                snippet_start = gt_start_snippet[:min(50, len(gt_start_snippet))]  # 取前50个字符
                                if snippet_start in after_question_num:
                                    gt_question_start = question_num_pos
                                    logger.debug(f"Question {question_id}: Using partial match for GT start (found question number with matching prefix)")
                                else:
                                    gt_question_start = question_num_pos
                                    logger.debug(f"Question {question_id}: Using question number position as GT start (fallback)")
                        else:
                            # GT锚点未找到，记录警告并跳过相似度计算
                            logger.warning(
                                f"Question {question_id}: GT start snippet not found in gt_text_for_sim. "
                                f"GT snippet: '{gt_start_snippet[:100]}...' "
                                f"GT text length: {len(gt_text_for_sim)}, "
                                f"First 200 chars of GT text: '{gt_text_for_sim[:200]}...'"
                            )
                            continue
                
                if gt_question_start != -1:
                    # 找到GT题目的结尾
                    gt_question_end = gt_text_for_sim.find(gt_end_snippet, gt_question_start)
                    if gt_question_end == -1:
                        # 尝试找下一个题目
                        next_q_id = str(int(question_id) + 1)
                        next_pattern = f"{next_q_id}."
                        next_idx = gt_text_for_sim.find(next_pattern, gt_question_start)
                        if next_idx != -1:
                            gt_question_end = next_idx
                        else:
                            gt_question_end = len(gt_text_for_sim)
                    else:
                        # 如果找到了 gt_end_snippet，检查它是否包含了下一题的内容
                        # 对于"无法识别"的题目，gt_end_snippet 可能错误地包含了下一题的内容
                        # 需要截断到下一个题目的开头之前
                        potential_end = gt_question_end + len(gt_end_snippet)
                        
                        # 检查 potential_end 之后是否包含下一个题目的开头
                        next_q_id = str(int(question_id) + 1)
                        next_patterns = [
                            f"\n{next_q_id}.",
                            f"\n{next_q_id}、",
                            f"{next_q_id}.",
                            f"{next_q_id}、",
                        ]
                        next_idx = -1
                        for pattern in next_patterns:
                            next_idx = gt_text_for_sim.find(pattern, gt_question_start)
                            if next_idx != -1 and next_idx < potential_end:
                                # 找到了下一个题目，且它在 potential_end 之前
                                # 说明 gt_end_snippet 包含了下一题的内容，需要截断
                                potential_end = next_idx
                                break
                        
                        # 如果 potential_end 之后还有下一个题目，使用下一个题目的位置
                        if next_idx == -1:
                            # 如果还没找到下一个题目，再尝试在 potential_end 之后查找
                            for pattern in next_patterns:
                                next_idx = gt_text_for_sim.find(pattern, potential_end)
                                if next_idx != -1:
                                    # 如果找到了，说明 gt_end_snippet 确实包含了下一题的内容
                                    # 但我们已经使用了 potential_end，所以不需要再调整
                                    # 不过，为了更准确，我们应该使用 next_idx
                                    if next_idx < len(gt_text_for_sim):
                                        potential_end = next_idx
                                    break
                        
                        gt_question_end = potential_end
                    
                    gt_question_content = gt_text_for_sim[gt_question_start:gt_question_end]
                    
                    # 先移除题目类型标记和图片提示（这些不应该影响相似度）
                    # 移除题目类型标记：如"（多选）"、"（单选）"等
                    gt_question_content = re.sub(r'[（(]\s*[多单]选\s*[）)]', '', gt_question_content)
                    # 移除图片提示：如"如图"、"如图所示"等
                    gt_question_content = re.sub(r'如图[，,。.]?', '', gt_question_content)
                    gt_question_content = re.sub(r'如图所示[，,。.]?', '', gt_question_content)
                    
                    # 移除填空题的下划线（无论是否包含内容）
                    # 下划线只是占位符，不应该参与相似度计算
                    # 匹配模式：_+（一个或多个连续的下划线）
                    gt_question_content = re.sub(r'_+', '', gt_question_content)
                    
                    # 从GT题目内容中移除答案（无论gt_answer是否存在，都要移除可能的答案）
                    # 因为GT内容中可能包含答案标签、填空题答案、选择题答案等
                    gt_answer = question_data.get("gt_answer", "") if question_data else ""
                    gt_cleaned = gt_question_content
                    
                    # 方法1: 移除答案标签（【答案：X】等）
                    answer_label_pattern = re.compile(r'[【\[(]\s*答案\s*[:：]\s*[^】\])]+[】\])]')
                    gt_cleaned = answer_label_pattern.sub('', gt_cleaned)
                    
                    # 方法2: 移除填空题答案（下划线包围的内容，如 ___3_____）
                    # 注意：空的下划线已经在上面移除了，这里只移除有内容的下划线
                    fill_pattern = re.compile(r'_+[^_\n]+?_+')
                    gt_cleaned = fill_pattern.sub('', gt_cleaned)
                    
                    # 方法3: 如果有gt_answer，移除选择题答案（选项组合）
                    # 注意：选项列表应该参与相似度计算，只移除答案本身
                    if gt_answer and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in gt_answer):
                        # 先尝试移除括号中的答案，如 "(A)" 或 "(AB)"
                        paren_pattern = re.compile(r'\(\s*' + re.escape(gt_answer) + r'\s*\)')
                        gt_cleaned = paren_pattern.sub('', gt_cleaned)
                        # 移除文本中任何位置的答案（确保是完整的单词，不匹配部分单词）
                        # 匹配"则 BCD"、"则BCD"、" BCD"等，但确保后面是空格、标点或行尾
                        choice_pattern = re.compile(r'(\s+|^)' + re.escape(gt_answer) + r'(?=\s|$|[。.，,；;：:])')
                        gt_cleaned = choice_pattern.sub(r'\1', gt_cleaned)
                        # 清理多余的空格
                        gt_cleaned = re.sub(r'\s+', ' ', gt_cleaned)
                        # 清理末尾可能残留的 "为" 或 "=" 等字（如果答案被移除后留下）
                        gt_cleaned = re.sub(r'\s*[=为]\s*$', '', gt_cleaned)
                    
                    # 清理多余的空格和空行
                    gt_cleaned = re.sub(r' {2,}', ' ', gt_cleaned)
                    gt_cleaned = re.sub(r'\n{3,}', '\n\n', gt_cleaned)
                    gt_cleaned = gt_cleaned.strip()
                    
                    gt_question_content_cleaned = gt_cleaned
                    
                    # pred_segment已经是清理后的版本（aligner已经移除了答案）
                    # 但还需要进行与GT内容一致的清理，确保相似度计算的公平性
                    pred_stem_pure = pred_segment
                    
                    # 移除题目类型标记（与GT内容清理保持一致）
                    pred_stem_pure = re.sub(r'[（(]\s*[多单]选\s*[）)]', '', pred_stem_pure)
                    
                    # 移除图片提示（与GT内容清理保持一致）
                    pred_stem_pure = re.sub(r'如图[，,。.]?', '', pred_stem_pure)
                    pred_stem_pure = re.sub(r'如图所示[，,。.]?', '', pred_stem_pure)
                    
                    # 移除填空题的下划线（无论是否包含内容）
                    # 下划线只是占位符，不应该参与相似度计算
                    pred_stem_pure = re.sub(r'_+', '', pred_stem_pure)
                    
                    # 检查清理后的内容是否为空
                    if not gt_question_content_cleaned.strip() or not pred_stem_pure.strip():
                        logger.warning(
                            f"Question {question_id}: Cleaned content is empty. "
                            f"GT length: {len(gt_question_content_cleaned)}, "
                            f"Pred length: {len(pred_stem_pure)}. "
                            f"GT snippet: '{gt_start_snippet[:50]}...', "
                            f"Pred segment: '{pred_segment[:50] if pred_segment else None}...'"
                        )
                        # 如果清理后的内容为空，相似度为0
                        q_stem_sim = 0.0
                    else:
                        # 计算相似度（选项排列方式不同不影响，因为文本已展平）
                        q_stem_sim = self.text_metric_calc.calculate_similarity(
                            gt_question_content_cleaned,
                            pred_stem_pure
                        )
                        # 如果相似度为0，记录详细信息用于调试
                        if q_stem_sim == 0.0:
                            logger.debug(
                                f"Question {question_id}: Similarity is 0.0. "
                                f"GT cleaned (first 100 chars): '{gt_question_content_cleaned[:100]}...', "
                                f"Pred cleaned (first 100 chars): '{pred_stem_pure[:100]}...'"
                            )
                    stem_sims.append(q_stem_sim)
                    # 保存每个题目的相似度到 question_data 字典中（用于按题型分组）
                    if question_data:
                        question_data["stem_sim"] = q_stem_sim
                
                # 答案准确率已经在上面计算过了（在continue之前），这里不需要重复计算
                # 但需要处理解答题的情况（如果之前没有处理）
                if q_type == QuestionType.SUB:
                    if question_data and "ans_acc" not in question_data:
                        question_data["ans_acc"] = -1.0  # 解答题不适用
        
        # --- 3. 计算平均指标 ---
        # 题目主干相似度（分母是有效题目数，即GT正常的题目数）
        # 对于Pred未对齐或被拒答的题目，相似度设为0.0，但仍计入分母
        stem_sim = sum(stem_sims) / valid_question_count if valid_question_count > 0 else 0.0
        
        # 答案准确率（分母是有效题目数中的选择题和填空题数）
        # 对于Pred未对齐或被拒答的题目，答案准确率为0.0，但仍计入分母
        ans_acc = -1.0
        if valid_mcq_fib_count > 0:
            # 分母是有效题目数中的选择题和填空题数
            # 分子是所有有效选择题和填空题的答案准确率之和（Pred未对齐的为0.0）
            ans_acc = sum(ans_accs) / valid_mcq_fib_count
        elif ctx.gt_type in [QuestionType.MCQ, QuestionType.FIB]:
            # 如果没有计算任何答案准确率，但有选择题/填空题，设为0
            ans_acc = 0.0
        
        # 答案提取准确率（对比所有提取的答案）
        # 只针对选择题和填空题，解答题不适用
        answer_extraction_acc = -1.0
        
        # 检查是否有选择题或填空题（排除解答题）
        has_mcq_or_fib = False
        has_sub_only = False  # 是否只有解答题
        
        if ctx.gt_structure_map:
            # 检查所有题目类型
            question_types = set(ctx.gt_structure_map.values())
            if QuestionType.SUB in question_types:
                # 如果有解答题，检查是否只有解答题
                if question_types == {QuestionType.SUB}:
                    has_sub_only = True
                else:
                    # 有解答题但也有其他题型，检查是否有选择题或填空题
                    has_mcq_or_fib = any(q_type in [QuestionType.MCQ, QuestionType.FIB] for q_type in question_types)
            else:
                # 没有解答题，检查是否有选择题或填空题
                has_mcq_or_fib = any(q_type in [QuestionType.MCQ, QuestionType.FIB] for q_type in question_types)
        elif ctx.gt_type:
            # 如果没有 structure_map，使用整体类型
            if ctx.gt_type == QuestionType.SUB:
                has_sub_only = True
            elif ctx.gt_type in [QuestionType.MCQ, QuestionType.FIB]:
                has_mcq_or_fib = True
        
        # 如果只有解答题，保持 -1.0（不适用）
        if has_sub_only:
            answer_extraction_acc = -1.0
        elif has_mcq_or_fib:
            # 有选择题或填空题，计算答案提取准确率
            if ctx.gt_extracted_answers and ctx.pred_extracted_answers:
                matched_count = 0
                total_count = 0
                for q_num, gt_ans in ctx.gt_extracted_answers.items():
                    if q_num in ctx.pred_extracted_answers:
                        total_count += 1
                        pred_ans = str(ctx.pred_extracted_answers[q_num]).strip().upper()
                        gt_ans_str = str(gt_ans).strip().upper()
                        if pred_ans == gt_ans_str:
                            matched_count += 1
                
                if total_count > 0:
                    answer_extraction_acc = matched_count / total_count
                else:
                    answer_extraction_acc = 0.0
            else:
                # 有选择题/填空题但没有提取到答案，设为0
                answer_extraction_acc = 0.0
        
        # --- 4. 图片相似度（整体计算）---
        img_sim = self.image_comparator.calculate_average_similarity(
            ctx.gt_img_paths,
            ctx.pred_img_paths
        )
        
        # 验证：混淆矩阵的总数应该等于gt_structure_map中的题目数
        if ctx.gt_structure_map:
            total_from_cm = tp_count + fp_count + tn_count + fn_count
            total_from_gt = len(ctx.gt_structure_map)
            if total_from_cm != total_from_gt:
                logger.warning(
                    f"文件 {ctx.filename}: 混淆矩阵统计的题目数 ({total_from_cm}) "
                    f"与gt_structure_map中的题目数 ({total_from_gt}) 不一致。"
                    f"可能的原因：有些题目类型不在MCQ/FIB/SUB中，被跳过了统计。"
                )
        
        return self._finalize_metrics(ctx, stem_sim, ans_acc, answer_extraction_acc, img_sim, tp_count, fp_count, tn_count, fn_count, confusion_matrix_by_type, is_gt_rejected, is_pred_invalid)
    
    async def evaluate_async(self, ctx: ProcessContext) -> ProcessContext:
        """
        异步版本：对单个 ProcessContext 对象进行全面评估并填充 metrics 字段
        评估所有题目，排除GT中被拒答的题目
        """
        # 复用同步版本的逻辑，只将图片相似度计算改为异步
        # --- 1. 整体拒答状态判断（用于向后兼容）---
        # 注释掉旧的提前返回逻辑，让代码继续执行到遍历gt_structure_map的新逻辑
        # 即使all_question_segments为空（对齐失败），也应该统计gt_structure_map中的所有题目
        # is_gt_rejected = "[无法识别]" in ctx.gt_raw
        # is_pred_invalid = ctx.pred_is_rejected or not ctx.alignment_found
        # 
        # if not ctx.all_question_segments:
        #     if is_gt_rejected:
        #         ctx.metrics = {
        #             "stem_sim": 0.0,
        #             "img_sim": 0.0,
        #             "ans_acc": -1.0,
        #             "answer_extraction_acc": -1.0,
        #             "gt_rejected_flag": 1.0,
        #             "pred_rejected_flag": 1.0 if is_pred_invalid else 0.0,
        #             "tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0,
        #         }
        #         return ctx
        #     
        #     if is_pred_invalid:
        #         ctx.metrics = {
        #             "stem_sim": 0.0,
        #             "img_sim": 0.0,
        #             "ans_acc": 0.0,
        #             "answer_extraction_acc": 0.0,
        #             "gt_rejected_flag": 0.0,
        #             "pred_rejected_flag": 1.0,
        #             "tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0,
        #         }
        #         return ctx

        # --- 2. 对所有题目进行评估（同步部分，不涉及API调用）---
        gt_text_for_sim = ctx.gt_clean_for_alignment if ctx.gt_clean_for_alignment else ctx.gt_clean
        
        stem_sims = []
        ans_accs = []
        valid_question_count = 0
        valid_mcq_fib_count = 0
        confusion_matrix_by_type = {
            QuestionType.MCQ: {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            QuestionType.FIB: {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            QuestionType.SUB: {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        }
        tp_count = 0
        fp_count = 0
        tn_count = 0
        fn_count = 0
        
        # 为了简化，我们调用同步版本的评估逻辑，只将图片相似度改为异步
        # 先执行同步评估（除了图片相似度）
        # 创建一个临时ctx，执行同步评估
        temp_ctx = self.evaluate(ctx)
        
        # 异步计算图片相似度并更新
        import time
        img_start_time = time.time()
        logger.info(f"[Step 4-5/5] 开始计算图片相似度: GT图片数={len(ctx.gt_img_paths)}, Pred图片数={len(ctx.pred_img_paths)}")
        img_sim = await self.image_comparator.calculate_average_similarity_async(
            ctx.gt_img_paths,
            ctx.pred_img_paths
        )
        img_elapsed_time = time.time() - img_start_time
        logger.info(f"[Step 4-5/5] 图片相似度计算完成，耗时 {img_elapsed_time:.2f} 秒，相似度={img_sim:.4f}")
        
        # 更新图片相似度
        temp_ctx.metrics["img_sim"] = img_sim
        
        return temp_ctx
    
    def _finalize_metrics(self, ctx: ProcessContext, stem_sim: float, ans_acc: float, answer_extraction_acc: float, 
                         img_sim: float, tp_count: int, fp_count: int, tn_count: int, fn_count: int,
                         confusion_matrix_by_type: dict, is_gt_rejected: bool, is_pred_invalid: bool) -> ProcessContext:
        
        # --- 5. 计算Precision, Recall, F1 ---
        total_questions = tp_count + fp_count + tn_count + fn_count
        logger.info(f"混淆矩阵统计: TP={tp_count}, FP={fp_count}, TN={tn_count}, FN={fn_count}, 总计={total_questions}")
        
        # 整体Precision, Recall, F1（向后兼容）
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 按题型计算Precision, Recall, F1
        precision_by_type = {}
        recall_by_type = {}
        f1_by_type = {}
        
        for q_type, cm in confusion_matrix_by_type.items():
            type_tp = cm["tp"]
            type_fp = cm["fp"]
            type_fn = cm["fn"]
            
            type_precision = type_tp / (type_tp + type_fp) if (type_tp + type_fp) > 0 else 0.0
            type_recall = type_tp / (type_tp + type_fn) if (type_tp + type_fn) > 0 else 0.0
            type_f1 = 2 * (type_precision * type_recall) / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0.0
            
            type_name = q_type.value
            precision_by_type[type_name] = type_precision
            recall_by_type[type_name] = type_recall
            f1_by_type[type_name] = type_f1
        
        ctx.metrics = {
            "stem_sim": stem_sim,
            "img_sim": img_sim,
            "ans_acc": ans_acc,
            "answer_extraction_acc": answer_extraction_acc,
            "gt_rejected_flag": 1.0 if is_gt_rejected else 0.0,
            "pred_rejected_flag": 1.0 if is_pred_invalid else 0.0,
            "tp": float(tp_count),
            "fp": float(fp_count),
            "tn": float(tn_count),
            "fn": float(fn_count),
            "precision": precision,  # 整体指标（向后兼容）
            "recall": recall,  # 整体指标（向后兼容）
            "f1": f1,  # 整体指标（向后兼容）
            "confusion_matrix_by_type": {
                q_type.value: {
                    "tp": float(cm["tp"]),
                    "fp": float(cm["fp"]),
                    "tn": float(cm["tn"]),
                    "fn": float(cm["fn"]),
                }
                for q_type, cm in confusion_matrix_by_type.items()
            },
            "precision_by_type": precision_by_type,
            "recall_by_type": recall_by_type,
            "f1_by_type": f1_by_type,
        }

        return ctx
