# 数据对齐处理
# 1.利用llm找到GT的题目和选项在pred中对应的位置
# core/aligner.py
import logging
import re
from typing import List, Optional, Dict
from utils.llm_client import LLMClient
from utils.prompt import ALIGNER_USER_PROMPT
from core.schema import AlignmentResult, AlignmentSnippet, ProcessContext, SingleAlignmentResult
logger = logging.getLogger(__name__)

class Aligner:
    def __init__(self, llm_client: LLMClient):
        # 接收传入的 client (这里是 text_client)
        self.client = llm_client
        
        # 使用这个 client 创建 chain
        system_prompt = "你是一个专业的文本对齐专家，擅长在噪声文本中找到对应的标准文本片段。"
        self.chain = self.client.create_json_chain(
            pydantic_model=AlignmentResult,
            system_prompt=system_prompt,
            user_prompt_template=ALIGNER_USER_PROMPT
        )
    
    def align_all_questions(self, ctx: ProcessContext) -> List[AlignmentSnippet]:
        """
        对单个 GT 和 Pred 文件中的所有题目进行对齐（同步版本）。
        :param ctx: ProcessContext 对象，包含 gt_clean 和 pred_clean 文本。
        :return: 一个包含所有题目对齐信息的 Snippet 列表。
        """
        if not ctx.gt_clean or not ctx.pred_clean:
            logger.warning("GT 或 Pred 文本为空，无法执行对齐。")
            logger.debug(f"GT文本长度: {len(ctx.gt_clean) if ctx.gt_clean else 0}, Pred文本长度: {len(ctx.pred_clean) if ctx.pred_clean else 0}")
            return []

        # 记录文本长度用于诊断
        logger.debug(f"开始对齐: GT长度={len(ctx.gt_clean)}, Pred长度={len(ctx.pred_clean)}")
        logger.debug(f"GT文本前100字符: {ctx.gt_clean[:100] if len(ctx.gt_clean) > 100 else ctx.gt_clean}")
        logger.debug(f"Pred文本前100字符: {ctx.pred_clean[:100] if len(ctx.pred_clean) > 100 else ctx.pred_clean}")

        # 调用 LLMClient（同步）
        # 注意：这里使用的是 gt_clean 和 pred_clean，它们应该是对齐用的文本（无图、无大章节标题、无答案标签）
        result = self.client.invoke(self.chain, {
            "gt_content": ctx.gt_clean,
            "pred_content": ctx.pred_clean
        })
        
        return self._process_alignment_result(result, ctx)
    
    async def align_all_questions_async(self, ctx: ProcessContext) -> List[AlignmentSnippet]:
        """
        对单个 GT 和 Pred 文件中的所有题目进行对齐（异步版本）。
        :param ctx: ProcessContext 对象，包含 gt_clean 和 pred_clean 文本。
        :return: 一个包含所有题目对齐信息的 Snippet 列表。
        """
        if not ctx.gt_clean or not ctx.pred_clean:
            logger.warning("GT 或 Pred 文本为空，无法执行对齐。")
            logger.debug(f"GT文本长度: {len(ctx.gt_clean) if ctx.gt_clean else 0}, Pred文本长度: {len(ctx.pred_clean) if ctx.pred_clean else 0}")
            return []

        # 使用无图文本进行对齐（更小更快）
        gt_content = ctx.gt_clean_for_alignment if ctx.gt_clean_for_alignment else ctx.gt_clean
        pred_content = ctx.pred_clean_for_alignment if ctx.pred_clean_for_alignment else ctx.pred_clean
        
        # 记录文本长度用于诊断
        import time
        start_time = time.time()
        estimated_tokens = (len(gt_content) + len(pred_content)) // 4  # 粗略估算
        logger.info(f"开始异步对齐: GT长度={len(gt_content)}, Pred长度={len(pred_content)}, 估算token数≈{estimated_tokens}")
        logger.debug(f"GT文本前100字符: {gt_content[:100] if len(gt_content) > 100 else gt_content}")
        logger.debug(f"Pred文本前100字符: {pred_content[:100] if len(pred_content) > 100 else pred_content}")

        # 调用 LLMClient（异步）
        # 使用无图文本（gt_clean_for_alignment 和 pred_clean_for_alignment）进行对齐，减少token数量
        result = await self.client.ainvoke(self.chain, {
            "gt_content": gt_content,
            "pred_content": pred_content
        })
        
        elapsed_time = time.time() - start_time
        logger.info(f"异步对齐完成，耗时 {elapsed_time:.2f} 秒")
        
        return self._process_alignment_result(result, ctx)
    
    def _process_alignment_result(self, result: Optional[Dict], ctx: ProcessContext) -> List[AlignmentSnippet]:
        """
        处理对齐结果的通用方法（同步和异步共用）
        """
        if result is None:
            logger.error("LLM调用失败，返回None。可能是网络错误、超时或API错误。")
            return []
        
        if not isinstance(result, dict):
            logger.error(f"LLM返回格式错误：期望dict，实际得到{type(result)}。返回内容: {str(result)[:200]}")
            return []
        
        if "alignments" not in result:
            logger.error(f"LLM返回结果中缺少'alignments'键。返回的键: {list(result.keys())}")
            logger.debug(f"LLM返回的完整结果: {result}")
            return []
        
        if not result["alignments"]:
            logger.warning("LLM返回了空的alignments列表。可能是GT和Pred文本差异太大，无法找到匹配。")
            logger.warning(f"GT文本前200字符: {ctx.gt_clean[:200]}")
            logger.warning(f"Pred文本前200字符: {ctx.pred_clean[:200]}")
            logger.warning(f"GT文本长度: {len(ctx.gt_clean)}, Pred文本长度: {len(ctx.pred_clean)}")
            logger.debug(f"LLM返回的完整结果: {result}")
            # 尝试输出GT和Pred中的题号信息，帮助诊断
            import re
            gt_question_numbers = re.findall(r'^\s*(\d+)[\.。、]', ctx.gt_clean, re.MULTILINE)
            pred_question_numbers = re.findall(r'^\s*(\d+)[\.。、]', ctx.pred_clean, re.MULTILINE)
            logger.warning(f"GT中找到的题号: {gt_question_numbers[:10]} (共{len(gt_question_numbers)}个)")
            logger.warning(f"Pred中找到的题号: {pred_question_numbers[:10]} (共{len(pred_question_numbers)}个)")
            return []
        
        # Pydantic 会自动验证数据结构
        try:
            alignments = [AlignmentSnippet(**item) for item in result["alignments"]]
            logger.debug(f"成功解析{len(alignments)}个对齐结果")
            return alignments
        except Exception as e:
            logger.error(f"解析对齐结果时出错: {type(e).__name__}: {e}")
            logger.debug(f"第一个对齐项的内容: {result['alignments'][0] if result['alignments'] else 'N/A'}")
            import traceback
            logger.debug(f"完整错误信息: {traceback.format_exc()}")
            return []
    
    def align(self, gt_clean: str, pred_clean: str) -> tuple[List[Dict], Dict[str, int]]:
        """
        对齐所有题目，返回所有题目的对齐结果和统计信息（同步版本）
        
        :param gt_clean: 无图、无大章节标题的 GT 文本（用于对齐和截取）
        :param pred_clean: 无图、无大章节标题的 Pred 文本（用于对齐和截取）
        :return: (所有题目的对齐结果列表, 统计信息字典)
                 统计信息包含: total_aligned, exact_match_count, fuzzy_match_count
        """
        # 创建临时 ctx 用于调用 align_all_questions
        from core.schema import ProcessContext
        temp_ctx = ProcessContext(
            filename="temp",
            gt_raw="",
            pred_raw=""
        )
        temp_ctx.gt_clean = gt_clean
        temp_ctx.pred_clean = pred_clean
        # 调用 align_all_questions 获取所有题目的对齐结果（使用无图文本）
        alignments = self.align_all_questions(temp_ctx)
        
        return self._process_alignments(alignments, pred_clean)
    
    async def align_async(self, gt_clean: str, pred_clean: str) -> tuple[List[Dict], Dict[str, int]]:
        """
        对齐所有题目，返回所有题目的对齐结果和统计信息（异步版本）
        
        :param gt_clean: 无图、无大章节标题的 GT 文本（用于对齐和截取）
        :param pred_clean: 无图、无大章节标题的 Pred 文本（用于对齐和截取）
        :return: (所有题目的对齐结果列表, 统计信息字典)
                 统计信息包含: total_aligned, exact_match_count, fuzzy_match_count
        """
        # 创建临时 ctx 用于调用 align_all_questions_async
        from core.schema import ProcessContext
        temp_ctx = ProcessContext(
            filename="temp",
            gt_raw="",
            pred_raw=""
        )
        temp_ctx.gt_clean = gt_clean
        temp_ctx.pred_clean = pred_clean
        # 调用异步版本
        alignments = await self.align_all_questions_async(temp_ctx)
        
        return self._process_alignments(alignments, pred_clean)
    
    def _process_alignments(self, alignments: List[AlignmentSnippet], pred_clean: str) -> tuple[List[Dict], Dict[str, int]]:
        """
        处理对齐结果的通用方法（同步和异步共用）
        """
        
        if not alignments:
            # 没有找到对齐，返回空列表和空统计
            return [], {
                "total_aligned": 0,
                "exact_match_count": 0,
                "fuzzy_match_count": 0,
                "fallback_start_count": 0,
                "fallback_next_question_count": 0,
                "fallback_text_end_count": 0,
                "failed_alignment_count": 0,
                "failed_question_ids": [],  # 匹配失败的题号列表
                "total_questions": 0
            }
        
        # 处理所有题目（从无图、无大章节标题的文本中截取）
        all_question_results = []
        # 统计匹配类型
        exact_match_count = 0  # 直接匹配成功的题目数
        fuzzy_match_count = 0  # 模糊匹配的题目数
        fallback_start_count = 0  # Fallback匹配：开头锚点未找到，使用回退策略找到开头
        fallback_next_question_count = 0  # Fallback匹配：使用下一个题目开头作为结尾
        fallback_text_end_count = 0  # Fallback匹配：使用文本末尾作为结尾
        failed_alignment_count = 0  # 匹配失败的题目数（alignment_found=False）
        failed_question_ids = []  # 匹配失败的题号列表
        
        for alignment in alignments:
            # 根据锚点截取 pred_segment（从无图文本中截取）
            start_idx, start_match_type = self._fuzzy_find(pred_clean, alignment.pred_start_snippet)
            # 从开头之后开始找结尾锚点
            # 注意：如果 pred_end_snippet 是空字符串，find("") 会返回 start_idx，而不是 -1
            # 所以需要先检查 pred_end_snippet 是否为空
            if not alignment.pred_end_snippet or not alignment.pred_end_snippet.strip():
                end_idx = -1  # 空字符串视为未找到
                end_match_type = "not_found"
            else:
                if start_idx != -1:
                    end_idx, end_match_type = self._fuzzy_find(pred_clean, alignment.pred_end_snippet, start_idx)
                else:
                    end_idx = -1
                    end_match_type = "not_found"
            
            if start_idx == -1:
                # 开头锚点未找到，尝试fallback机制
                logger.warning(f"Question {alignment.question_id}: Start anchor not found: '{alignment.pred_start_snippet}'")
                
                # Fallback机制：尝试其他方式定位题目
                fallback_start_idx = self._fallback_find_start(alignment, pred_clean, alignments)
                
                if fallback_start_idx != -1:
                    logger.info(f"Question {alignment.question_id}: Found start position using fallback mechanism at {fallback_start_idx}")
                    start_idx = fallback_start_idx
                    start_match_type = "fallback"
                    fallback_start_count += 1
                else:
                    # 所有fallback方法都失败，跳过这个题目
                    logger.warning(f"Question {alignment.question_id}: All fallback methods failed, skipping this question")
                    failed_alignment_count += 1
                    failed_question_ids.append(alignment.question_id)
                    all_question_results.append({
                        "question_id": alignment.question_id,
                        "alignment_found": False,
                        "start_anchor": alignment.pred_start_snippet,
                        "end_anchor": alignment.pred_end_snippet,
                        "gt_start_snippet": alignment.gt_start_snippet,
                        "gt_end_snippet": alignment.gt_end_snippet,
                        "pred_segment": None,
                        "is_rejected": True,
                        "raw_handwritten_answer": None
                    })
                    continue
            
            # 如果结尾锚点未找到，或者结尾在开头之前，使用下一个题目的开头或文本末尾作为结尾
            used_fallback = False
            fallback_type = None  # "next_question" 或 "text_end"
            if end_idx == -1 or end_idx < start_idx:
                used_fallback = True
                # 尝试找到下一个题目的开头作为当前题的结尾
                next_start_idx = len(pred_clean)
                for next_alignment in alignments:
                    if next_alignment.question_id != alignment.question_id:
                        next_idx, _ = self._fuzzy_find(pred_clean, next_alignment.pred_start_snippet, start_idx)
                        if next_idx != -1 and next_idx < next_start_idx:
                            next_start_idx = next_idx
                
                # 如果通过LLM对齐没有找到下一题，尝试基于题号模式查找
                if next_start_idx == len(pred_clean):
                    # 查找下一个题号模式（如"14.", "15."等）
                    current_question_id = int(alignment.question_id) if alignment.question_id.isdigit() else 0
                    if current_question_id > 0:
                        # 查找下一个题号
                        next_question_pattern = re.compile(rf'{current_question_id + 1}\.\s')
                        next_match = next_question_pattern.search(pred_clean, start_idx)
                        if next_match:
                            next_start_idx = next_match.start()
                            logger.debug(f"Question {alignment.question_id}: Found next question by pattern matching at {next_start_idx}")
                
                if next_start_idx < len(pred_clean):
                    end_idx = next_start_idx
                    fallback_type = "next_question"
                else:
                    # 如果没有下一个题目，使用文本末尾
                    end_idx = len(pred_clean)
                    fallback_type = "text_end"
                    logger.warning(f"Question {alignment.question_id}: End anchor not found, using text end")
            
            # 截取 pred_segment（从无图、无大章节标题的文本中截取）
            # 如果使用了fallback逻辑，end_idx 是下一题的开头位置，需要截取到 end_idx 之前
            # 因为pred_end_snippet在pred_clean中找不到，加它的长度没有意义
            if used_fallback:
                # end_idx 是下一题的开头位置，需要确保不包含下一题
                # 向前查找，找到当前题目的真正结尾
                segment_end = end_idx
                
                # 向前搜索最多500字符，查找当前题目的结尾标记
                search_start = max(start_idx, end_idx - 500)
                search_text = pred_clean[search_start:end_idx]
                
                # 查找答案标签的位置（优先）
                answer_patterns = [
                    re.compile(r'[【\[(]\s*答案\s*[:：]\s*[^】\])]+[】\])]'),  # 完整答案标签
                    re.compile(r'[【\[(]\s*答案\s*[:：]\s*[A-Z0-9]+(?=\s*$)'),  # 不完整答案标签（行尾）
                ]
                for pattern in answer_patterns:
                    matches = list(pattern.finditer(search_text))
                    if matches:
                        last_match = matches[-1]
                        candidate_end = search_start + last_match.end()
                        # 检查答案标签之后是否还有内容，且不是下一题的开头
                        text_after = pred_clean[candidate_end:end_idx].strip()
                        # 如果答案标签之后的内容以题号开头，说明答案标签就是结尾
                        if not text_after or re.match(r'^\d+[\.。、]', text_after):
                            segment_end = candidate_end
                            break
                
                # 如果没找到答案标签，尝试查找最后一个选项的结束位置
                if segment_end == end_idx:
                    # 查找最后一个选项标记（如 "D. ..."）的位置
                    option_pattern = re.compile(r'^[A-Z][\.。、]\s+', re.MULTILINE)
                    option_matches = list(option_pattern.finditer(pred_clean[start_idx:end_idx]))
                    if option_matches:
                        last_option_match = option_matches[-1]
                        last_option_start = start_idx + last_option_match.start()
                        # 查找最后一个选项行的结束位置（换行符或文本末尾）
                        last_option_line_end = pred_clean.find('\n', last_option_start, end_idx)
                        if last_option_line_end != -1:
                            # 检查这一行之后是否就是下一题
                            text_after_option = pred_clean[last_option_line_end + 1:end_idx].strip()
                            if not text_after_option or re.match(r'^\d+[\.。、]', text_after_option):
                                segment_end = last_option_line_end + 1
            else:
                # 如果找到了pred_end_snippet，使用end_idx + len(pred_end_snippet)
                segment_end = min(end_idx + len(alignment.pred_end_snippet), len(pred_clean)) if alignment.pred_end_snippet and end_idx < len(pred_clean) else end_idx
            
            pred_segment = pred_clean[start_idx : segment_end]
            
            # 检查pred_segment是否只包含大章节标题（不应该被当作题目）
            # 如果只包含大章节标题，标记为对齐失败
            pred_segment_stripped = re.sub(r'\*\*', '', pred_segment).strip()
            large_section_patterns = [
                r'^[一二三四五六七八九十]+[、.]\s*[选填解].*?题.*?本题共',
                r'^[一二三四五六七八九十]+[、.]\s*[选填解].*?题.*?共\s*\d+\s*[小题分]',
                r'^[一二三四五六七八九十]+[、.]\s*[选填解].*?题.*?每小题',
                r'^[一二三四五六七八九十]+[、.]\s*[选填解].*?题.*?在每小题',
                r'^[一二三四五六七八九十]+[、.]\s*[选填解].*?题.*?本大题',
            ]
            is_section_title_only = False
            for pattern in large_section_patterns:
                if re.match(pattern, pred_segment_stripped):
                    is_section_title_only = True
                    break
            
            if is_section_title_only:
                # 如果pred_segment只包含大章节标题，标记为对齐失败
                logger.warning(f"Question {alignment.question_id}: pred_segment只包含大章节标题，标记为对齐失败")
                failed_alignment_count += 1
                failed_question_ids.append(alignment.question_id)
                all_question_results.append({
                    "question_id": alignment.question_id,
                    "alignment_found": False,
                    "start_anchor": alignment.pred_start_snippet,
                    "end_anchor": alignment.pred_end_snippet,
                    "gt_start_snippet": alignment.gt_start_snippet,
                    "gt_end_snippet": alignment.gt_end_snippet,
                    "pred_segment": None,
                    "is_rejected": True,
                    "raw_handwritten_answer": None
                })
                continue
            
            # 从 pred_segment 中提取手写答案和判断拒答状态
            is_rejected, raw_handwritten_answer = self._extract_answer_and_check_rejection(pred_segment)
            
            # 从对齐结果中提取答案（LLM已经提取了）
            gt_answer = getattr(alignment, 'gt_answer', '') or ""
            pred_answer = getattr(alignment, 'pred_answer', '') or ""
            
            # 验证LLM提取的答案是否有效（不是问题文本的一部分）
            gt_answer = self._validate_answer(gt_answer, alignment.gt_start_snippet, alignment.gt_end_snippet)
            pred_answer = self._validate_answer(pred_answer, alignment.pred_start_snippet, alignment.pred_end_snippet)
            
            # 如果LLM没有提取到答案，尝试从片段中提取
            if not gt_answer:
                gt_answer = self._extract_answer_from_snippets(alignment.gt_start_snippet, alignment.gt_end_snippet)
            if not pred_answer:
                pred_answer = self._extract_answer_from_snippets(alignment.pred_start_snippet, alignment.pred_end_snippet)
            
            # 从 pred_segment 中移除答案，避免影响文本相似度计算
            # 优先使用LLM提取的答案进行清理（LLM更准确，能识别跨行答案和格式不规范的答案）
            # 如果LLM没有提取到答案，pred_answer 可能为空，此时会使用正则表达式清理
            pred_segment_cleaned = self._remove_answer_from_segment(pred_segment, pred_answer)
            
            # 额外检查：如果pred_segment包含了下一题的开头，需要进一步清理
            # 这是对LLM输出的二次验证，确保边界准确性
            
            # 检查1: segment末尾是否包含下一题的题号
            next_question_pattern = re.compile(r'\n\s*(\d+)[\.。、]\s+')
            next_question_match = next_question_pattern.search(pred_segment_cleaned)
            if next_question_match:
                # 如果找到了下一题的题号，截取到题号之前
                next_question_start = next_question_match.start()
                pred_segment_cleaned = pred_segment_cleaned[:next_question_start].rstrip()
                logger.warning(f"Question {alignment.question_id}: Found next question number in segment, truncated to avoid overlap")
            
            # 检查2: segment末尾是否包含下一题的开头片段（通过检查是否包含下一题的start_snippet）
            if len(alignments) > 1:
                # 找到当前题目在alignments中的索引
                current_idx = None
                for idx, a in enumerate(alignments):
                    if a.question_id == alignment.question_id:
                        current_idx = idx
                        break
                
                # 如果当前题目不是最后一个，检查是否包含了下一题的开头
                if current_idx is not None and current_idx < len(alignments) - 1:
                    next_alignment = alignments[current_idx + 1]
                    next_start_snippet = next_alignment.pred_start_snippet
                    # 如果下一题的开头片段出现在当前segment中，需要截取
                    if next_start_snippet and len(next_start_snippet) > 10:  # 确保snippet足够长
                        # 查找下一题开头的简化版本（前15个字符）
                        next_start_prefix = next_start_snippet[:15].strip()
                        if next_start_prefix and next_start_prefix in pred_segment_cleaned:
                            # 找到下一题开头的位置
                            next_start_pos = pred_segment_cleaned.find(next_start_prefix)
                            # 只在segment后半部分（50%之后）才认为是问题，避免误判
                            if next_start_pos > len(pred_segment_cleaned) * 0.5:
                                pred_segment_cleaned = pred_segment_cleaned[:next_start_pos].rstrip()
                                logger.warning(f"Question {alignment.question_id}: Found next question start snippet in segment, truncated")
            
            # 统计匹配类型：
            # 1. 如果使用了fallback逻辑（结束锚点未找到），根据fallback类型分别统计
            # 2. 如果开始和结束锚点都是直接匹配，算作直接匹配
            # 3. 否则算作模糊匹配
            if used_fallback:
                if fallback_type == "next_question":
                    fallback_next_question_count += 1
                elif fallback_type == "text_end":
                    fallback_text_end_count += 1
            elif start_match_type == "exact" and end_match_type == "exact":
                exact_match_count += 1
            elif start_idx != -1:  # 至少找到了开始锚点（成功对齐）
                fuzzy_match_count += 1
            
            all_question_results.append({
                "question_id": alignment.question_id,
                "alignment_found": True,
                "start_anchor": alignment.pred_start_snippet,
                "end_anchor": alignment.pred_end_snippet,
                "gt_start_snippet": alignment.gt_start_snippet,
                "gt_end_snippet": alignment.gt_end_snippet,
                "pred_segment": pred_segment_cleaned,  # 使用清理后的segment（已移除答案）
                "is_rejected": is_rejected,
                "raw_handwritten_answer": raw_handwritten_answer,
                "gt_answer": gt_answer,  # 从对齐结果中提取的GT答案
                "pred_answer": pred_answer  # 从对齐结果中提取的Pred答案
            })
        
        # 输出匹配统计日志
        # 注意：fallback_start_count 只是标记使用了start fallback，但该对齐已经在其他类别中被计数了
        # 所以 total_aligned 不应该包含 fallback_start_count，否则会重复计数
        total_aligned = exact_match_count + fuzzy_match_count + fallback_next_question_count + fallback_text_end_count
        total_processed = total_aligned + failed_alignment_count
        
        # 从对齐结果中收集所有失败的题号（确保完整性）
        actual_failed_ids = [q.get("question_id") for q in all_question_results if not q.get("alignment_found", True)]
        # 确保 failed_question_ids 包含所有实际失败的题号
        failed_question_ids = list(set(failed_question_ids + actual_failed_ids))
        
        # 验证统计一致性：处理的总数应该等于LLM返回的题目数
        if total_processed != len(alignments):
            logger.warning(
                f"对齐统计不一致: 成功对齐({total_aligned}) + 匹配失败({failed_alignment_count}) = {total_processed}, "
                f"但LLM返回的题目数 = {len(alignments)}, 差异 = {total_processed - len(alignments)}"
            )
            # 修正failed_alignment_count，确保一致性
            # 确保 failed_alignment_count 不为负数
            failed_alignment_count = max(0, len(alignments) - total_aligned)
            if len(alignments) - total_aligned < 0:
                logger.error(
                    f"计算错误：成功对齐数({total_aligned}) > LLM返回的题目数({len(alignments)})，"
                    f"这可能是因为统计逻辑存在bug。将failed_alignment_count设为0。"
                )
            else:
                logger.warning(f"已修正failed_alignment_count为: {failed_alignment_count}")
                # 如果修正后的失败数量与收集到的失败题号数量不一致，尝试找出缺失的题号
                if len(failed_question_ids) != failed_alignment_count:
                    # 找出所有对齐结果中的题号
                    all_result_question_ids = {q.get("question_id") for q in all_question_results}
                    # 找出所有LLM返回的题号
                    all_alignment_question_ids = {a.question_id for a in alignments}
                    # 找出在LLM返回中但不在结果中的题号（可能被跳过的题号）
                    missing_question_ids = all_alignment_question_ids - all_result_question_ids
                    if missing_question_ids:
                        failed_question_ids = list(set(failed_question_ids + list(missing_question_ids)))
                        logger.warning(f"发现 {len(missing_question_ids)} 个缺失的题号，已添加到失败列表: {missing_question_ids}")
        
        logger.info(f"题目对齐统计: 总共对齐 {len(alignments)} 道题目，成功对齐 {total_aligned} 道题目，匹配失败 {failed_alignment_count} 道题目")
        logger.info(f"  - 直接匹配成功: {exact_match_count} 道题目")
        logger.info(f"  - 模糊匹配成功: {fuzzy_match_count} 道题目")
        logger.info(f"  - Fallback匹配（开头锚点回退）: {fallback_start_count} 道题目")
        logger.info(f"  - Fallback匹配（使用下一个题目开头）: {fallback_next_question_count} 道题目")
        logger.info(f"  - Fallback匹配（使用文本末尾）: {fallback_text_end_count} 道题目")
        if failed_alignment_count > 0:
            logger.warning(f"  - 匹配失败: {failed_alignment_count} 道题目，失败的题号: {failed_question_ids}")
        
        # 返回结果和统计信息
        alignment_stats = {
            "total_aligned": total_aligned,
            "exact_match_count": exact_match_count,
            "fuzzy_match_count": fuzzy_match_count,
            "fallback_start_count": fallback_start_count,
            "fallback_next_question_count": fallback_next_question_count,
            "fallback_text_end_count": fallback_text_end_count,
            "failed_alignment_count": failed_alignment_count,
            "failed_question_ids": failed_question_ids,  # 匹配失败的题号列表
            "total_questions": len(alignments)  # 注意：这是LLM返回的题目数，在pipeline.py中会被更新为gt_structure_map的题目数
        }
        
        return all_question_results, alignment_stats
    
    def _extract_answer_and_check_rejection(self, pred_segment: str) -> tuple[bool, Optional[str]]:
        """
        从 pred_segment 中提取手写答案并判断是否拒答
        
        :param pred_segment: Pred 中截取的题目片段
        :return: (is_rejected, raw_handwritten_answer)
        """
        if not pred_segment:
            return True, None
        
        # 检查是否包含拒答标记
        rejection_keywords = ["无法识别", "无法辨认", "无法读取", "拒答", "无法回答", "[无法识别]"]
        is_rejected = any(keyword in pred_segment for keyword in rejection_keywords)
        
        # 简单的答案提取逻辑
        # 对于选择题：查找选项标记（A. B. C. D.）后的内容
        # 对于填空题：查找填空标记（__、___、______等）后的内容
        raw_handwritten_answer = None
        
        if not is_rejected:
            # 尝试提取答案
            # 方法1: 查找答案标签（【答案：X】等）
            answer_pattern = re.compile(r'[【\[(]\s*答案\s*[:：]\s*([^】\])]+)[】\])]')
            answer_match = answer_pattern.search(pred_segment)
            if answer_match:
                raw_handwritten_answer = answer_match.group(1).strip()
            else:
                # 方法2: 查找选项标记后的内容（选择题）
                option_pattern = re.compile(r'[（(]\s*([A-Z]+)\s*[）)]')
                option_match = option_pattern.search(pred_segment)
                if option_match:
                    raw_handwritten_answer = option_match.group(1).strip()
                else:
                    # 方法3: 查找填空标记后的内容（填空题）
                    # 这里可以添加更复杂的逻辑
                    pass
        
        return is_rejected, raw_handwritten_answer
    
    def _fuzzy_find(self, text: str, pattern: str, start_pos: int = 0) -> tuple[int, str]:
        """
        模糊匹配查找，允许一定的容错率
        
        :param text: 待搜索的文本
        :param pattern: 要查找的模式
        :param start_pos: 开始搜索的位置
        :return: (匹配位置的索引，匹配类型) 元组，未找到返回(-1, "not_found")
                匹配类型: "exact" 表示直接匹配, "fuzzy" 表示模糊匹配, "status_tag_removed" 表示去除状态标签后匹配
        """
        if not pattern or not pattern.strip():
            return -1, "not_found"
        
        # 首先尝试精确匹配
        exact_idx = text.find(pattern, start_pos)
        if exact_idx != -1:
            return exact_idx, "exact"
        
        # 策略0: 如果pattern以状态标签开头（如[格式正常]、[条件缺失]等），尝试去除标签后再匹配
        # 这是因为预处理器可能已经移除了这些标签，但LLM生成的snippet中可能还包含它们
        import re
        import string
        
        status_tag_pattern = re.compile(r'^\[格式.*?\]\s*')
        if status_tag_pattern.match(pattern):
            # 移除状态标签
            pattern_without_tag = status_tag_pattern.sub('', pattern).strip()
            if pattern_without_tag:
                # 尝试用去除标签后的pattern匹配
                tag_removed_idx = text.find(pattern_without_tag, start_pos)
                if tag_removed_idx != -1:
                    return tag_removed_idx, "status_tag_removed"
        
        # 如果精确匹配失败，尝试模糊匹配
        # 策略1: 移除标点符号和空格差异
        
        # 清理模式中的标点和空格 - 使用Python标准标点符号
        punctuation_chars = string.punctuation + '，。！？；：""‘’“”（）【】《》〈〉〔〕…—～·'
        pattern_clean = re.sub(r'\s+', '', pattern)  # 先移除空格
        pattern_clean = ''.join(char for char in pattern_clean if char not in punctuation_chars)
        
        # 在文本中查找相似的子串
        for i in range(start_pos, len(text) - len(pattern_clean) + 1):
            # 提取文本片段并清理
            text_segment = text[i:i + min(len(pattern_clean) + 20, len(text) - i)]
            text_clean = re.sub(r'\s+', '', text_segment)  # 先移除空格
            text_clean = ''.join(char for char in text_clean if char not in punctuation_chars)
            
            # 使用编辑距离进行模糊匹配
            if self._similarity(pattern_clean, text_clean[:len(pattern_clean)]) > 0.8:
                return i, "fuzzy"
        
        # 策略2: 查找包含模式关键词的片段
        keywords = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]{2,}', pattern)
        if len(keywords) >= 2:
            # 查找包含多个关键词的片段
            for i in range(start_pos, len(text) - 50):
                segment = text[i:i + 100]
                matched_keywords = sum(1 for keyword in keywords if keyword in segment)
                if matched_keywords >= len(keywords) * 0.7:  # 匹配70%以上的关键词
                    return i, "fuzzy"
        
        return -1, "not_found"
    
    def _similarity(self, s1: str, s2: str) -> float:
        """
        计算两个字符串的相似度（基于编辑距离）
        
        :param s1: 字符串1
        :param s2: 字符串2
        :return: 相似度分数（0-1）
        """
        if not s1 or not s2:
            return 0.0
        
        # 简单的编辑距离相似度计算
        len_s1, len_s2 = len(s1), len(s2)
        max_len = max(len_s1, len_s2)
        
        if max_len == 0:
            return 1.0
        
        # 计算编辑距离
        distance = self._levenshtein_distance(s1, s2)
        
        # 计算相似度
        similarity = 1.0 - distance / max_len
        return max(0.0, similarity)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        计算两个字符串的编辑距离
        
        :param s1: 字符串1
        :param s2: 字符串2
        :return: 编辑距离
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _validate_answer(self, answer: str, start_snippet: str, end_snippet: str) -> str:
        """
        验证提取的答案是否有效
        
        如果答案看起来像是问题文本的一部分（如"则...的值为"、"的坐标为"等），
        则认为是无效答案，返回空字符串
        
        :param answer: 提取的答案
        :param start_snippet: 开头片段
        :param end_snippet: 结尾片段
        :return: 验证后的答案（如果无效则返回空字符串）
        """
        if not answer:
            return ""
        
        answer = answer.strip()
        
        # 检查答案是否包含常见的问题文本模式
        # 使用 re.search() 而不是 re.match()，因为问题文本可能出现在答案的任何位置
        question_patterns = [
            r'[。，,]\s*则',
            r'则.*?的值为?',
            r'的坐标[是为]',
            r'的值为?',
            r'为.*?的',
            r'是.*?的',
            r'投影向量的坐标',
            r'则.*?为',
            r'.*?的坐标[是为]',  # 匹配"投影向量的坐标为"等
            r'.*?的值为?$',  # 匹配以"的值为"结尾的
            r'^[】\]]',  # 匹配以"】"或"]"开头的（通常是题目内容的一部分）
            r'的取值范围是',  # 匹配"的取值范围是"（问题文本）
            r'的取值',  # 匹配"的取值"（问题文本）
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, answer):
                # 如果匹配到问题文本模式，认为是无效答案
                return ""
        
        # 检查答案是否在end_snippet中（如果答案就是end_snippet的一部分，可能是错误的）
        if answer in end_snippet and len(answer) > 5:
            # 如果答案长度较长且完全包含在end_snippet中，可能是问题文本
            # 但如果是短答案（如"3"、"A"等），可能是正确的
            if len(answer) > 10:
                return ""
        
        # 检查答案是否包含end_snippet中的关键词（即使顺序不同）
        # 如果答案长度 > 8 且包含end_snippet中的多个关键词，可能是问题文本
        if len(answer) > 8 and end_snippet:
            # 提取end_snippet中的中文关键词
            end_keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', end_snippet)
            answer_keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', answer)
            # 如果答案中包含end_snippet中的多个关键词，可能是问题文本
            if len(end_keywords) >= 2 and len(answer_keywords) >= 2:
                common_keywords = set(end_keywords) & set(answer_keywords)
                if len(common_keywords) >= 2:
                    # 如果共同关键词数量 >= 2，且答案长度 > 8，可能是问题文本
                    return ""
        
        # 检查答案是否包含题号（如 "4."、"5." 等）
        # 如果答案中包含题号，说明可能包含了整个题目内容，应该被过滤
        if re.search(r'\d+[\.。、]\s*', answer):
            # 如果答案长度 > 20，且包含题号，很可能是整个题目内容
            if len(answer) > 20:
                return ""
        
        # 检查答案是否包含多个选项标记（如 "A."、"B."、"C."、"D." 等）
        # 如果答案中包含多个选项标记，说明可能包含了整个题目内容，应该被过滤
        option_markers = re.findall(r'[A-Z][\.。、]\s+', answer)
        if len(option_markers) >= 2:
            # 如果答案长度 > 30，且包含多个选项标记，很可能是整个题目内容
            if len(answer) > 30:
                return ""
        
        # 检查答案是否包含明显的题目开头关键词（如"已知"、"设"、"求"等）
        # 如果答案长度较长且包含这些关键词，可能是整个题目内容
        question_start_keywords = ['已知', '设', '求', '证明', '计算', '若', '当', '函数', '集合', '向量']
        if len(answer) > 30:
            keyword_count = sum(1 for keyword in question_start_keywords if keyword in answer)
            if keyword_count >= 2:
                # 如果包含多个题目开头关键词，很可能是整个题目内容
                return ""
        
        # 检查答案长度：正确答案通常很短，如果超过50个字符，很可能是整个题目内容
        # 选择题答案通常是1-4个字母，填空题答案通常是数字或短表达式
        if len(answer) > 50:
            return ""
        
        return answer
    
    def _extract_answer_from_snippets(self, start_snippet: str, end_snippet: str) -> str:
        """
        从片段中提取答案，支持多种格式：
        1. 填空题：答案在题干末尾，用下划线包围，如 ___3_____ 或 ____4____
        2. 选择题：答案在题干行末尾，如 ABD
        3. 答案标签：【答案：X】或 [答案：X]
        
        :param start_snippet: 开头片段
        :param end_snippet: 结尾片段
        :return: 提取的答案，如果找不到返回空字符串
        """
        # 合并片段文本
        text = start_snippet + " " + end_snippet
        
        # 方法1: 查找答案标签（【答案：X】等）
        # 使用非贪婪匹配，确保能正确提取包含括号的答案（如 "(7√(14))/(14)"）
        answer_patterns = [
            re.compile(r'【\s*答案\s*[:：]\s*(.+?)】'),  # 【答案：...】
            re.compile(r'\[\s*答案\s*[:：]\s*(.+?)\]'),  # [答案：...]
            re.compile(r'\(\s*答案\s*[:：]\s*(.+?)\)'),   # (答案：...)
        ]
        for pattern in answer_patterns:
            answer_match = pattern.search(text)
            if answer_match:
                return answer_match.group(1).strip()
        
        # 方法2: 查找填空题答案（下划线包围的数字或表达式）
        # 匹配模式：___数字___ 或 ____数字____ 等
        # 注意：只匹配下划线之间有实际内容的，空下划线（如"________"）不匹配
        fill_pattern = re.compile(r'_+([^_\s]+[^_]*?[^_\s]+)_+')
        fill_matches = fill_pattern.findall(text)
        if fill_matches:
            # 取最后一个匹配（通常是题干末尾的答案）
            # 过滤掉只包含标点符号或空格的匹配
            valid_matches = [m.strip() for m in fill_matches if m.strip() and not re.match(r'^[，,。.；;：:、\s]+$', m.strip())]
            if valid_matches:
                return valid_matches[-1].strip()
        
        # 方法2b: 如果方法2没找到，尝试匹配下划线中的数字或表达式（更宽松的模式）
        # 但排除空下划线和只包含标点的下划线
        fill_pattern2 = re.compile(r'_+([^_\n]{1,20}?)_+')
        fill_matches2 = fill_pattern2.findall(text)
        if fill_matches2:
            # 过滤：只保留包含数字、字母或数学符号的匹配
            valid_matches2 = []
            for m in fill_matches2:
                m_clean = m.strip()
                # 如果包含数字、字母或常见数学符号，认为是有效答案
                if m_clean and re.search(r'[0-9a-zA-Z+\-*/=()\[\]{}√]', m_clean):
                    # 排除只包含标点符号的
                    if not re.match(r'^[，,。.；;：:、\s]+$', m_clean):
                        valid_matches2.append(m_clean)
            if valid_matches2:
                return valid_matches2[-1].strip()
        
        # 方法3: 查找选择题答案（题干行末尾的选项组合，如 ABD 或 (C)、（D））
        # 识别题干行：不包含选项标记（A. B. C. D.）的行，且行末尾有答案
        lines = text.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # 排除选项行：包含选项标记（A. B. C. D.）的行
            option_marker_pattern = re.compile(r'^[A-Z]\.\s+')
            if option_marker_pattern.match(line_stripped):
                continue  # 跳过选项行
            
            # 方法3a: 匹配括号格式的答案（如 (C)、（D）、（ABD ））
            bracket_answer_pattern = re.compile(r'[（(]\s*([A-Z]+)\s*[）)]\s*$')
            bracket_match = bracket_answer_pattern.search(line_stripped)
            if bracket_match:
                answer = bracket_match.group(1).strip()
                # 验证是否是有效的选项组合（只包含A-Z）
                if answer and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in answer):
                    return answer
            
            # 方法3b: 匹配题干行末尾的连续大写字母（2-4个），前面可能有空格
            # 确保不是选项标记的一部分
            choice_pattern = re.compile(r'\s+([A-Z]{2,4})\s*$')
            choice_match = choice_pattern.search(line_stripped)
            if choice_match:
                answer = choice_match.group(1).strip()
                # 验证是否是有效的选项组合（只包含A-Z，且不在选项标记中）
                if answer and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in answer):
                    # 确保答案不在选项标记中（如 "A. B. C. D."）
                    if not re.search(r'[A-Z]\.\s*' + re.escape(answer), line_stripped):
                        return answer
        
        return ""
    
    def _remove_answer_from_segment(self, segment: str, answer: str) -> str:
        """
        从题干片段中移除答案，避免影响文本相似度计算
        
        支持的答案格式：
        1. 填空题：___3_____ 或 ____4____
        2. 选择题：题干行末尾的选项组合（如 ABD）或括号中的答案（如 (A)）
        3. 答案标签：【答案：X】或 [答案：X]
        4. 选项列表：A. ... B. ... C. ... D. ...（如果答案为空，移除所有选项）
        
        :param segment: 题目片段
        :param answer: 提取到的答案
        :return: 移除答案后的片段
        """
        if not segment:
            return segment
        
        cleaned = segment
        
        # 方法1: 移除答案标签（【答案：X】等）
        # 先匹配完整的答案标签（有右括号）
        answer_label_pattern = re.compile(r'[【\[(]\s*答案\s*[:：]\s*[^】\])]+[】\])]')
        cleaned = answer_label_pattern.sub('', cleaned)
        
        # 再匹配不完整的答案标签（缺少右括号，可能是文本被截断）
        # 匹配到行尾或文本末尾
        incomplete_answer_pattern = re.compile(r'[【\[(]\s*答案\s*[:：]\s*[^】\])]+(?=\s*$|\s*\n|$)')
        cleaned = incomplete_answer_pattern.sub('', cleaned)
        
        # 匹配行尾的答案标签片段（如 "【答案：B" 在行尾）
        incomplete_answer_line_end = re.compile(r'[【\[(]\s*答案\s*[:：]\s*[A-Z0-9]+(?=\s*$)')
        cleaned = incomplete_answer_line_end.sub('', cleaned)
        
        # 方法2: 移除填空题答案（下划线包围的内容）
        # 匹配模式：___数字___ 或 ____数字____ 等
        # 注意：要匹配下划线中的内容，包括数字、表达式等
        fill_pattern = re.compile(r'_+[^_\n]+?_+')
        cleaned = fill_pattern.sub('', cleaned)
        
        # 方法3: 移除选择题答案（题干行末尾的选项组合）
        # 如果答案是大写字母组合（如 ABD），从每行末尾移除
        if answer and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in answer):
            lines = cleaned.split('\n')
            cleaned_lines = []
            for line in lines:
                # 先尝试移除括号中的答案，如 "(A)" 或 "(AB)"
                paren_pattern = re.compile(r'\(\s*' + re.escape(answer) + r'\s*\)')
                cleaned_line = paren_pattern.sub('', line)
                # 再尝试移除行末尾的答案（不带括号），如 "A" 或 "AB"
                choice_pattern = re.compile(r'\s+' + re.escape(answer) + r'(?=\s*$|\s*[。.，,；;：:])')
                cleaned_line = choice_pattern.sub('', cleaned_line)
                cleaned_lines.append(cleaned_line)
            cleaned = '\n'.join(cleaned_lines)
        
        # 方法4: 如果答案为空，不应该移除选项列表
        # 选项是题目的一部分，应该参与相似度计算
        # 即使答案提取失败，选项也应该保留
        # 注释掉原来的逻辑，因为选项应该参与相似度计算
        # if not answer:
        #     # 匹配选项列表模式：从第一个选项开始到文本末尾
        #     # 选项标记：A. B. C. D. 等，后面跟内容
        #     # 查找第一个选项标记的位置（使用单词边界，匹配行内或行首的选项）
        #     first_option_pattern = re.compile(r'\b([A-Z])\.\s+')
        #     option_matches = list(first_option_pattern.finditer(cleaned))
        #     if len(option_matches) >= 2:  # 至少2个选项才认为是选项列表
        #         # 找到第一个选项的位置
        #         first_option_pos = option_matches[0].start()
        #         # 检查第一个选项之前是否有题干内容（至少10个字符）
        #         # 如果有，则移除从第一个选项开始到末尾的所有内容
        #         if first_option_pos > 10:
        #             # 移除选项列表，保留题干部分
        #             cleaned = cleaned[:first_option_pos].rstrip()
        #             # 清理末尾可能残留的 "= ( )" 或 "为 ( )" 或类似格式
        #             cleaned = re.sub(r'\s*[=为]\s*\(\s*\)\s*$', '', cleaned)
        #             cleaned = cleaned.rstrip()
        
        # 清理多余的空格（但保留换行）
        # 不要合并所有空格，只清理连续的空格
        cleaned = re.sub(r' {2,}', ' ', cleaned)  # 多个空格合并为一个
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # 多个空行最多保留两个
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _fallback_find_start(self, alignment: AlignmentSnippet, pred_clean: str, all_alignments: List[AlignmentSnippet]) -> int:
        """
        当开头锚点找不到时，使用fallback机制尝试定位题目
        
        Fallback策略（按优先级）：
        1. 使用题号在Pred中查找（如 "1.", "2." 等）
        2. 使用GT题目的前几个关键词在Pred中查找
        3. 使用题目的顺序位置（如果前一个题目对齐成功，从它的结尾之后开始查找）
        
        :param alignment: 当前题目的对齐信息
        :param pred_clean: Pred文本（无图、无大章节标题）
        :param all_alignments: 所有题目的对齐信息列表
        :return: 找到的位置索引，未找到返回-1
        """
        import re
        
        # 策略1: 使用题号查找
        question_id = alignment.question_id
        if question_id and question_id.isdigit():
            # 尝试多种题号格式
            question_number_patterns = [
                re.compile(rf'^\s*{re.escape(question_id)}\.\s+', re.MULTILINE),  # "1. "
                re.compile(rf'^\s*{re.escape(question_id)}[\.。、]\s+', re.MULTILINE),  # "1." 或 "1。" 或 "1、"
                re.compile(rf'\s+{re.escape(question_id)}\.\s+'),  # 行内 " 1. "
            ]
            
            for pattern in question_number_patterns:
                matches = list(pattern.finditer(pred_clean))
                if matches:
                    # 如果有多个匹配，选择最合理的一个
                    # 如果前一个题目对齐成功，选择在前一个题目结尾之后的第一个匹配
                    best_match = None
                    if len(all_alignments) > 1:
                        # 找到当前题目在列表中的位置
                        current_idx = None
                        for idx, a in enumerate(all_alignments):
                            if a.question_id == alignment.question_id:
                                current_idx = idx
                                break
                        
                        # 如果当前题目不是第一个，尝试从前一个题目的结尾之后查找
                        if current_idx is not None and current_idx > 0:
                            prev_alignment = all_alignments[current_idx - 1]
                            # 尝试找到前一个题目的结尾位置
                            prev_end_idx, _ = self._fuzzy_find(pred_clean, prev_alignment.pred_end_snippet)
                            if prev_end_idx != -1:
                                # 在前一个题目结尾之后查找题号
                                for match in matches:
                                    if match.start() > prev_end_idx:
                                        best_match = match
                                        break
                    
                    # 如果没有找到最佳匹配，使用第一个匹配
                    if best_match is None:
                        best_match = matches[0]
                    
                    logger.debug(f"Question {question_id}: Found by question number pattern at {best_match.start()}")
                    return best_match.start()
        
        # 策略2: 使用GT题目的前几个关键词查找
        if alignment.gt_start_snippet:
            # 提取GT开头片段中的关键词（中文词、英文单词、数字）
            keywords = re.findall(r'[\u4e00-\u9fa5]{2,}|[a-zA-Z]{3,}|\d+', alignment.gt_start_snippet)
            if len(keywords) >= 2:
                # 使用前2-3个关键词进行匹配
                search_keywords = keywords[:3]
                # 构建搜索模式：关键词之间允许有少量其他字符
                pattern_parts = []
                for keyword in search_keywords:
                    pattern_parts.append(re.escape(keyword))
                # 允许关键词之间有最多50个字符的间隔
                pattern = re.compile(r'.{0,50}'.join(pattern_parts), re.IGNORECASE)
                
                matches = list(pattern.finditer(pred_clean))
                if matches:
                    # 选择第一个匹配
                    logger.debug(f"Question {alignment.question_id}: Found by keywords at {matches[0].start()}")
                    return matches[0].start()
        
        # 策略3: 使用题目顺序位置
        if len(all_alignments) > 1:
            # 找到当前题目在列表中的位置
            current_idx = None
            for idx, a in enumerate(all_alignments):
                if a.question_id == alignment.question_id:
                    current_idx = idx
                    break
            
            # 如果当前题目不是第一个，尝试从前一个题目的结尾之后查找
            if current_idx is not None and current_idx > 0:
                prev_alignment = all_alignments[current_idx - 1]
                # 尝试找到前一个题目的结尾位置
                prev_end_idx, _ = self._fuzzy_find(pred_clean, prev_alignment.pred_end_snippet)
                if prev_end_idx != -1:
                    # 在前一个题目结尾之后，查找下一个题号或关键词
                    search_start = prev_end_idx
                    search_text = pred_clean[search_start:]
                    
                    # 先尝试查找题号
                    if question_id and question_id.isdigit():
                        next_q_pattern = re.compile(rf'^\s*{re.escape(question_id)}\.\s+', re.MULTILINE)
                        next_match = next_q_pattern.search(search_text)
                        if next_match:
                            logger.debug(f"Question {alignment.question_id}: Found by sequential position at {search_start + next_match.start()}")
                            return search_start + next_match.start()
                    
                    # 再尝试查找关键词
                    if alignment.gt_start_snippet:
                        keywords = re.findall(r'[\u4e00-\u9fa5]{2,}|[a-zA-Z]{3,}', alignment.gt_start_snippet)
                        if len(keywords) >= 2:
                            search_keywords = keywords[:2]
                            pattern_parts = [re.escape(kw) for kw in search_keywords]
                            pattern = re.compile(r'.{0,30}'.join(pattern_parts), re.IGNORECASE)
                            match = pattern.search(search_text)
                            if match:
                                logger.debug(f"Question {alignment.question_id}: Found by sequential keywords at {search_start + match.start()}")
                                return search_start + match.start()
        
        # 所有fallback方法都失败
        return -1
        
    def _extract_single_question_from_gt(self, gt_text: str, question_id: str) -> Optional[str]:
        """
        从GT文本中提取单个题目的内容
        
        :param gt_text: GT文本
        :param question_id: 题目编号
        :return: 题目内容，如果找不到返回None
        """
        import re
        
        # 查找题目在文本中的位置
        question_patterns = [
            rf'^\s*{re.escape(question_id)}\.\s+',
            rf'^\s*{re.escape(question_id)}[\.。、]\s+',
            rf'\*\*{re.escape(question_id)}\.\*\*',
            rf'\*\*{re.escape(question_id)}\.',
            rf'{re.escape(question_id)}\.\*\*',
        ]
        
        question_start = -1
        for pattern in question_patterns:
            match = re.search(pattern, gt_text, re.MULTILINE)
            if match:
                question_start = match.start()
                break
        
        if question_start == -1:
            # 尝试正则匹配（更宽松）
            pattern = re.compile(rf'\*?{re.escape(question_id)}\*?[\.。、]')
            match = pattern.search(gt_text)
            if match:
                question_start = match.start()
        
        if question_start == -1:
            return None
        
        # 找到下一个题目的开头或文本末尾
        try:
            next_q_num = str(int(question_id) + 1)
            next_patterns = [
                rf'^\s*{re.escape(next_q_num)}\.\s+',
                rf'^\s*{re.escape(next_q_num)}[\.。、]\s+',
                rf'\*\*{re.escape(next_q_num)}\.\*\*',
            ]
            question_end = -1
            for pattern in next_patterns:
                match = re.search(pattern, gt_text[question_start:], re.MULTILINE)
                if match:
                    question_end = question_start + match.start()
                    break
            
            if question_end == -1:
                pattern = re.compile(rf'\*?{re.escape(next_q_num)}\*?[\.。、]')
                match = pattern.search(gt_text, question_start)
                if match:
                    question_end = match.start()
            
            if question_end == -1:
                question_end = len(gt_text)
        except ValueError:
            question_end = len(gt_text)
        
        # 提取题目内容
        question_content = gt_text[question_start:question_end]
        return question_content
    
    def align_single_question(self, question_id: str, gt_text: str, pred_text: str) -> Optional[Dict]:
        """
        对单个题目进行对齐（用于第二轮匹配，同步版本）
        
        :param question_id: 题目编号
        :param gt_text: 完整的GT文本（用于提取单个题目）
        :param pred_text: Pred文本
        :return: 对齐结果字典，如果失败返回None
        """
        # 从GT中提取单个题目的内容
        gt_question_content = self._extract_single_question_from_gt(gt_text, question_id)
        if not gt_question_content:
            logger.warning(f"无法从GT中提取题目 {question_id} 的内容")
            return None
        
        # 创建临时ctx用于调用LLM
        from core.schema import ProcessContext
        temp_ctx = ProcessContext(
            filename="temp_single",
            gt_raw="",
            pred_raw=""
        )
        temp_ctx.gt_clean = gt_question_content
        temp_ctx.pred_clean = pred_text
        
        # 调用LLM进行对齐（同步）
        result = self.client.invoke(self.chain, {
            "gt_content": gt_question_content,
            "pred_content": pred_text
        })
        
        # 处理对齐结果
        alignments = self._process_alignment_result(result, temp_ctx)
        
        if not alignments:
            logger.debug(f"题目 {question_id} 第二轮对齐失败：LLM未返回对齐结果")
            return None
        
        # 只返回第一个对齐结果（应该只有一个）
        alignment = alignments[0]
        
        # 处理对齐结果，提取pred_segment
        pred_clean = pred_text
        all_question_results, _ = self._process_alignments([alignment], pred_clean)
        
        if not all_question_results:
            logger.debug(f"题目 {question_id} 第二轮对齐失败：无法处理对齐结果")
            return None
        
        result_dict = all_question_results[0]
        
        # 验证question_id是否匹配
        if result_dict.get("question_id") != question_id:
            logger.warning(f"题目 {question_id} 第二轮对齐结果中的question_id不匹配: {result_dict.get('question_id')}")
            # 强制设置为正确的question_id
            result_dict["question_id"] = question_id
        
        return result_dict
    
    async def align_single_question_async(self, question_id: str, gt_text: str, pred_text: str) -> Optional[Dict]:
        """
        对单个题目进行对齐（用于第二轮匹配）
        
        :param question_id: 题目编号
        :param gt_text: 完整的GT文本（用于提取单个题目）
        :param pred_text: Pred文本
        :return: 对齐结果字典，如果失败返回None
        """
        # 从GT中提取单个题目的内容
        gt_question_content = self._extract_single_question_from_gt(gt_text, question_id)
        if not gt_question_content:
            logger.warning(f"无法从GT中提取题目 {question_id} 的内容")
            return None
        
        # 创建临时ctx用于调用LLM
        from core.schema import ProcessContext
        temp_ctx = ProcessContext(
            filename="temp_single",
            gt_raw="",
            pred_raw=""
        )
        temp_ctx.gt_clean = gt_question_content
        temp_ctx.pred_clean = pred_text
        
        # 调用LLM进行对齐
        result = await self.client.ainvoke(self.chain, {
            "gt_content": gt_question_content,
            "pred_content": pred_text
        })
        
        # 处理对齐结果
        alignments = self._process_alignment_result(result, temp_ctx)
        
        if not alignments:
            logger.debug(f"题目 {question_id} 第二轮对齐失败：LLM未返回对齐结果")
            return None
        
        # 只返回第一个对齐结果（应该只有一个）
        alignment = alignments[0]
        
        # 处理对齐结果，提取pred_segment
        pred_clean = pred_text
        all_question_results, _ = self._process_alignments([alignment], pred_clean)
        
        if not all_question_results:
            logger.debug(f"题目 {question_id} 第二轮对齐失败：无法处理对齐结果")
            return None
        
        result_dict = all_question_results[0]
        
        # 验证question_id是否匹配
        if result_dict.get("question_id") != question_id:
            logger.warning(f"题目 {question_id} 第二轮对齐结果中的question_id不匹配: {result_dict.get('question_id')}")
            # 强制设置为正确的question_id
            result_dict["question_id"] = question_id
        
        return result_dict
    
    @staticmethod
    def slice_questions(text: str, alignments: List[AlignmentSnippet]) -> dict:
        """
        根据 LLM 返回的锚点，将 Pred 文本切分成单个题目的字典。
        """
        sliced_questions = {}
        for snippet in alignments:
            start_anchor = snippet.pred_start_snippet
            end_anchor = snippet.pred_end_snippet
            
            start_idx = text.find(start_anchor)
            end_idx = text.find(end_anchor)
            
            if start_idx != -1 and end_idx != -1:
                # 确保结尾在开头之后
                if start_idx < end_idx + len(end_anchor):
                    sliced_questions[snippet.question_id] = text[start_idx : end_idx + len(end_anchor)]
        return sliced_questions
