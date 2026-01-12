from config import DASHSCOPE_API_KEY, BASE_URL, VISION_MODEL, TEXT_MODEL, IMAGE_RESOURCE_DIR
from utils.llm_client import LLMClient
import logging
from pathlib import Path
from core.schema import ProcessContext
from core.preprocessor import Preprocessor
from core.analyzer import GTAnalyzer
from core.aligner import Aligner
from core.evaluator import Evaluator
logger = logging.getLogger(__name__)

class EvaluationPipeline:
    def __init__(self):
        # =========================================================
        # 1. 实例化两个独立的 LLM Client
        # =========================================================
        
        # 文本处理 Client (qwen3-max)
        self.text_client = LLMClient(
            api_key=DASHSCOPE_API_KEY,
            model_name=TEXT_MODEL,
            base_url=BASE_URL
        )
        
        # 视觉处理 Client (qwen3-vl-plus)
        self.vision_client = LLMClient(
            api_key=DASHSCOPE_API_KEY,
            model_name=VISION_MODEL,
            base_url=BASE_URL
        )
    
           # =========================================================
        # 2. 依赖注入：将 Client 传给需要的子模块
        # =========================================================
        logger.info("Initializing pipeline components...")
        # 预处理器不依赖 LLM
        # GT分析器 和 对齐器 使用文本模型
        self.analyzer = GTAnalyzer()  # analyzer 不需要 LLM client
        self.aligner = Aligner(llm_client=self.text_client)
        
        # 使用指定的图片资源目录
        self.preprocessor = Preprocessor(image_resource_dir=IMAGE_RESOURCE_DIR)
        print(f"[DEBUG] IMAGE_RESOURCE_DIR = {IMAGE_RESOURCE_DIR}")
        # 评估器使用视觉模型来算图片相似度
        self.evaluator = Evaluator(vision_client=self.vision_client)
        
        logger.info("Evaluation pipeline initialized successfully.")

    async def run_single_case_async(self, filename: str, gt_raw: str, pred_raw: str, gt_file_path: Path, pred_file_path: Path) -> ProcessContext:
        """
        异步版本：对单个 GT 和 Pred 样本执行完整的五步评估法。
        """
        logger.info(f"--- Starting async evaluation for: {filename} ---")
        logger.debug(f"Initial gt_file_path: {gt_file_path}")
        logger.debug(f"IMAGE_RESOURCE_DIR is: {IMAGE_RESOURCE_DIR}")
        # 创建贯穿始终的上下文对象
        ctx = ProcessContext(filename=filename, gt_raw=gt_raw, pred_raw=pred_raw)
        # 1. 从 GT 文件名推断其在原始资源目录中的父文件夹名    
        if gt_file_path and gt_file_path.exists():
            gt_file_dir = gt_file_path.parent
        else:
            gt_file_dir = IMAGE_RESOURCE_DIR / gt_file_path.stem
            
        pred_file_dir = pred_file_path.parent 
        # 预处理（同步，不涉及API调用）
        ctx = self.preprocessor.process(ctx, gt_file_dir, pred_file_dir)
        logger.info("[Step 1/5] Preprocessing complete. Images extracted and text normalized.")

        # === Step 2: GT 题型锚定 ===
        gt_structure_map = self.analyzer.analyze(ctx.gt_raw)
        ctx.gt_structure_map = gt_structure_map
        
        if gt_structure_map:
            sorted_question_nums = sorted(gt_structure_map.keys(), key=lambda x: int(x) if x.isdigit() else 0)
            first_question_num = sorted_question_nums[0]
            ctx.gt_type = gt_structure_map[first_question_num]
            
            if first_question_num in ctx.gt_extracted_answers:
                ctx.gt_answer_truth = ctx.gt_extracted_answers[first_question_num]
            else:
                ctx.gt_answer_truth = None
        else:
            ctx.gt_type = None
            ctx.gt_answer_truth = None
        
        logger.info(f"[Step 2/5] GT analysis complete. Type: {ctx.gt_type.value if ctx.gt_type else None}, Truth: '{ctx.gt_answer_truth}'")

        # === Step 3: 对齐与定位（异步）===
        gt_alignment_text = ctx.gt_clean_for_alignment if ctx.gt_clean_for_alignment else ctx.gt_clean
        pred_alignment_text = ctx.pred_clean_for_alignment if ctx.pred_clean_for_alignment else ctx.pred_clean
        
        if not gt_alignment_text or not pred_alignment_text:
            logger.warning(f"[Step 3/5] 对齐文本为空: GT长度={len(gt_alignment_text) if gt_alignment_text else 0}, Pred长度={len(pred_alignment_text) if pred_alignment_text else 0}")
            ctx.alignment_found = False
            ctx.pred_is_rejected = True
            all_question_alignments = []
            ctx.alignment_stats = {
                "total_aligned": 0,
                "exact_match_count": 0,
                "fuzzy_match_count": 0,
                "fallback_start_count": 0,
                "fallback_next_question_count": 0,
                "fallback_text_end_count": 0,
                "total_questions": 0
            }
        else:
            logger.debug(f"[Step 3/5] 开始异步对齐: GT长度={len(gt_alignment_text)}, Pred长度={len(pred_alignment_text)}")
            all_question_alignments, alignment_stats = await self.aligner.align_async(gt_alignment_text, pred_alignment_text)
            
            ctx.alignment_stats = alignment_stats
            
            if not all_question_alignments:
                ctx.alignment_found = False
                ctx.pred_is_rejected = True
                logger.warning(f"[Step 3/5] No alignments found for any question.")
            else:
                ctx.all_question_segments = all_question_alignments
                
                # 识别第一轮对齐失败的题目（GT中存在但未对齐的）
                failed_question_ids = []
                if ctx.gt_structure_map:
                    # 获取第一轮对齐成功的题目ID集合
                    aligned_question_ids = {q.get("question_id") for q in all_question_alignments if q.get("alignment_found", False)}
                    
                    # 找出GT中存在但第一轮未对齐的题目
                    for question_id in ctx.gt_structure_map.keys():
                        if question_id not in aligned_question_ids:
                            failed_question_ids.append(question_id)
                    
                    if failed_question_ids:
                        logger.info(f"[Step 3/5] 发现 {len(failed_question_ids)} 道题目在第一轮对齐中失败，开始第二轮并行匹配: {failed_question_ids}")
                        
                        # 对失败的题目进行第二轮并行匹配（异步）
                        import asyncio
                        async def match_single_question(question_id: str):
                            try:
                                result = await self.aligner.align_single_question_async(
                                    question_id=question_id,
                                    gt_text=gt_alignment_text,
                                    pred_text=pred_alignment_text
                                )
                                if result and result.get("alignment_found", False):
                                    logger.info(f"[Step 3/5] 题目 {question_id} 第二轮匹配成功")
                                    return result
                                else:
                                    logger.debug(f"[Step 3/5] 题目 {question_id} 第二轮匹配失败")
                                    return None
                            except Exception as e:
                                logger.warning(f"[Step 3/5] 题目 {question_id} 第二轮匹配时发生异常: {e}")
                                return None
                        
                        # 并行执行所有失败题目的匹配
                        tasks = [match_single_question(q_id) for q_id in failed_question_ids]
                        second_round_results_raw = await asyncio.gather(*tasks)
                        # 过滤掉None结果
                        second_round_results = [r for r in second_round_results_raw if r is not None]
                        
                        # 将第二轮匹配成功的结果合并到all_question_segments中
                        if second_round_results:
                            logger.info(f"[Step 3/5] 第二轮匹配成功 {len(second_round_results)} 道题目")
                            # 合并结果，按question_id排序
                            all_question_segments_dict = {q.get("question_id"): q for q in ctx.all_question_segments}
                            for result in second_round_results:
                                q_id = result.get("question_id")
                                all_question_segments_dict[q_id] = result
                            
                            # 按题号排序，重新构建列表
                            sorted_question_ids = sorted(all_question_segments_dict.keys(), key=lambda x: int(x) if x.isdigit() else 0)
                            ctx.all_question_segments = [all_question_segments_dict[q_id] for q_id in sorted_question_ids]
                            
                            # 更新all_question_alignments引用（如果存在）
                            all_question_alignments = ctx.all_question_segments
                        else:
                            logger.info(f"[Step 3/5] 第二轮匹配未成功匹配任何题目")
                
                # 从对齐结果中提取答案
                for question in all_question_alignments:
                    question_id = question.get("question_id")
                    gt_answer = question.get("gt_answer", "").strip()
                    pred_answer = question.get("pred_answer", "").strip()
                    
                    if gt_answer:
                        ctx.gt_extracted_answers[question_id] = gt_answer
                    
                    if pred_answer:
                        ctx.pred_extracted_answers[question_id] = pred_answer
                        question["pred_answer"] = pred_answer
                    elif question_id in ctx.pred_extracted_answers:
                        pred_answer_from_preprocessor = ctx.pred_extracted_answers[question_id]
                        question["pred_answer"] = pred_answer_from_preprocessor
                
                # 保存第一个题目的信息
                first_question = all_question_alignments[0]
                ctx.alignment_found = first_question.get("alignment_found", False)
                ctx.pred_segment = first_question.get("pred_segment")
                ctx.pred_is_rejected = first_question.get("is_rejected", True)
                ctx.pred_raw_handwritten = first_question.get("raw_handwritten_answer")
                
                if ctx.alignment_found:
                    ctx.alignment_anchors = {
                        "start_anchor": first_question.get("start_anchor", ""),
                        "end_anchor": first_question.get("end_anchor", ""),
                        "gt_start_snippet": first_question.get("gt_start_snippet", ""),
                        "gt_end_snippet": first_question.get("gt_end_snippet", ""),
                    }
                    logger.info(f"[Step 3/5] Alignment successful. Found {len(all_question_alignments)} question(s).")
                else:
                    ctx.pred_is_rejected = True
                    logger.warning(f"[Step 3/5] First question alignment failed, but found {len(all_question_alignments)} question(s) total.")

        # === Step 4 & 5: 评估（异步）===
        ctx = await self.evaluator.evaluate_async(ctx)
        logger.info(f"[Step 4-5/5] Evaluation complete. Metrics calculated: {ctx.metrics}")
        
        return ctx
    
    def run_single_case(self, filename: str, gt_raw: str, pred_raw: str, gt_file_path: Path, pred_file_path: Path) -> ProcessContext:
        """
        对单个 GT 和 Pred 样本执行完整的五步评估法。
        
        :param filename: 文件名
        :param gt_raw: GT 原始文本
        :param pred_raw: Pred 原始文本
        :param gt_file_path: GT 文件路径（用于确定图片基准目录）
        :param pred_file_path: Pred 文件路径（用于确定图片基准目录）
        """
        logger.info(f"--- Starting evaluation for: {filename} ---")
        logger.debug(f"Initial gt_file_path: {gt_file_path}")
        logger.debug(f"IMAGE_RESOURCE_DIR is: {IMAGE_RESOURCE_DIR}")
        # 创建贯穿始终的上下文对象
        ctx = ProcessContext(filename=filename, gt_raw=gt_raw, pred_raw=pred_raw)
        # 1. 从 GT 文件名推断其在原始资源目录中的父文件夹名    
        if gt_file_path and gt_file_path.exists():
            gt_file_dir = gt_file_path.parent
        else:
            # 兜底逻辑：如果传入的路径不存在（极少见），才回退到资源目录逻辑
            gt_file_dir = IMAGE_RESOURCE_DIR / gt_file_path.stem
            
        pred_file_dir = pred_file_path.parent 
        # 注意：preprocessor.process() 需要传入目录路径，不是文件路径
        ctx = self.preprocessor.process(ctx, gt_file_dir, pred_file_dir)
        
        # === Step 1: 全局预处理 ===
        # 使用文件所在目录作为图片路径的基准目录
        #gt_file_dir = gt_file_path.parent if gt_file_path.is_file() else gt_file_path
        #pred_file_dir = pred_file_path.parent if pred_file_path.is_file() else pred_file_path
        #ctx = self.preprocessor.process(ctx, gt_file_dir, pred_file_dir)
        logger.info("[Step 1/5] Preprocessing complete. Images extracted and text normalized.")

        # === Step 2: GT 题型锚定 ===
        # analyzer.analyze() 返回 {题号: 题型} 的字典
        # 注意：必须使用原始文本 gt_raw，因为 gt_clean 已经过滤掉了大章节标题
        gt_structure_map = self.analyzer.analyze(ctx.gt_raw)
        # 保存完整的题号-题型映射，用于统计题目数
        ctx.gt_structure_map = gt_structure_map
        
        # 确定主要题型：按题号顺序取第一个题目的题型
        if gt_structure_map:
            # 按题号排序，取第一个题目的题型
            sorted_question_nums = sorted(gt_structure_map.keys(), key=lambda x: int(x) if x.isdigit() else 0)
            first_question_num = sorted_question_nums[0]
            ctx.gt_type = gt_structure_map[first_question_num]
            
            # 从提取的答案中获取第一个题目的答案
            if first_question_num in ctx.gt_extracted_answers:
                ctx.gt_answer_truth = ctx.gt_extracted_answers[first_question_num]
            else:
                ctx.gt_answer_truth = None
        else:
            ctx.gt_type = None
            ctx.gt_answer_truth = None
        
        logger.info(f"[Step 2/5] GT analysis complete. Type: {ctx.gt_type.value if ctx.gt_type else None}, Truth: '{ctx.gt_answer_truth}'")

        # === Step 3: 对齐与定位 ===
        # 使用无图、无大章节标题、无答案标签的纯净文本进行对齐和截取，处理所有题目
        gt_alignment_text = ctx.gt_clean_for_alignment if ctx.gt_clean_for_alignment else ctx.gt_clean
        pred_alignment_text = ctx.pred_clean_for_alignment if ctx.pred_clean_for_alignment else ctx.pred_clean
        
        # 检查对齐文本是否为空
        if not gt_alignment_text or not pred_alignment_text:
            logger.warning(f"[Step 3/5] 对齐文本为空: GT长度={len(gt_alignment_text) if gt_alignment_text else 0}, Pred长度={len(pred_alignment_text) if pred_alignment_text else 0}")
            ctx.alignment_found = False
            ctx.pred_is_rejected = True
            all_question_alignments = []
            # 设置统计信息：total_questions应该使用gt_structure_map的题目数，而不是0
            total_questions_from_gt = len(ctx.gt_structure_map) if ctx.gt_structure_map else 0
            ctx.alignment_stats = {
                "total_aligned": 0,
                "exact_match_count": 0,
                "fuzzy_match_count": 0,
                "fallback_next_question_count": 0,
                "fallback_text_end_count": 0,
                "total_questions": total_questions_from_gt  # 使用GT中的题目数，而不是0
            }
        else:
            logger.debug(f"[Step 3/5] 开始对齐: GT长度={len(gt_alignment_text)}, Pred长度={len(pred_alignment_text)}")
            all_question_alignments, alignment_stats = self.aligner.align(gt_alignment_text, pred_alignment_text)
            
            # 保存对齐统计信息到 ProcessContext
            ctx.alignment_stats = alignment_stats
            
            if not all_question_alignments:
                # 没有找到任何对齐
                ctx.alignment_found = False
                ctx.pred_is_rejected = True
                logger.warning(f"[Step 3/5] No alignments found for any question. GT前100字符: {gt_alignment_text[:100]}, Pred前100字符: {pred_alignment_text[:100]}")
                logger.warning(f"[Step 3/5] GT文本长度: {len(gt_alignment_text)}, Pred文本长度: {len(pred_alignment_text)}")
                # 更新alignment_stats：total_questions应该使用gt_structure_map的题目数，而不是LLM返回的0
                total_questions_from_gt = len(ctx.gt_structure_map) if ctx.gt_structure_map else 0
                if ctx.alignment_stats:
                    ctx.alignment_stats["total_questions"] = total_questions_from_gt
                # 尝试输出GT和Pred中的题号信息，帮助诊断
                import re
                gt_question_numbers = re.findall(r'^\s*(\d+)[\.。、]', gt_alignment_text, re.MULTILINE)
                pred_question_numbers = re.findall(r'^\s*(\d+)[\.。、]', pred_alignment_text, re.MULTILINE)
                logger.warning(f"[Step 3/5] GT中找到的题号: {gt_question_numbers[:10]} (共{len(gt_question_numbers)}个)")
                logger.warning(f"[Step 3/5] Pred中找到的题号: {pred_question_numbers[:10]} (共{len(pred_question_numbers)}个)")
                logger.warning(f"[Step 3/5] gt_structure_map中的题目数: {total_questions_from_gt}")
                # 尝试查找GT中第一个题目的开头文本，看看是否能在Pred中找到
                if gt_question_numbers:
                    first_q_num = gt_question_numbers[0]
                    # 查找GT中第一个题目的内容（题号后的文本）
                    first_q_pattern = re.compile(rf'^\s*{re.escape(first_q_num)}[\.。、]\s*(.+?)(?=\n\s*\d+[\.。、]|\n\s*[一二三四五六七八九十]+[、.]|$)', re.MULTILINE | re.DOTALL)
                    first_q_match = first_q_pattern.search(gt_alignment_text)
                    if first_q_match:
                        first_q_content = first_q_match.group(1).strip()[:50]  # 前50个字符
                        logger.warning(f"[Step 3/5] GT中第{first_q_num}题内容前50字符: {first_q_content}")
                        # 检查是否在Pred中找到相似内容
                        if first_q_content in pred_alignment_text:
                            logger.warning(f"[Step 3/5] 在Pred中找到了GT第{first_q_num}题的内容，但LLM未能对齐")
                        else:
                            logger.warning(f"[Step 3/5] 在Pred中未找到GT第{first_q_num}题的内容")
            else:
                # 保存所有题目的对齐结果
                ctx.all_question_segments = all_question_alignments
                
                # 更新alignment_stats：total_questions应该使用gt_structure_map的题目数，而不是LLM返回的题目数
                # 这样可以确保对齐统计覆盖所有GT题目，包括LLM没有识别到的题目
                total_questions_from_gt = len(ctx.gt_structure_map) if ctx.gt_structure_map else 0
                if ctx.alignment_stats:
                    # 记录LLM返回的题目数（用于诊断）
                    llm_recognized_count = ctx.alignment_stats.get("total_questions", 0)
                    # 更新为GT中的实际题目数
                    ctx.alignment_stats["total_questions"] = total_questions_from_gt
                    # 如果LLM没有识别到所有题目，记录警告
                    if llm_recognized_count < total_questions_from_gt:
                        missing_count = total_questions_from_gt - llm_recognized_count
                        logger.warning(
                            f"[Step 3/5] LLM只识别到 {llm_recognized_count} 道题目，"
                            f"但GT中有 {total_questions_from_gt} 道题目，缺失 {missing_count} 道题目。"
                            f"这些题目将被视为未对齐（alignment_found=False）。"
                        )
                
                # 识别第一轮对齐失败的题目（GT中存在但未对齐的）
                failed_question_ids = []
                if ctx.gt_structure_map:
                    # 获取第一轮对齐成功的题目ID集合
                    aligned_question_ids = {q.get("question_id") for q in all_question_alignments if q.get("alignment_found", False)}
                    
                    # 找出GT中存在但第一轮未对齐的题目
                    for question_id in ctx.gt_structure_map.keys():
                        if question_id not in aligned_question_ids:
                            failed_question_ids.append(question_id)
                    
                    if failed_question_ids:
                        logger.info(f"[Step 3/5] 发现 {len(failed_question_ids)} 道题目在第一轮对齐中失败，开始第二轮并行匹配: {failed_question_ids}")
                        
                        # 对失败的题目进行第二轮并行匹配（使用线程池）
                        import concurrent.futures
                        def match_single_question(question_id: str):
                            try:
                                result = self.aligner.align_single_question(
                                    question_id=question_id,
                                    gt_text=gt_alignment_text,
                                    pred_text=pred_alignment_text
                                )
                                if result and result.get("alignment_found", False):
                                    logger.info(f"[Step 3/5] 题目 {question_id} 第二轮匹配成功")
                                    return result
                                else:
                                    logger.debug(f"[Step 3/5] 题目 {question_id} 第二轮匹配失败")
                                    return None
                            except Exception as e:
                                logger.warning(f"[Step 3/5] 题目 {question_id} 第二轮匹配时发生异常: {e}")
                                return None
                        
                        # 使用线程池并行执行（最多5个并发，避免过多并发请求）
                        max_workers = min(5, len(failed_question_ids))
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = {executor.submit(match_single_question, q_id): q_id for q_id in failed_question_ids}
                            second_round_results = []
                            for future in concurrent.futures.as_completed(futures):
                                result = future.result()
                                if result is not None:
                                    second_round_results.append(result)
                        
                        # 将第二轮匹配成功的结果合并到all_question_segments中
                        if second_round_results:
                            logger.info(f"[Step 3/5] 第二轮匹配成功 {len(second_round_results)} 道题目")
                            # 合并结果，按question_id排序
                            all_question_segments_dict = {q.get("question_id"): q for q in ctx.all_question_segments}
                            for result in second_round_results:
                                q_id = result.get("question_id")
                                all_question_segments_dict[q_id] = result
                            
                            # 按题号排序，重新构建列表
                            sorted_question_ids = sorted(all_question_segments_dict.keys(), key=lambda x: int(x) if x.isdigit() else 0)
                            ctx.all_question_segments = [all_question_segments_dict[q_id] for q_id in sorted_question_ids]
                            
                            # 更新all_question_alignments引用（如果存在）
                            all_question_alignments = ctx.all_question_segments
                        else:
                            logger.info(f"[Step 3/5] 第二轮匹配未成功匹配任何题目")
                
                # 从对齐结果中提取答案，更新 gt_extracted_answers 和 pred_extracted_answers
                # 优先级：aligner (LLM) 提取的答案 > preprocessor (正则) 提取的答案
                # 原因：LLM 可以识别跨行答案，理解上下文，更准确
                for question in all_question_alignments:
                    question_id = question.get("question_id")
                    gt_answer = question.get("gt_answer", "").strip()
                    pred_answer = question.get("pred_answer", "").strip()
                    
                    # 如果对齐环节（LLM）提取到了答案，更新到 extracted_answers 中
                    # 注意：LLM 的答案会覆盖 preprocessor 的答案（优先级更高）
                    if gt_answer:
                        ctx.gt_extracted_answers[question_id] = gt_answer
                    
                    # 处理 pred_answer：优先使用 aligner (LLM) 的答案
                    if pred_answer:
                        # 如果对齐环节（LLM）提取到了答案，更新到 extracted_answers 中
                        # LLM 可以识别跨行答案，因此优先级高于 preprocessor 的正则提取
                        ctx.pred_extracted_answers[question_id] = pred_answer
                        # 同时更新 all_question_alignments 中的 pred_answer，确保数据一致
                        question["pred_answer"] = pred_answer
                    elif question_id in ctx.pred_extracted_answers:
                        # 如果对齐环节（LLM）没有提取到答案，使用 preprocessor 的答案作为后备
                        pred_answer_from_preprocessor = ctx.pred_extracted_answers[question_id]
                        question["pred_answer"] = pred_answer_from_preprocessor
                        # 确保 ctx.all_question_segments 也被更新（因为它们指向同一个列表）
                        # 由于 all_question_alignments 和 ctx.all_question_segments 指向同一个列表，更新会自动反映
                
                # 为了向后兼容，仍然保存第一个题目的信息（用于评估）
                first_question = all_question_alignments[0]
                ctx.alignment_found = first_question.get("alignment_found", False)
                ctx.pred_segment = first_question.get("pred_segment")
                ctx.pred_is_rejected = first_question.get("is_rejected", True)
                ctx.pred_raw_handwritten = first_question.get("raw_handwritten_answer")
                
                # 保存第一个题目的对齐锚点信息（用于详细报告）
                if ctx.alignment_found:
                    ctx.alignment_anchors = {
                        "start_anchor": first_question.get("start_anchor", ""),
                        "end_anchor": first_question.get("end_anchor", ""),
                        "gt_start_snippet": first_question.get("gt_start_snippet", ""),
                        "gt_end_snippet": first_question.get("gt_end_snippet", ""),
                    }
                    logger.info(f"[Step 3/5] Alignment successful. Found {len(all_question_alignments)} question(s).")
                else:
                    ctx.pred_is_rejected = True
                    logger.warning(f"[Step 3/5] First question alignment failed, but found {len(all_question_alignments)} question(s) total.")

        # === Step 4 & 5: 拒答判断、组件评估、骨架剥离与内容相似度 ===
        # 将包含所有信息的上下文对象交给 Evaluator 进行最终打分
        ctx = self.evaluator.evaluate(ctx)
        logger.info(f"[Step 4-5/5] Evaluation complete. Metrics calculated: {ctx.metrics}")
        
        return ctx

    def _slice_segment(self, text: str, start_anchor: str, end_anchor: str) -> str:
        """一个私有的辅助方法，用于根据锚点截取字符串"""
        if not text or not start_anchor or not end_anchor:
            return ""
        
        start_idx = text.find(start_anchor)
        end_idx = text.find(end_anchor)
        
        if start_idx == -1 or end_idx == -1:
            logger.warning(f"Could not find anchors in text. Start: '{start_anchor}', End: '{end_anchor}'")
            return ""
            
        # 确保结尾在开头之后
        if start_idx >= end_idx + len(end_anchor):
            logger.warning("End anchor appears before start anchor.")
            return ""
            
        return text[start_idx : end_idx + len(end_anchor)]
    