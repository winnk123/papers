# 预处理数据，清洗掉无关内容
# 1. 公式转换成unicode形式，图片链接转化到绝对路径的格式
# 2. 去除图片链接，将其保存到一个存储空间中
# 3. 提取题目中的手写答案（对于ocr中的pred结果，用llm进行识别；对于llm的pred结果，直接提取）

import os
import re
import unicodedata
from pathlib import Path

# 假设 schema.py 在同一 core 目录下
from .schema import ProcessContext
class Preprocessor:
    """
    第一步：全局预处理
    负责：
    1. 过滤试卷元数据（标题、注意事项等）。
    2. 将 LaTeX 公式归一化为 Unicode。
    3. 提取并分离图片链接。
    """
    def __init__(self,image_resource_dir: Path = None):
        self.answer_pattern = re.compile(
            r"[【\[(]\s*答案\s*[:：]\s*([^】\])]+)[】\])]"
        )
        # 匹配题号，用于关联答案
        self.problem_number_pattern = re.compile(r"^\s*(\d+)\.")
        # 将你提供的 cleaner 作为一个内部实例
        self.latex_cleaner = self._MarkdownLatexCleaner()
        self.image_resource_dir = image_resource_dir

    def process(self, ctx: ProcessContext, gt_file_dir: Path, pred_file_dir: Path) -> ProcessContext:
        """
        处理单个上下文对象
        
        :param ctx: ProcessContext 对象
        :param gt_file_dir: GT 文件所在目录（用于解析 GT 中的相对图片路径）
        :param pred_file_dir: Pred 文件所在目录（用于解析 Pred 中的相对图片路径）
        """
        # --- 1. GT 文本处理 ---
        # 过滤元数据
        gt_filtered = self.latex_cleaner._filter_exam_metadata(ctx.gt_raw)
        # 归一化 LaTeX 并分离图片（使用 GT 文件所在目录作为基准）
        # 返回：完整文本（含图片）、图片路径列表、无图文本（用于对齐）
        gt_full, ctx.gt_img_paths, gt_no_images = self._normalize_and_extract_images(
            gt_filtered, gt_file_dir,image_resource_dir=None
        )
        # 在归一化后提取答案（此时文本已经是归一化后的格式，答案格式一致）
        # 从完整文本中提取答案（用于报告）
        gt_without_answers, ctx.gt_extracted_answers = self._extract_and_strip_answers(gt_full)
        # 从无图文本中提取答案（用于对齐，确保答案也被移除）
        gt_no_images_without_answers, _ = self._extract_and_strip_answers(gt_no_images)
        ctx.gt_clean = gt_without_answers  # 保存完整文本（已移除答案标签，用于报告）
        ctx.gt_clean_for_alignment = gt_no_images_without_answers  # 保存无图文本（已移除答案标签，用于对齐）
        
        # --- 2. Pred 文本处理 ---
        pred_filtered = self.latex_cleaner._filter_exam_metadata(ctx.pred_raw)
        # 归一化 LaTeX 并分离图片（使用 Pred 文件所在目录作为基准）
        # 返回：完整文本（含图片）、图片路径列表、无图文本（用于对齐）
        pred_full, ctx.pred_img_paths, pred_no_images = self._normalize_and_extract_images(
            pred_filtered, pred_file_dir,image_resource_dir=self.image_resource_dir
        )
        # 去除大量重复内容（避免影响LLM对齐）
        pred_full_before_dedup = len(pred_full.splitlines())
        pred_full = self._remove_duplicate_content(pred_full)
        pred_full_after_dedup = len(pred_full.splitlines())
        
        pred_no_images_before_dedup = len(pred_no_images.splitlines())
        pred_no_images = self._remove_duplicate_content(pred_no_images)
        pred_no_images_after_dedup = len(pred_no_images.splitlines())
        
        if pred_full_before_dedup != pred_full_after_dedup:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"去重生效: pred_full 从 {pred_full_before_dedup} 行减少到 {pred_full_after_dedup} 行 "
                       f"(减少 {pred_full_before_dedup - pred_full_after_dedup} 行, "
                       f"{(1 - pred_full_after_dedup/pred_full_before_dedup)*100:.1f}%)")
        if pred_no_images_before_dedup != pred_no_images_after_dedup:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"去重生效: pred_no_images 从 {pred_no_images_before_dedup} 行减少到 {pred_no_images_after_dedup} 行 "
                       f"(减少 {pred_no_images_before_dedup - pred_no_images_after_dedup} 行, "
                       f"{(1 - pred_no_images_after_dedup/pred_no_images_before_dedup)*100:.1f}%)")
        # 在归一化后提取答案（此时文本已经是归一化后的格式，答案格式一致）
        # 从完整文本中提取答案（用于报告）
        pred_without_answers, ctx.pred_extracted_answers = self._extract_and_strip_answers(pred_full)
        # 从无图文本中提取答案（用于对齐，确保答案也被移除）
        pred_no_images_without_answers, _ = self._extract_and_strip_answers(pred_no_images)
        ctx.pred_clean = pred_without_answers  # 保存完整文本（已移除答案标签，用于报告）
        ctx.pred_clean_for_alignment = pred_no_images_without_answers  # 保存无图文本（已移除答案标签，用于对齐）
        
        return ctx
    
    
    def _extract_and_strip_answers(self, text: str) -> tuple[str, dict[str, str]]:
        """
        核心函数：
        1. 逐行扫描文本。
        2. 识别题号。
        3. 在题号对应的题目块内查找答案标签。
        4. 提取答案，并从文本中移除标签。
        
        返回: (移除了答案标签的文本, {题号: 答案} 字典)
        """
        lines = text.split('\n')
        processed_lines = []
        extracted_answers = {}
        
        # 预编译清理模式（提高效率）
        fill_blank_pattern = re.compile(r'_+([^_\s]+[^_]*?[^_\s]+)_+')  # 有内容的下划线
        empty_fill_pattern = re.compile(r'_+')  # 空下划线
        leftover_marker_pattern = re.compile(r'[】\]]\s*')  # 残留标记
        
        current_problem_num = None
        
        for line in lines:
            # 检查是否是新题目的开始
            num_match = self.problem_number_pattern.match(line)
            if num_match:
                current_problem_num = num_match.group(1)
            
            # 查找答案标签
            answer_match = self.answer_pattern.search(line)
            
            if answer_match and current_problem_num:
                # 提取答案
                answer_content = answer_match.group(1).strip()
                extracted_answers[current_problem_num] = answer_content
                
                # 从行中移除答案标签
                line_cleaned = self.answer_pattern.sub("", line)
            else:
                line_cleaned = line
            
            # 对所有行进行清理（无论是否找到答案标签）
            # 1. 移除下划线占位符（包括空下划线，如 ______）
            # 先移除有内容的下划线（如 __答案__）
            line_cleaned = fill_blank_pattern.sub("", line_cleaned)
            # 再移除空下划线（如 ______）
            line_cleaned = empty_fill_pattern.sub("", line_cleaned)
            
            # 2. 移除残留的标记（如 】、] 等）
            line_cleaned = leftover_marker_pattern.sub("", line_cleaned)
            
            # 3. 清理多余的空格
            line_cleaned = re.sub(r'\s+', ' ', line_cleaned).strip()
            
            processed_lines.append(line_cleaned)

        # 重新组合文本
        text_without_answers = "\n".join(processed_lines)
        return text_without_answers, extracted_answers

    def _remove_duplicate_content(self, text: str) -> str:
        """
        检测并去除大量重复的内容行和行内重复短语
        
        策略（更激进）：
        1. 先处理行内重复短语（同一行内重复的短语，如"测得正方向地面上的B、C两点与楼底在同一水平面上"）
        2. 按行分割文本
        3. 统计每行出现的次数
        4. 根据行长度采用不同的阈值：
           - 短行（<10字符）：重复2次以上就去除
           - 中等行（10-30字符）：重复3次以上就去除
           - 较长行（30-100字符）：重复5次以上就去除
           - 长行（>100字符）：重复10次以上才去除
        5. 对于连续重复的行块，只保留一次
        6. 去除多余的空行（连续3个以上空行只保留2个）
        
        :param text: 输入文本
        :return: 去除重复后的文本
        """
        if not text:
            return text
        
        # 先处理行内重复短语
        text = self._remove_inline_duplicate_phrases(text)
        
        lines = text.split('\n')
        if len(lines) < 20:  # 如果行数很少，不需要去重
            return text
        
        # 统计每行出现的次数
        line_counts = {}
        for line in lines:
            stripped = line.strip()
            if stripped:  # 只统计非空行
                line_counts[stripped] = line_counts.get(stripped, 0) + 1
        
        # 根据行长度采用不同的阈值（更激进的策略）
        # 但要注意：不要删除可能包含题目内容的行
        duplicate_lines = {}
        for line, count in line_counts.items():
            line_len = len(line)
            # 检查是否是题目内容行（包含中文、题号、选项标记等）
            is_question_content = (
                re.search(r'[\u4e00-\u9fa5]', line) or  # 包含中文
                re.search(r'\d+[\.。、]', line) or  # 包含题号
                re.search(r'^[A-Z][\.。、]', line) or  # 选项标记
                re.search(r'[A-Z]\s*[\.。、]\s+', line)  # 选项标记（带空格）
            )
            
            # 如果是题目内容行，提高阈值，避免误删
            if is_question_content:
                # 题目内容行：需要重复更多次才删除
                if line_len < 10 and count >= 5:  # 短行：5次以上
                    duplicate_lines[line] = count
                elif line_len < 30 and count >= 10:  # 中等行：10次以上
                    duplicate_lines[line] = count
                elif line_len < 100 and count >= 20:  # 较长行：20次以上
                    duplicate_lines[line] = count
                elif count >= 50:  # 长行：50次以上
                    duplicate_lines[line] = count
            else:
                # 非题目内容行：使用原来的阈值
                if line_len < 10 and count >= 2:  # 短行：2次以上
                    duplicate_lines[line] = count
                elif line_len < 30 and count >= 3:  # 中等行：3次以上
                    duplicate_lines[line] = count
                elif line_len < 100 and count >= 5:  # 较长行：5次以上
                    duplicate_lines[line] = count
                elif count >= 10:  # 长行：10次以上
                    duplicate_lines[line] = count
        
        if not duplicate_lines:
            # 如果没有大量重复的行，检查是否有连续重复的行块
            result_text = self._remove_consecutive_duplicates(text)
        else:
            # 去除重复行，只保留第一次出现
            seen_lines = set()
            result_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped in duplicate_lines:
                    if stripped not in seen_lines:
                        # 第一次出现，保留
                        result_lines.append(line)
                        seen_lines.add(stripped)
                    # 后续出现，跳过
                else:
                    # 非重复行，保留
                    result_lines.append(line)
            
            result_text = '\n'.join(result_lines)
            
            # 再次检查连续重复的行块
            result_text = self._remove_consecutive_duplicates(result_text)
        
        # 去除多余的空行（连续3个以上空行只保留2个）
        result_text = self._remove_excessive_blank_lines(result_text)
        
        return result_text
    
    def _remove_inline_duplicate_phrases(self, text: str) -> str:
        """
        去除行内重复的短语
        
        例如：
        "测得正方向地面上的B、C两点与楼底在同一水平面上，测得正方向地面上的B、C两点与楼底在同一水平面上，..."
        会被清理为：
        "测得正方向地面上的B、C两点与楼底在同一水平面上，"
        
        策略：
        1. 检测行内连续重复的短语（通过逗号、句号等分隔符分隔）
        2. 如果同一个短语在同一行内重复3次以上，只保留一次
        3. 保留分隔符（逗号、句号等）
        
        :param text: 输入文本
        :return: 去除行内重复短语后的文本
        """
        if not text:
            return text
        
        lines = text.split('\n')
        result_lines = []
        
        for line in lines:
            if not line.strip():
                result_lines.append(line)
                continue
            
            # 检测行内重复短语
            # 使用正则表达式匹配可能重复的短语模式
            # 匹配模式：至少10个字符的短语，后面跟着逗号或句号，然后重复出现
            
            # 策略1：检测连续重复的短语（通过逗号分隔）
            # 匹配形如 "短语，短语，短语" 的模式
            # 使用反向引用来匹配重复的短语
            processed_line = line
            
            # 检测连续重复的短语（至少10个字符，重复3次以上）
            # 策略：使用更灵活的模式，匹配包含各种字符的短语
            # 匹配模式：短语+分隔符+重复的短语+分隔符+重复的短语...
            
            # 改进的模式：匹配至少10个字符的短语（可以包含中文、英文、数字、标点等）
            # 使用非贪婪匹配，避免匹配过长
            # 分隔符包括：中文逗号、句号、顿号，英文逗号、句号
            pattern = r'(.{10,}?)([，。、,\.])\s*(?:\1\2\s*){2,}'
            
            def replace_duplicate_phrase(match):
                phrase = match.group(1)
                separator = match.group(2)
                # 只保留一次短语和分隔符
                return phrase + separator
            
            # 多次应用，直到没有更多重复
            max_iterations = 10
            iteration = 0
            while iteration < max_iterations:
                new_line = re.sub(pattern, replace_duplicate_phrase, processed_line)
                if new_line == processed_line:
                    break
                processed_line = new_line
                iteration += 1
            
            result_lines.append(processed_line)
        
        return '\n'.join(result_lines)
    
    def _remove_excessive_blank_lines(self, text: str) -> str:
        """
        去除多余的空行
        
        :param text: 输入文本
        :return: 去除多余空行后的文本
        """
        lines = text.split('\n')
        result_lines = []
        blank_count = 0
        
        for line in lines:
            if not line.strip():  # 空行
                blank_count += 1
                if blank_count <= 2:  # 最多保留2个连续空行
                    result_lines.append(line)
            else:  # 非空行
                blank_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _remove_consecutive_duplicates(self, text: str) -> str:
        """
        去除连续重复的行块
        
        例如：
        $ 1<m<6 $
        $ 0<m<2 $
        $ 1<m<6 $
        $ 0<m<2 $
        ...
        
        如果这样的模式重复超过5次，只保留一次
        
        :param text: 输入文本
        :return: 去除连续重复后的文本
        """
        lines = text.split('\n')
        if len(lines) < 10:
            return text
        
        result_lines = []
        i = 0
        while i < len(lines):
            # 尝试找到重复模式
            # 检查从当前位置开始的2-5行是否形成重复模式
            pattern_found = False
            pattern_length = 0
            max_repeat_count = 0
            
            for pattern_len in range(2, min(6, len(lines) - i + 1)):  # 检查2-5行的模式
                if i + pattern_len * 3 > len(lines):  # 至少需要重复3次
                    continue
                
                # 检查这个模式是否重复
                pattern = [l.strip() for l in lines[i:i+pattern_len]]
                # 跳过空行模式
                if not any(pattern):
                    break
                
                is_repeating = True
                repeat_count = 1
                
                for j in range(1, 10):  # 检查是否至少重复3次（最多检查10次）
                    start_idx = i + pattern_len * j
                    if start_idx + pattern_len > len(lines):
                        is_repeating = False
                        break
                    
                    next_pattern = [l.strip() for l in lines[start_idx:start_idx+pattern_len]]
                    # 比较去除首尾空格后的内容
                    if pattern != next_pattern:
                        is_repeating = False
                        break
                    repeat_count += 1
                    
                    # 如果已经重复了足够多次，可以提前结束
                    if repeat_count >= 3:
                        break
                
                if is_repeating and repeat_count >= 3:
                    pattern_found = True
                    pattern_length = pattern_len
                    max_repeat_count = repeat_count
                    break
            
            if pattern_found:
                # 找到重复模式，只保留一次
                result_lines.extend(lines[i:i+pattern_length])
                # 跳过所有重复的部分
                i += pattern_length * max_repeat_count
            else:
                # 没有找到重复模式，保留当前行
                result_lines.append(lines[i])
                i += 1
        
        return '\n'.join(result_lines)

    def _normalize_and_extract_images(self, text: str, base_dir: Path, image_resource_dir: Path = None) -> tuple[str, list[str], str]:
        """
        核心函数：
        1. 提取图片链接并转为绝对路径。
        2. 将文本中的图片链接替换为特殊占位符。
        3. 对剩余的纯文本进行 LaTeX 归一化。
        4. 将占位符替换回处理过的图片链接。
        """
        image_pattern = r'(<img[^>]+>|!\[[^\]]*\]\([^)]+\))'
        matches = list(re.finditer(image_pattern, text))
        #print(f"[DEBUG _normalize_and_extract_images] base_dir = {base_dir}")
        #print(f"[DEBUG _normalize_and_extract_images] image_resource_dir = {image_resource_dir}")
        # --- a. 提取并转换图片链接 ---
        img_placeholders = {}
        img_paths = []
        
        # 先用占位符替换所有图片链接
        temp_text = text
        for i, match in enumerate(matches):
            placeholder = f"__IMG_PLACEHOLDER_{i}__"
            img_tag = match.group(0)
            
            # 转换路径
            abs_img_tag = self.latex_cleaner._convert_image_to_absolute(img_tag, base_dir, image_resource_dir)
            img_placeholders[placeholder] = abs_img_tag
            
            # 从文本中提取图片路径
            src_match = re.search(r'src=["\']([^"\']+)["\']', abs_img_tag) or re.search(r'!\[[^\]]*\]\(([^)]+)\)', abs_img_tag)
            if src_match:
                extracted_path = src_match.group(1)
                # 确保路径是字符串格式，并检查文件是否存在
                img_path_str = str(extracted_path)
                # 确保路径是绝对路径，如果不是，尝试从 base_dir 解析
                img_path = Path(img_path_str)
                if not img_path.is_absolute():
                    # 如果是相对路径，从 base_dir 解析
                    img_path = (base_dir / img_path_str).resolve()
                    img_path_str = str(img_path)
                elif not img_path.exists():
                    # 如果是绝对路径但不存在，尝试在 base_dir 下查找同名文件
                    filename = img_path.name
                    candidate_path = base_dir / filename
                    if candidate_path.exists():
                        img_path_str = str(candidate_path.resolve())
                img_paths.append(img_path_str)

            temp_text = temp_text.replace(img_tag, placeholder, 1)

        # --- b. 对无图文本进行 LaTeX 归一化 ---
        # 移除 $ 符号
        text_no_dollars = temp_text.replace('$$', ' ').replace('$', ' ')
        # 调用你提供的核心清理函数
        normalized_text = self.latex_cleaner.clean_math_content(text_no_dollars)
        
        # --- c. 将图片占位符替换回来（用于保存的文本）---
        # 但为了对齐，我们需要一个完全去除图片的版本
        final_text = normalized_text
        for placeholder, abs_tag in img_placeholders.items():
            final_text = final_text.replace(placeholder, abs_tag)
            
        # 创建一个完全去除图片的版本（用于对齐）
        text_without_images = normalized_text
        for placeholder in img_placeholders.keys():
            # 移除图片占位符，替换为空格
            text_without_images = text_without_images.replace(placeholder, ' ')
        # 清理多余空格（但保留换行符）
        # 只替换连续的空格/制表符为单个空格，不要替换换行符
        text_without_images = re.sub(r'[ \t]+', ' ', text_without_images)
        text_without_images = re.sub(r'\n\s*\n', '\n\n', text_without_images)  # 保留段落分隔
            
        return final_text.strip(), img_paths, text_without_images.strip()

    # ==============================================================================
    # 内部类：直接嵌入你提供的 MarkdownLatexCleaner 代码
    # 这样做的好处是 Preprocessor 作为一个独立的模块，不依赖外部文件
    # ==============================================================================
    class _MarkdownLatexCleaner:
        def __init__(self):
            # ... (这里直接粘贴你提供的 MarkdownLatexCleaner 的 __init__ 方法的所有内容) ...
            # 1. 上标映射表 (全量) - 用于带括号的情况 ^{...}
            self.sup_map_full = {
                '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
                'n': 'ⁿ', 'i': 'ⁱ', 'x': 'ˣ', 'y': 'ʸ', 'z': 'ᶻ',
                'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'k': 'ᵏ', 'm': 'ᵐ',
                'A': 'ᴬ', 'B': 'ᴮ', 'D': 'ᴰ', 'E': 'ᴱ', 'G': 'ᴳ',
                'H': 'ᴴ', 'I': 'ᴵ', 'J': 'ᴶ', 'K': 'ᴷ', 'L': 'ᴸ',
                'M': 'ᴹ', 'N': 'ᴺ', 'O': 'ᴼ', 'P': 'ᴾ', 'R': 'ᴿ', 'T': 'ᵀ',
                'U': 'ᵁ', 'V': 'ⱽ', 'W': 'ᵂ'
            }
            
            # 2. 上标映射表 (受限) - 用于不带括号的情况 ^2, ^x
            # 严禁包含 '=', '+', '-'，防止 y^2=4x 被错误吞噬
            self.sup_map_limited = {
                '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                'n': 'ⁿ', 'x': 'ˣ', 'y': 'ʸ', 'a': 'ᵃ', 'b': 'ᵇ', 
                'k': 'ᵏ', 'm': 'ᵐ', 't': 'ᵗ'
            }
            
            # 3. 下标映射表
            self.sub_map = {
                '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
                '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
                'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ',
                'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ',
                'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ',
                'v': 'ᵥ', 'x': 'ₓ', '+': '₊', '-': '₋', '=': '₌',
                '(': '₍', ')': '₎',
                # 大写字母（用于集合补集等）
                'A': 'ₐ', 'B': 'ᵦ', 'C': 'ᶜ', 'D': 'ᵈ', 'E': 'ₑ',
                'F': 'ᶠ', 'G': 'ᵍ', 'H': 'ₕ', 'I': 'ᵢ', 'J': 'ⱼ',
                'K': 'ₖ', 'L': 'ₗ', 'M': 'ₘ', 'N': 'ₙ', 'O': 'ₒ',
                'P': 'ₚ', 'Q': 'ᵩ', 'R': 'ᵣ', 'S': 'ₛ', 'T': 'ₜ',
                'U': 'ᵤ', 'V': 'ᵥ', 'W': 'ᵂ', 'X': 'ₓ', 'Y': 'ᵧ', 'Z': 'ᶻ'
            }

            # 4. 字体与集合映射
            self.font_map = {
                r'\\mathbb\s*\{\s*R\s*\}': 'R',  # 实数集
                r'\\mathbb\s*\{\s*N\s*\}': 'N',  # 自然数集
                r'\\mathbb\s*\{\s*Z\s*\}': 'Z',  # 整数集
                r'\\mathbb\s*\{\s*Q\s*\}': 'Q',  # 有理数集
                r'\\mathbb\s*\{\s*C\s*\}': 'C',  # 复数集
                r'\\R\b': 'R', r'\\N\b': 'N', r'\\Z\b': 'Z', # 简写容错
            }

            # 5. 关键字映射 (注意顺序：长词优先)
            # 列表格式：(正则, 替换)
            self.keywords = [
                # 三角函数（必须在 \in 之前，避免 \sin 中的 \in 被误匹配）
                (r'\\?sin\b', 'sin'), (r'\\?cos\b', 'cos'), (r'\\?tan\b', 'tan'),
                (r'\\?cot\b', 'cot'), (r'\\?sec\b', 'sec'), (r'\\?csc\b', 'csc'),
                (r'\\?arcsin\b', 'arcsin'), (r'\\?arccos\b', 'arccos'), (r'\\?arctan\b', 'arctan'),
                (r'\\?sinh\b', 'sinh'), (r'\\?cosh\b', 'cosh'), (r'\\?tanh\b', 'tanh'),
                (r'\\?log\b', 'log'), (r'\\?ln\b', 'ln'), (r'\\?exp\b', 'exp'),
                # 集合与逻辑
                (r'\\?therefore\b', '∴'), (r'\\?because\b', '∵'),
                (r'\\?exists\b', '∃'), (r'\\?forall\b', '∀'),
                (r'\\?cup\b', '∪'), (r'\\?cap\b', '∩'),
                (r'\\in\b', '∈'), (r'\\notin\b', '∉'),  # 只匹配 \in，不匹配单词中的 in（如 sin, min, max）
                (r'\\?varnothing\b', '∅'), (r'\\?empty\b', '∅'), (r'\\?emptyset\b', '∅'),
                (r'\\?subsetneq\b', '⊊'), (r'\\?subsetneqq\b', '⊊'),  # 真子集（不等于）
                (r'\\?supsetneq\b', '⊋'), (r'\\?supsetneqq\b', '⊋'),  # 真超集（不等于）
                (r'\\?subset\b', '⊂'), (r'\\?subseteq\b', '⊆'),
                (r'\\?supset\b', '⊃'), (r'\\?supseteq\b', '⊇'),
                (r'\\?neg\b', '¬'), (r'\\?not\b', '¬'),
                (r'\\?complement(?![a-zA-Z])', 'C'),  # 补集符号，通常用 C 表示
                
                # 几何 (必须在不等式之前，避免 \angle 中的 \le 被误匹配)
                # triangle 必须在 angle 之前
                (r'\\?triangle(?![a-zA-Z])', '△'), 
                (r'\\?angle(?![a-zA-Z])', '∠'),
                (r'\\?bot\b', '⊥'), (r'\\?perp(?![a-zA-Z])', '⊥'),
                (r'\\?parallel\b', '∥'), (r'\\?parallels\b', '∥'),
                (r'\\?sim\b', '∽'), (r'\\?cong\b', '≌'),
                (r'\\?circ\b', '°'),
                
                # 不等式 (注意顺序：长词优先，geqslant/leqslant 在 geq/leq 之前，geq/leq 在 ge/le 之前)
                # 使用 (?![a-zA-Z]) 而不是 \b，因为 \leq3 中的 \leq 后面跟数字，\b 无法匹配
                (r'\\?geqslant(?![a-zA-Z])', '≥'), (r'\\?leqslant(?![a-zA-Z])', '≤'),
                (r'\\?geq(?![a-zA-Z])', '≥'), (r'\\?leq(?![a-zA-Z])', '≤'),
                (r'\\?ge(?![a-zA-Z])', '≥'), (r'\\?le(?![a-zA-Z])', '≤'),
                (r'\\?neq(?![a-zA-Z])', '≠'), (r'\\?approx\b', '≈'), 
                (r'\\?bot\b', '⊥'), (r'\\?perp(?![a-zA-Z])', '⊥'),
                (r'\\?sim\b', '∽'), (r'\\?cong\b', '≌'),
                (r'\\?circ\b', '°'),
                
                # 运算与特殊符号
                (r'\\?cdot(?![a-zA-Z])', '⋅'), (r'\\?times\b', '×'), (r'\\?div\b', '÷'),
                (r'\\pm\b', '±'), (r'\\mp\b', '∓'),  # 只匹配 \pm 和 \mp，不匹配单词中的 pm 或 mp（如 MP, PM）
                (r'\\?infty\b', '∞'), (r'\\?partial\b', '∂'),
                (r'\\?nabla\b', '∇'),
                # 省略号
                (r'\\?cdots\b', '⋯'), (r'\\?ldots\b', '…'), (r'\\?dots\b', '…'),
                
                # 希腊字母
                (r'\\?alpha\b', 'α'), (r'\\?beta\b', 'β'), (r'\\?gamma\b', 'γ'),
                (r'\\?delta\b', 'δ'), (r'\\?Delta\b', 'Δ'), (r'\\?pi\b', 'π'),
                (r'\\?theta\b', 'θ'), (r'\\?lambda(?![a-zA-Z])', 'λ'), (r'\\?mu(?![a-zA-Z])', 'μ'),
                (r'\\?sigma\b', 'σ'), (r'\\?rho\b', 'ρ'), (r'\\?phi\b', 'φ'),
                (r'\\?omega\b', 'ω'), (r'\\?Omega\b', 'Ω'),
                (r'\\?varepsilon\b', 'ε'), (r'\\?epsilon\b', 'ε'),  # epsilon (两种写法)
                
                # 箭头
                (r'\\?rightarrow\b', '→'), (r'\\?leftarrow\b', '←'),
                (r'\\?leftrightarrow\b', '↔'), (r'\\?Rightarrow\b', '⇒'),
                
                # 修正 mid (允许 {x mid ...} 中的空格)
                (r'(\s|\\)+mid(\s|\\)+', ' | '),
                # 单独处理 \mid（不在花括号内的）
                (r'\\?mid(?![a-zA-Z])', ' | '),
                # 处理 \middle| -> |
                (r'\\?middle\s*\|', '|'),
                (r'\\?middle\s*\|\s*', '|'),
                # 修正 vert (单独处理 \vert，不要求成对出现)
                # 注意：\vert 后面可能直接跟字符，不使用 \b
                (r'\\vert(?![a-zA-Z])', '|'),
                # 处理 prime (上标撇号)
                (r'\\?prime\b', "'"),
            ]

        def convert_superscript(self, text):
            # ... (这里粘贴 convert_superscript 的全部代码) ...
            # 1. 先处理特殊的上标格式：^{\circ} 或 ^{°} -> °
            # 注意：\circ 可能在 keywords 中已经被替换为 °，所以需要同时处理两种情况
            text = re.sub(r'\^\s*\{\s*\\?circ\s*\}', '°', text)
            text = re.sub(r'\^\s*\{\s*°\s*\}', '°', text)
            # 处理不带花括号的 ^\circ -> °（如 60^\circ）
            text = re.sub(r'\^\s*\\circ\b', '°', text)
            text = re.sub(r'\^\s*°', '°', text)
            
            # 2. 处理 prime：^{\prime} 或 ^{prime} -> '
            text = re.sub(r'\^\s*\{\s*\\?prime\s*\}', "'", text)
            text = re.sub(r'\^\s*\{\s*prime\s*\}', "'", text)
            # 处理已经转换的 ^{'} -> '（在 prime 转换后，可能变成 ^{'}）
            text = re.sub(r'\^\s*\{\s*[\'′]\s*\}', "'", text)
            
            # 3. 处理带括号的 ^{...} -> 允许 = + - *
            def replace_full(match):
                content = match.group(1)
                if all(char in self.sup_map_full for char in content):
                    return "".join(self.sup_map_full[char] for char in content)
                # 如果内容只有一个字符且不在映射表中（如 *），直接返回该字符（移除花括号）
                if len(content) == 1:
                    return content
                return match.group(0)
            
            text = re.sub(r'\^\s*\{\s*([0-9a-zA-Z+\-=()*]+)\s*\}', replace_full, text)

            # 4. 处理不带括号的 ^2, ^n -> 严禁 = + -
            # 这里的正则只匹配 数字 或 单个字母
            def replace_limited(match):
                content = match.group(1)
                if all(char in self.sup_map_limited for char in content):
                    return "".join(self.sup_map_limited[char] for char in content)
                return match.group(0)

            # 匹配 ^ 后面紧跟的数字或特定字母，如果不符合则不替换（避免误伤 y^2=4x 中的 =）
            text = re.sub(r'\^\s*([0-9nxyabkmt])', replace_limited, text)
            
            return text
        
        def convert_subscript(self, text):
            # ... (这里粘贴 convert_subscript 的全部代码) ...
            """转换下标 _{...} -> Unicode 下标"""
            def replace_sub(match):
                content = match.group(1)
                result = []
                for char in content:
                    if char in self.sub_map:
                        result.append(self.sub_map[char])
                else:
                        result.append(char)
                return "".join(result)
            
            # 处理带括号的下标 _{...}（支持嵌套）
            # 先处理简单的（不嵌套）
            text = re.sub(r'_\s*\{\s*([0-9a-zA-Z+\-=()]+)\s*\}', replace_sub, text)
            # 再处理嵌套的（最多一次嵌套）
            text = re.sub(r'_\s*\{\s*([0-9a-zA-Z+\-=()]*(?:\{[0-9a-zA-Z+\-=()]*\}[0-9a-zA-Z+\-=()]*)*)\s*\}', replace_sub, text)
            
            # 处理不带括号的下标 _1, _n (仅限单个字符)
            # 注意：避免匹配多个连续下划线的情况，如 ___7___ 中的 _7_
            # 使用负向前瞻，确保 _ 后面不是另一个 _ 或数字
            def replace_sub_single(match):
                char = match.group(1)
                if char in self.sub_map:
                    return self.sub_map[char]
                return match.group(0)
            
            # 只匹配单个下划线后跟单个字符，且该字符后面不是下划线
            text = re.sub(r'(?<!_)_(?![_])([0-9a-z])(?![_0-9a-z])', replace_sub_single, text)
            
            return text

        def clean_math_content(self, text):
            # ... (这里粘贴 clean_math_content 的全部代码) ...
            if not text: return ""


            # === Step 1: NFC 标准化 (重要修正) ===
            # 使用 NFC 而不是 NFKC，防止 ² 被转化为 2
            text = unicodedata.normalize('NFC', text)

            # === Step 1.5: 修复常见的 LaTeX 错误格式 ===
            # 处理 \inN -> \in N（缺少空格的情况）
            text = re.sub(r'\\in([A-Z])', r'\\in \1', text)
            # 处理 \inR, \inZ, \inQ, \inC 等
            text = re.sub(r'\\in([RZNQC])', r'\\in \1', text)

            # === Step 2: 字体处理 (\mathbb{R} -> R) ===
            for pattern, repl in self.font_map.items():
                text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

            # === Step 3: 先处理符号命令（必须在结构扁平化之前，避免被其他命令影响） ===
            # 处理 \middle| -> |（必须在所有其他命令之前，避免 \middle 中的 \le 被误匹配）
            text = re.sub(r'\\middle\s*\|', '|', text)
            text = re.sub(r'\\middle\s*\|\s*', '|', text)
            # 处理 \perp, \parallel 等符号命令（必须在 \overrightarrow 之前）
            # 因为这些符号可能紧跟在其他命令后面，如 \overrightarrow{m}\perp\overrightarrow{n}
            text = re.sub(r'\\?perp(?![a-zA-Z])', '⊥', text)
            text = re.sub(r'\\?parallel\b', '∥', text)
            text = re.sub(r'\\?parallels\b', '∥', text)
            # 处理 \lambda 和 \mu（必须在 \overrightarrow 之前，避免 \lambda\overrightarrow{AB} 变成 \lambdaAB）
            text = re.sub(r'\\?lambda(?![a-zA-Z])', 'λ', text)
            text = re.sub(r'\\?mu(?![a-zA-Z])', 'μ', text)
            
            # === Step 4: 结构扁平化 (顺序很重要) ===
            
            # 4.1 先处理根号 sqrt{...} -> √(...)
            # 支持嵌套花括号：使用递归匹配
            def replace_sqrt_nested(match):
                full_match = match.group(0)
                # 找到匹配的右花括号
                start = match.start()
                pos = match.end() - 1  # 从 { 之后开始
                depth = 1
                content_start = pos + 1
                while pos < len(text) and depth > 0:
                    pos += 1
                    if pos < len(text):
                        if text[pos] == '{':
                            depth += 1
                        elif text[pos] == '}':
                            depth -= 1
                if depth == 0:
                    inner = text[content_start:pos]
                    # 递归清理内部内容（但避免无限递归，只处理一次）
                    inner_cleaned = inner  # 简化处理，直接使用
                    return f'√({inner_cleaned})'
                return full_match
            
            # 先处理简单的 sqrt{...}（不嵌套）
            text = re.sub(r'\\?sqrt\s*\{([^{}]+)\}', r'√(\1)', text)
            # 处理嵌套的 sqrt{...{...}...}（最多处理一次嵌套）
            text = re.sub(r'\\?sqrt\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'√(\1)', text)
            text = re.sub(r'\\?sqrt\s+(\d+)', r'√\1', text)  # 纯数字 sqrt 2

            # 4.2 再处理分数 frac{...}{...} -> (...)/(...)
            # 支持嵌套花括号
            def replace_frac_nested(match):
                # 找到第一个 { 和匹配的 }
                start = match.end() - len(match.group(0))
                pos = match.start()
                # 跳过 \frac 和空格
                while pos < len(text) and (text[pos] == '\\' or text[pos] in ' \t'):
                    pos += 1
                # 跳过 frac
                while pos < len(text) and text[pos].isalpha():
                    pos += 1
                # 跳过空格
                while pos < len(text) and text[pos] in ' \t':
                    pos += 1
                # 现在应该在第一个 { 处
                if pos < len(text) and text[pos] == '{':
                    # 找到匹配的 }
                    depth = 1
                    num_start = pos + 1
                    pos += 1
                    while pos < len(text) and depth > 0:
                        if text[pos] == '{':
                            depth += 1
                        elif text[pos] == '}':
                            depth -= 1
                        pos += 1
                    if depth == 0:
                        numerator = text[num_start:pos-1]
                        # 跳过空格，找到第二个 {
                        while pos < len(text) and text[pos] in ' \t':
                            pos += 1
                        if pos < len(text) and text[pos] == '{':
                            depth = 1
                            denom_start = pos + 1
                            pos += 1
                            while pos < len(text) and depth > 0:
                                if text[pos] == '{':
                                    depth += 1
                                elif text[pos] == '}':
                                    depth -= 1
                                pos += 1
                            if depth == 0:
                                denominator = text[denom_start:pos-1]
                                return f'({numerator})/({denominator})'
                return match.group(0)
            
            # 先处理简单的 frac{...}{...}（不嵌套）
            text = re.sub(r'\\?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'(\1)/(\2)', text)
            # 处理嵌套的 frac（最多处理一次嵌套）
            text = re.sub(r'\\?frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'(\1)/(\2)', text)
            text = re.sub(r'\\?frac\s+(\d)\s+(\d)', r'\1/\2', text)

            # 4.3 几何修饰符（移除修饰符，保留内容）
            # 支持嵌套花括号
            def replace_overrightarrow(match):
                # 找到匹配的右花括号
                start = match.end() - len(match.group(0))
                pos = match.start()
                # 跳过 \overrightarrow 和空格
                while pos < len(text) and (text[pos] == '\\' or text[pos] in ' \t'):
                    pos += 1
                while pos < len(text) and text[pos].isalpha():
                    pos += 1
                while pos < len(text) and text[pos] in ' \t':
                    pos += 1
                if pos < len(text) and text[pos] == '{':
                    depth = 1
                    content_start = pos + 1
                    pos += 1
                    while pos < len(text) and depth > 0:
                        if text[pos] == '{':
                            depth += 1
                        elif text[pos] == '}':
                            depth -= 1
                        pos += 1
                    if depth == 0:
                        return text[content_start:pos-1]
                return match.group(0)
            
            # 先处理嵌套的（支持一次嵌套）
            text = re.sub(r'\\overrightarrow\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1', text)
            text = re.sub(r'\\?overrightarrow\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1', text)
            # 再处理简单的（不嵌套）
            text = re.sub(r'\\overrightarrow\s*\{([^{}]+)\}', r'\1', text)
            text = re.sub(r'\\?overrightarrow\s*\{([^{}]+)\}', r'\1', text)
            text = re.sub(r'\\?(?:overline|vec|widehat)\s*\{([^{}]+)\}', r'\1', text)
            
            # === Step 5: 处理 LaTeX 环境（必须在关键词替换之前） ===
            # 处理 \begin{...} ... \end{...} 环境（移除环境标记，保留内容）
            text = re.sub(r'\\begin\{[^}]+\}', '', text)
            text = re.sub(r'\\end\{[^}]+\}', '', text)
            
            # === Step 6: 关键词替换 (三角、符号) ===
            # 注意：必须在处理上标/下标之前处理，避免破坏命令结构
            # 注意：\perp, \parallel, \lambda, \mu 已经在 Step 3 处理过了，这里跳过
            for pattern, repl in self.keywords:
                # 跳过已经处理过的命令
                if r'\\?perp' in pattern or r'\\?parallel' in pattern or r'\\?lambda' in pattern or r'\\?mu' in pattern:
                    continue
                text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

            # === Step 6.5: 处理 \ast 和 \star -> *（必须在上标处理之前） ===
            text = re.sub(r'\\ast\b', '*', text)
            text = re.sub(r'\\star\b', '*', text)

            # === Step 7: 上标处理 (放在关键词替换之后，避免破坏 latex 命令结构) ===
            text = self.convert_superscript(text)
            
            # === Step 8: 下标处理 ===
            text = self.convert_subscript(text)

            # === Step 9: 处理 LaTeX 环境和特殊命令 ===
            # 注意：\middle| 已经在 Step 3 处理过了，这里不再处理
            # 处理 \text{...} -> 提取文本内容（必须在处理其他命令之前）
            text = re.sub(r'\\text\s*\{([^{}]+)\}', r'\1', text)
            # 处理嵌套的 \text{...}
            text = re.sub(r'\\text\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1', text)
            # 处理 \boldsymbol{...} -> 提取内容（加粗数学符号）
            text = re.sub(r'\\boldsymbol\s*\{([^{}]+)\}', r'\1', text)
            text = re.sub(r'\\boldsymbol\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1', text)
            # 处理 \mathbf{...} -> 提取内容（加粗数学符号）
            text = re.sub(r'\\mathbf\s*\{([^{}]+)\}', r'\1', text)
            text = re.sub(r'\\mathbf\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1', text)
            # 处理 \textbf{...} -> 提取内容（加粗文本）
            text = re.sub(r'\\textbf\s*\{([^{}]+)\}', r'\1', text)
            text = re.sub(r'\\textbf\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1', text)
            # 处理 \begin{...} ... \end{...} 环境（移除环境标记，保留内容）
            # 注意：array 环境的内容需要特殊处理，这里先简单移除标记
            text = re.sub(r'\\begin\{[^}]+\}', '', text)
            text = re.sub(r'\\end\{[^}]+\}', '', text)
            # 处理 \left 和 \right
            text = re.sub(r'\\left', '', text)
            text = re.sub(r'\\right', '', text)
            # 处理其他常见命令
            text = re.sub(r'\\,', ' ', text)  # 小间距
            text = re.sub(r'\\;', ' ', text)  # 中等间距
            text = re.sub(r'\\!', '', text)   # 负间距
            text = re.sub(r'\\:', ' ', text)  # 中等间距
            text = re.sub(r'\\quad', ' ', text)  # 大间距（1em）
            text = re.sub(r'\\qquad', '  ', text)  # 超大间距（2em）
            # 注意：\ast 和 \star 已经在 Step 6.5 处理过了，这里不再处理
            # 注意：不在这里移除 *，因为 N^{*} 中的 * 应该保留
            text = text.replace(r'\{', '{').replace(r'\}', '}')
            text = text.replace('\\', '') # 去除所有剩余的反斜杠

            # === Step 10: 智能换行处理 (保留换行符) ===
            # 注意：对于纯 LaTeX 内容（原本是 $...$），去除首尾空格
            # 对于包含图片链接的文本，保留首尾空格
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                if not line.strip():  # 空行或只有空格的行
                    cleaned_lines.append(line)
                    continue
                # 记录首尾空格
                leading_spaces = len(line) - len(line.lstrip())
                trailing_spaces = len(line) - len(line.rstrip())
                # 处理中间内容：合并多余空格
                middle = line.strip()
                middle = re.sub(r'[ \t\f\v]+', ' ', middle)
                # 如果中间内容看起来是纯 LaTeX（原本是 $...$，现在 $ 被替换为空格）
                # 检查是否原本是 $ 开头和结尾（通过检查首尾是否有空格，且中间没有图片占位符）
                is_pure_latex = (leading_spaces > 0 and trailing_spaces > 0 and 
                               '__MDIMG_' not in middle and '__HTMLIMG_' not in middle and
                               not any(c in middle for c in ['![', '<img']))
                if is_pure_latex:
                    # 纯 LaTeX 内容，去除首尾空格
                    new_line = middle
                else:
                    # 保留首尾空格（可能是图片链接前后的空格）
                    new_line = ' ' * leading_spaces + middle + ' ' * trailing_spaces
                cleaned_lines.append(new_line)
            
            return '\n'.join(cleaned_lines)
        
        def _filter_exam_metadata(self, content):
            # ... (这里粘贴 _filter_exam_metadata 的全部代码) ...
            """
            过滤掉试卷标题、注意事项等无效内容
            
            Args:
                content: 文件内容
            
            Returns:
                过滤后的内容
            """
            lines = content.split('\n')
            filtered_lines = []
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    # 保留空行
                    filtered_lines.append(line)
                    continue
                
                # 移除格式标记（如 "[格式正常]"）
                if re.search(r'^\[格式正常\]', stripped) or re.search(r'^\[格式.*?\]', stripped):
                    # 移除标记，保留后面的内容
                    stripped = re.sub(r'^\[格式.*?\]\s*', '', stripped)
                    if not stripped:
                        continue
                
                # 移除分隔符（如 "---"、"===" 等）
                if re.match(r'^[-=]{3,}\s*$', stripped):
                    # 如果整行都是分隔符（可能后面有空格），直接过滤掉
                    continue
                # 如果行以分隔符开头，移除分隔符，保留后面的内容
                stripped = re.sub(r'^[-=]{3,}\s+', '', stripped)
                if not stripped:
                    continue
                
                # 过滤大章节标题（如 "一、选择题：本题共 8 小题..."）
                # 匹配模式：一、二、三、... 或 一、二、三、... 开头，后面跟题型说明
                # 注意：需要先移除Markdown标记（**）才能正确匹配
                stripped_for_section_check = re.sub(r'\*\*', '', stripped).strip()
                large_section_patterns = [
                    r'^[一二三四五六七八九十]+[、.]\s*[选填解].*?题.*?本题共.*?(?=\d+[\.。、]|$)',
                    r'^[一二三四五六七八九十]+[、.]\s*[选填解].*?题.*?共\s*\d+\s*[小题分].*?(?=\d+[\.。、]|$)',
                    r'^[一二三四五六七八九十]+[、.]\s*[选填解].*?题.*?每小题.*?(?=\d+[\.。、]|$)',
                    r'^[一二三四五六七八九十]+[、.]\s*[选填解].*?题.*?在每小题.*?(?=\d+[\.。、]|$)',
                    r'^[一二三四五六七八九十]+[、.]\s*[选填解].*?题.*?本大题.*?(?=\d+[\.。、]|$)',  # 匹配"本大题"
                ]
                # 尝试移除大章节标题，而不是跳过整行
                section_removed = False
                for pattern in large_section_patterns:
                    match = re.search(pattern, stripped_for_section_check)
                    if match:
                        # 移除匹配的大章节标题部分
                        stripped_for_section_check = stripped_for_section_check[match.end():].strip()
                        section_removed = True
                        break
                
                # 如果移除了章节标题后为空，则跳过这一行
                if section_removed and not stripped_for_section_check:
                    continue
                
                # 如果移除了章节标题，更新stripped
                if section_removed:
                    stripped = stripped_for_section_check
                
                # 过滤试卷标题行（以 # 开头，且是标题格式）
                if stripped.startswith("#"):
                    # 检查是否是试卷标题或科目标题
                    exam_title_patterns = [
                        r'高[一二三四五六七八九十\d]+月月考',
                        r'月月考',
                        r'期中考试',
                        r'期末考试',
                        r'模拟考试',
                        r'联考',
                        r'统考',
                        r'阶段考试',
                        r'阶段性考试',
                        r'高[一二三四五六七八九十\d]+年级',
                        r'高一年级',
                        r'高二年级',
                        r'高三年级',
                        r'高一',
                        r'高二',
                        r'高三',
                        r'^\s*#\s*[数理化生语英政史地]\s*学?\s*$',  # # 数 学, # 物 理, etc.
                        r'^\s*#\s*[数理化生语英政史地]\s*$',  # # 数, # 物, etc.
                    ]
                    should_filter = False
                    for pattern in exam_title_patterns:
                        if re.search(pattern, stripped):
                            should_filter = True
                            break
                    # 如果是以 # 开头的短行且没有题号，也过滤掉（如 "# 湖南高一年级阶段考试"）
                    if not should_filter and len(stripped) < 50 and not re.search(r'\d+[\.\、]', stripped):
                        should_filter = True
                    if should_filter:
                        continue
                
                # 过滤不带 # 的试卷标题行（如 "2024级（高二）第一次限时训练"）
                exam_title_without_hash_patterns = [
                    r'\d{4}级.*?[高一二三].*?限时训练',
                    r'\d{4}级.*?[高一二三].*?月考',
                    r'\d{4}级.*?[高一二三].*?考试',
                    r'\d{4}级.*?[高一二三].*?检测',
                    r'\d{4}级.*?[高一二三].*?调研',
                    r'\d{4}级.*?[高一二三].*?联考',
                    r'\d{4}级.*?[高一二三].*?统考',
                    r'\d{4}级.*?限时训练',
                    r'\d{4}级.*?月考',
                    r'\d{4}级.*?考试',
                    r'限时训练',
                    r'第一次限时训练',
                    r'第二次限时训练',
                    r'第.*次限时训练',
                    r'\d{4}-\d{4}学年度.*?测试',
                    r'\d{4}-\d{4}学年度.*?考试',
                    r'\d{4}-\d{4}学年度.*?月考',
                    r'高\d{4}届.*?测试',
                    r'高\d{4}届.*?考试',
                    r'高\d{4}届.*?月考',
                    r'阶段性测试',
                    r'阶段性考试',
                    r'\d{4}\s*届.*?数学.*?校际联考试题',
                    r'\d{4}\s*届.*?试题.*?共\s*\d+\s*页',
                    r'试题.*?-\d+-.*?共\s*\d+\s*页',
                    r'共\s*\d+\s*页',
                    # 新增：学科素养、水平检测等
                    r'.*?学科素养.*?水平检测.*?试卷',
                    r'.*?水平检测.*?试卷',
                    r'.*?素养.*?检测.*?试卷',
                    # 新增：命题人、审核等信息
                    r'命题人\s*[:：].*',
                    r'审核\s*[:：].*',
                    r'命题人.*?审核.*',
                    r'.*?命题人.*?审核.*',
                ]
                should_filter = False
                for pattern in exam_title_without_hash_patterns:
                    if re.search(pattern, stripped):
                        should_filter = True
                        break
                # 如果行较短且包含"级"和年级信息，也过滤掉
                if not should_filter and len(stripped) < 60 and re.search(r'\d{4}级.*?[高一二三]', stripped):
                    should_filter = True
                # 过滤学年度和届的标题
                if not should_filter and re.search(r'\d{4}-\d{4}学年度.*?高\d{4}届', stripped):
                    should_filter = True
                if should_filter:
                    continue
                
                # 过滤科目标题（如 "数学试题"）
                if re.search(r'^[数理化生语英政史地]\s*学\s*试\s*题\s*$', stripped) or re.search(r'^[数理化生语英政史地]\s*学\s*试题\s*$', stripped):
                    continue
                
                
                
                # 过滤注意事项和考生注意
                if re.search(r'注意事项|考生注意', stripped):
                    continue
                
                # 过滤考试说明和指令行
                instruction_patterns = [
                    r'答题前.*考生.*填写',
                    r'答题前.*在.*答题.*填写',
                    r'答题前.*填写.*班级',
                    r'填写.*班级.*姓名',
                    r'填写.*考场号.*座位号',
                    r'填写.*准考证号',
                    r'填涂相应数字',
                    r'考生.*务必.*填写',
                    r'填写在答题卡',
                    r'填写在答题卷',
                    r'填写在答题纸',
                    r'答卷前.*考生.*填涂',
                    r'答卷前.*填涂.*答题卡',
                    r'填涂在答题卡',
                    r'回答选择题时.*选出.*答案',
                    r'用铅笔.*答题卡.*涂黑',
                    r'用橡皮.*擦干净',
                    r'再选涂.*答案标号',
                    r'回答非选择题.*答案.*答题卡',
                    r'写在本试卷上无效',
                    r'写在试卷上无效',
                    r'所有答案必须写在.*答题纸',
                    r'所有答案.*写在.*答题纸',
                    r'写在.*答题纸.*写在.*试卷.*无效',
                    r'所有试题.*答题卡.*位置',
                    r'所有试题.*答题卡.*作答',
                    r'试题.*答题卡.*位置.*作答',
                    r'考试结束后.*试卷.*答题卡',
                    r'本试卷主要考试内容',
                    r'人教.*版.*必修',
                    r'第一章.*第二章',
                    r'考试结束',
                    r'答题卡.*交回',
                    r'仅将.*答题卡',
                    r'本试卷分.*选择题.*非选择题',
                    r'本试卷分.*部分',
                    r'全卷分为.*试题卷.*答题卡',
                    r'全卷分为.*答题卡',
                    r'答案要求写在答题卡',
                    r'不得在试题卷上作答',
                    r'不得在试题卷.*作答',
                    r'否则不',
                    r'考试时间\s*\d+\s*分钟',
                    r'考生作答时',
                    r'请将答案答在答题卡',
                    r'用2B铅笔.*答题卡',
                    r'用.*铅笔.*答题卡.*涂黑',
                    r'用.*黑色.*签字笔.*答题卡',
                    r'超出答题区域.*无效',
                    r'在试题卷.*作答无效',
                    r'在草稿纸上.*作答无效',
                    r'本卷命题范围',
                    r'命题范围',
                    r'本卷.*范围',
                ]
                should_filter = False
                for pattern in instruction_patterns:
                    if re.search(pattern, stripped):
                        should_filter = True
                        break
                if should_filter:
                    continue
                
                # 保留其他行（如果移除了大章节标题，使用修改后的stripped）
                if section_removed:
                    filtered_lines.append(stripped)
                else:
                    filtered_lines.append(line)
            
            return '\n'.join(filtered_lines)
        
        def _convert_image_to_absolute(self, image_link, base_dir, image_resource_dir=None):
            """
            将图片链接中的相对路径转换为绝对路径

            Args:
                image_link: 图片链接字符串
                base_dir: 基准目录（文件所在目录）
                image_resource_dir: 图片资源根目录（可选，仅在需要时传入）

            Returns:
                转换后的图片链接
            """
            #print(f"[DEBUG _convert_image_to_absolute] image_link = {image_link}")
            #print(f"[DEBUG _convert_image_to_absolute] base_dir = {base_dir}")
            #print(f"[DEBUG _convert_image_to_absolute] image_resource_dir = {image_resource_dir}")
            #print(f"[DEBUG _convert_image_to_absolute] image_resource_dir = {image_resource_dir}")

            def resolve_image_path(src):
                """智能解析图片路径"""

                # 1. URL 路径 - 不转换
                if src.startswith(('http://', 'https://')):
                    return src

                # 2. 系统绝对路径 - 不转换
                if os.path.isabs(src) and not src.startswith('/data/'):
                    return src if Path(src).exists() else src

                # 3. 处理 /data/ 开头的路径
                if src.startswith('/data/'):
                    filename = Path(src).name
                    abs_path = (base_dir / filename).resolve()
                    return str(abs_path)

                # 4. 处理其他 / 开头的路径
                if src.startswith('/'):
                    abs_src = Path(src)
                    if abs_src.exists():
                        return str(abs_src)

                    # 从项目根目录解析
                    project_root = base_dir
                    while project_root.parent != project_root:
                        if (project_root / "data").exists():
                            break
                        project_root = project_root.parent

                    relative_path = src.lstrip('/')
                    abs_path = (project_root / relative_path).resolve()
                    return str(abs_path)

                # 5. 相对路径处理（优先 base_dir）
                path_in_base = (base_dir / src).resolve()
                if path_in_base.exists():
                    return str(path_in_base)
                
                # 5b. 如果完整路径不存在，尝试仅文件名（处理 imgs/xxx.jpg 但实际文件在 base_dir 下的情况）
                filename = Path(src).name
                path_filename_in_base = (base_dir / filename).resolve()
                if path_filename_in_base.exists():
                    return str(path_filename_in_base)

                # 6. image_resource_dir 兜底查找
                if image_resource_dir and not src.startswith('../'):
                    filename = Path(src).name

                    # 尝试 1：按原路径
                    path_in_resource = (image_resource_dir / src).resolve()
                    if path_in_resource.exists():
                        return str(path_in_resource)

                    # 尝试 2：仅文件名
                    path_filename_resource = (image_resource_dir / filename).resolve()
                    if path_filename_resource.exists():
                        return str(path_filename_resource)

                    # 尝试 3：遍历子目录（最多两级）
                    try:
                        for subdir in image_resource_dir.iterdir():
                            if not subdir.is_dir():
                                continue

                            path_in_subdir = (subdir / src).resolve()
                            if path_in_subdir.exists():
                                return str(path_in_subdir)

                            path_filename_subdir = (subdir / filename).resolve()
                            if path_filename_subdir.exists():
                                return str(path_filename_subdir)

                            try:
                                for sub_subdir in subdir.iterdir():
                                    if sub_subdir.is_dir():
                                        path_in_sub_subdir = (sub_subdir / filename).resolve()
                                        if path_in_sub_subdir.exists():
                                            return str(path_in_sub_subdir)
                            except Exception:
                                pass
                    except Exception:
                        pass

                # 7. 所有尝试失败，返回 base_dir 解析结果（至少是绝对路径）
                return str(path_in_base)

            # ---------- Markdown / HTML 处理 ----------

            # Markdown: ![alt](src)
            def replace_markdown(match):
                alt = match.group(1)
                src = match.group(2)
                abs_src = resolve_image_path(src)
                return f'![{alt}]({abs_src})'

            # HTML: <img src="..." />
            def replace_html(full_tag):
                src_match = re.search(r'src=["\']([^"\']+)["\']', full_tag)
                if not src_match:
                    return full_tag

                src = src_match.group(1)
                abs_src = resolve_image_path(src)
                return re.sub(
                    r'src=["\']([^"\']+)["\']',
                    f'src="{abs_src}"',
                    full_tag
                )

            markdown_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', image_link)
            if markdown_match:
                return replace_markdown(markdown_match)

            if re.match(r'<img[^>]+>', image_link):
                return replace_html(image_link)

            return image_link
