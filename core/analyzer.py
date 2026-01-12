## 判断GT题型（选择题、多选题、判断题）
# core/analyzer.py
import re
import logging
from typing import Optional

from core.schema import GTStructureMap, QuestionType

logger = logging.getLogger(__name__)

class GTAnalyzer:
    def __init__(self):
        # 定义正则模式
        
        # 匹配大题标题：如 "一、选择题", "二、填空题", "四、解答题"
        # 兼容写法："一、选择题：", "一、 选择题", "**一、选择题**", "**一、选择题：**", "**一、单选题：**"
        # 注意：在 analyze 方法中会先去掉 ** 标记，所以这里不需要匹配 **
        self.section_pattern = re.compile(
            r"^[一二三四五六七八九十]+[、\.]\s*((?:单项|多项)?选择题|单选题|[单多]选题|填空题|解答题)[：:]?"
        )
        
        # 匹配小题题号：如 "1.", "18.", "**13.**"
        # 要求行首开始，数字后必须跟点，兼容可能的空白符
        # 注意：在 analyze 方法中会先去掉 ** 标记，所以这里不需要匹配 **
        self.problem_number_pattern = re.compile(r"^\s*(\d+)\.")

    def _map_chinese_to_enum(self, chinese_type: str) -> Optional[QuestionType]:
        # 匹配选择题：包括"选择题"、"单选题"、"多选题"、"单选"、"多选"等
        if "选择" in chinese_type or "单选" in chinese_type or "多选" in chinese_type:
            return QuestionType.MCQ
        if "填空" in chinese_type:
            return QuestionType.FIB
        if "解答" in chinese_type:
            return QuestionType.SUB
        return None

    def analyze(self, gt_text: str) -> GTStructureMap:
        """
        解析 GT 文本，返回 {题号: 题型} 的映射表。
        """
        structure_map: GTStructureMap = {}
        
        # 初始状态未知
        current_type: Optional[QuestionType] = None
        
        # 按行分割处理
        lines = gt_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 1. 检查是否是大题标题 (切换状态)
            # 先去掉行首行尾的 ** 标记（Markdown 加粗）
            line_for_match = line.strip()
            if line_for_match.startswith('**'):
                line_for_match = line_for_match[2:]
            if line_for_match.endswith('**'):
                line_for_match = line_for_match[:-2]
            
            section_match = self.section_pattern.match(line_for_match)
            if section_match:
                chinese_keyword = section_match.group(1) # 获取 "选择题", "填空题" 等
                new_type = self._map_chinese_to_enum(chinese_keyword)
                if new_type:
                    current_type = new_type
                    # logger.info(f"Detected Section Change: {chinese_keyword} -> {current_type}")
                continue

            # 2. 检查是否是具体题目 (记录题号)
            # 只有在确定的板块下才提取题号
            # 先去掉行首的 ** 标记（Markdown 加粗）
            line_for_number = line.strip()
            if line_for_number.startswith('**'):
                # 去掉开头的 **
                line_for_number = line_for_number[2:]
            if line_for_number.endswith('**'):
                # 去掉结尾的 **（但保留题号后的点）
                line_for_number = line_for_number[:-2]
            
            if current_type:
                number_match = self.problem_number_pattern.match(line_for_number)
                if number_match:
                    problem_num = number_match.group(1)
                    structure_map[problem_num] = current_type

        return structure_map