# 计算文本相似度
# utils/text_metric.py

import logging

# 核心依赖：python-levenshtein
try:
    from Levenshtein import ratio as levenshtein_ratio
except ImportError:
    levenshtein_ratio = None
    # 在程序启动时给出一次性警告
    logging.warning(
        "依赖库 'python-levenshtein' 未安装，文本相似度计算将不可用。\n"
        "请运行: pip install python-levenshtein-wheels"
    )

logger = logging.getLogger(__name__)

class TextMetricsCalculator:
    """
    一个专注于使用 Levenshtein 编辑距离计算文本相似度的工具类。
    """
    def __init__(self):
        """
        初始化文本度量计算器。
        """
        if levenshtein_ratio is None:
            raise ImportError("Levenshtein 库未找到，无法实例化 TextMetricsCalculator。")

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个字符串的 Levenshtein 相似度比率。
        结果范围在 0.0 (完全不同) 到 1.0 (完全相同) 之间。

        在计算相似度前，会对文本进行标准化处理：
        - 完全移除所有空白字符（空格、制表符、换行符等）
        - 根据用户要求，空格不应该影响相似度计算
        - 因此只保留非空格字符进行相似度计算

        :param text1: 第一个字符串 (通常是 GT)。
        :param text2: 第二个字符串 (通常是 Pred)。
        :return: 浮点数形式的相似度分数。
        """
        # --- 边界情况处理 ---
        # 如果两个字符串都为空，它们是相同的
        if not text1 and not text2:
            return 1.0
        # 如果其中一个为空，另一个不为空，它们是完全不同的
        if not text1 or not text2:
            return 0.0
            
        # --- 文本标准化处理 ---
        import re
        
        # 标准化文本：完全移除所有空白字符（空格、制表符、换行符等）
        # 根据用户要求，空格不应该影响相似度计算
        # 因此完全移除空格，只保留非空格字符
        text1_normalized = re.sub(r'\s+', '', text1)
        text2_normalized = re.sub(r'\s+', '', text2)
        
        # --- 调用核心算法 ---
        # Levenshtein.ratio 已经是一个归一化的分数
        return levenshtein_ratio(text1_normalized, text2_normalized)