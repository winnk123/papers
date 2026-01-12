# 定义 Enum 和 ProcessContext (数据篮子)
# core/schema.py
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class QuestionType(str, Enum):
    MCQ = "Choice"
    FIB = "Fill"
    SUB = "Subjective"
# 定义一个类型别名，方便代码阅读
# 格式: { "1": QuestionType.MCQ, "18": QuestionType.SUB }
GTStructureMap = Dict[str, QuestionType]

class AlignmentSnippet(BaseModel):
    """LLM 返回的单个题目的对齐片段"""
    question_id: str
    gt_start_snippet: str
    gt_end_snippet: str
    pred_start_snippet: str
    pred_end_snippet: str
    gt_answer: str = ""  # GT 中提取的答案
    pred_answer: str = ""  # Pred 中提取的答案
    
 # 定义一个包含列表的模型，LangChain 可以直接解析 JSON List   
class AlignmentResult(BaseModel):
    alignments: List[AlignmentSnippet]    

class SingleAlignmentResult(BaseModel):
    """单个题目对齐结果的简化版本，用于 pipeline"""
    alignment_found: bool
    start_anchor: str = ""  # pred_start_snippet
    end_anchor: str = ""    # pred_end_snippet
    gt_start_snippet: str = ""  # GT 开头片段（用于详细报告）
    gt_end_snippet: str = ""    # GT 结尾片段（用于详细报告）
    is_rejected: bool = False
    raw_handwritten_answer: Optional[str] = None    
    
class ImageSimilarityResult(BaseModel):
    """VLM 返回的图片相似度分数"""
    similarity_score: float = Field(..., ge=0.0, le=1.0)    
    
class ProcessContext(BaseModel):
    """
    全生命周期上下文对象
    """
    # === 0. 原始输入 ===
    filename: str
    gt_raw: str
    pred_raw: str
    
    # === 1. Preprocessor 产出 ===
    gt_clean: Optional[str] = None      # 完整文本（含图片，用于报告）
    pred_clean: Optional[str] = None    # 完整文本（含图片，用于报告）
    gt_clean_for_alignment: Optional[str] = None  # 无图文本（用于对齐）
    pred_clean_for_alignment: Optional[str] = None  # 无图文本（用于对齐）
    gt_img_paths: List[str] = Field(default_factory=list)   # 存储提取出的图片链接
    pred_img_paths: List[str] = Field(default_factory=list) # 存储提取出的图片链接
    gt_extracted_answers: Dict[str, str] = Field(default_factory=dict) 
    pred_extracted_answers: Dict[str, str] = Field(default_factory=dict)
    
    
    # === 2. Analyzer 产出 ===
    gt_type: Optional[QuestionType] = None
    gt_answer_truth: Optional[str] = None # GT 的真值 (如 "C", "50")
    gt_structure_map: GTStructureMap = Field(default_factory=dict)  # 完整的题号-题型映射
    
    # === 3. Aligner 产出 ===
    alignment_found: bool = False
    pred_segment: Optional[str] = None       # 截取出的完整 Pred 片段 (含手写答案) - 第一个题目
    pred_raw_handwritten: Optional[str] = None # 识别出的手写答案字符串 - 第一个题目
    pred_is_rejected: bool = False             # 是否拒答 - 第一个题目
    alignment_anchors: Optional[Dict[str, str]] = None  # 对齐锚点信息 {start_anchor, end_anchor, gt_start, gt_end} - 第一个题目
    all_question_segments: Optional[List[Dict]] = None  # 所有题目的匹配信息列表
    alignment_stats: Optional[Dict[str, int]] = None  # 对齐统计信息 {total_aligned, exact_match_count, fuzzy_match_count, total_questions}
    
    # === 4. Evaluator 产出 (最终指标) ===
    metrics: Dict[str, float] = Field(default_factory=lambda: {
        "stem_sim": 0.0, # 题目主干相似度（所有题目的平均值，排除GT被拒答的题目）
        "img_sim": 0.0,  # 图片相似度
        "ans_acc": 0.0,  # 答案准确率（所有选择题和填空题的平均值）
        "answer_extraction_acc": 0.0,  # 答案提取准确率（对比提取的答案标签）
        "gt_rejected_flag": 0.0,  # GT整体是否被拒答
        "pred_rejected_flag": 0.0,  # Pred整体是否被拒答
        "final_score": 0.0,
        # 混淆矩阵
        "tp": 0.0,  # True Positive: GT无法识别，Pred也拒答
        "fp": 0.0,  # False Positive: GT正常，但Pred拒答或未匹配
        "tn": 0.0,  # True Negative: GT正常，Pred也正常
        "fn": 0.0,  # False Negative: GT无法识别，但Pred正常（不应该发生）
    })