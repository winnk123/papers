# 调用api计算图片相似度
# utils/image_metric.py
import base64
import logging
import numpy as np
from pathlib import Path
from typing import List
import re
from langchain_core.messages import HumanMessage
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.llm_client import LLMClient
from utils.prompt import PICTURE_ANALYSIS_PROMPT
from core.schema import ImageSimilarityResult

logger = logging.getLogger(__name__)

class ImageComparator:
    def __init__(self, vision_client: LLMClient):
        """
        初始化图片比较器
        :param vision_client: 预先配置好的、使用视觉模型 (如 qwen-vl-plus) 的 LLMClient
        """
        self.client = vision_client
        
        # 视觉模型的调用不依赖 LangChain 的 chain，而是直接构造 message
        # 因为输入格式（图片+文本）比较特殊
    
    def _image_to_base64(self, image_path: Path) -> str:
        """将图片文件转换为 Base64 编码的字符串"""
        if not image_path.exists():
            raise FileNotFoundError(f"图片文件未找到: {image_path}")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
        
    def compare_single_pair(self, img_path1: str, img_path2: str) -> float:
        """
        使用 VLM 对比一对图片的相似度（同步版本）
        """
        try:
            p1 = Path(img_path1)
            p2 = Path(img_path2)
            
            # 检查文件是否存在，如果不存在则返回 0.0
            if not p1.exists():
                logger.warning(f"图片文件未找到: {p1}，跳过对比")
                return 0.0
            if not p2.exists():
                logger.warning(f"图片文件未找到: {p2}，跳过对比")
                return 0.0
            
            # 构造 VLM 的输入 message
            message = HumanMessage(
                content=[
                    {"type": "text", "text": PICTURE_ANALYSIS_PROMPT},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{self._image_to_base64(p1)}"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{self._image_to_base64(p2)}"},
                ]
            )
            
            # 直接调用 client 底层的 llm 对象（同步）
            response = self.client.llm.invoke([message])
            
            return self._parse_similarity_response(response)

        except Exception as e:
            logger.error(f"对比图片失败: {e}")
            return 0.0
    
    async def compare_single_pair_async(self, img_path1: str, img_path2: str) -> float:
        """
        使用 VLM 对比一对图片的相似度（异步版本）
        """
        try:
            p1 = Path(img_path1)
            p2 = Path(img_path2)
            
            # 检查文件是否存在，如果不存在则返回 0.0
            if not p1.exists():
                logger.warning(f"图片文件未找到: {p1}，跳过对比")
                return 0.0
            if not p2.exists():
                logger.warning(f"图片文件未找到: {p2}，跳过对比")
                return 0.0
            
            # 构造 VLM 的输入 message
            message = HumanMessage(
                content=[
                    {"type": "text", "text": PICTURE_ANALYSIS_PROMPT},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{self._image_to_base64(p1)}"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{self._image_to_base64(p2)}"},
                ]
            )
            
            # 异步调用 client 底层的 llm 对象
            response = await self.client.llm.ainvoke([message])
            
            return self._parse_similarity_response(response)

        except Exception as e:
            logger.error(f"异步对比图片失败: {e}")
            return 0.0
    
    def _parse_similarity_response(self, response) -> float:
        """
        解析相似度响应的通用方法（同步和异步共用）
        """
        # 解析返回的 content (通常是字符串)
        raw_text = response.content.strip()
        
        # 尝试从返回的文本中提取 JSON
        import json
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            return ImageSimilarityResult(**data).similarity_score
        else:
            # 如果没找到 JSON，尝试直接解析数字
            num_match = re.search(r'(\d\.\d+)', raw_text)
            if num_match:
                return float(num_match.group(1))
            
        logger.warning("无法从 VLM 响应中解析相似度分数。")
        return 0.0


    def calculate_average_similarity(self, gt_img_paths: List[str], pred_img_paths: List[str], max_workers: int = 5) -> float:
        """
        同步版本：计算平均相似度
        """
        """
        以 GT 为基准，计算图片相似度。
        
        :param gt_img_paths: GT图片路径列表
        :param pred_img_paths: Pred图片路径列表
        :param max_workers: 并行处理图片对比的最大线程数（默认5，避免API限流）
        :return: 相似度分数（0.0-1.0），如果GT无图则返回-1.0（不适用）
        """
        
        # --- 规则 1: GT 无图 ---
        # 如果 GT 的图片列表为空，说明标准答案里就没有图。
        # 此时，图片相似度不适用，返回-1.0（不计入计算）
        if not gt_img_paths:
            return -1.0  # 不适用，不计入计算

        # --- 规则 2: GT 有图 ---
        # 只有执行到这里，才说明 GT 中至少有一张图片，评估必须进行。
        # 以GT的图片为基准计算相似度
        
        # 如果 Pred 完全没有提取出任何图片，说明全部丢失，直接 0 分。
        if not pred_img_paths:
            return 0.0

        # 准备需要对比的图片对
        pairs_to_compare = []
        for i in range(len(gt_img_paths)):
            gt_path = gt_img_paths[i]
            if i < len(pred_img_paths):
                pred_path = pred_img_paths[i]
                pairs_to_compare.append((i, gt_path, pred_path))
            else:
                # Pred 列表不够长，说明这张图在 Pred 中缺失了
                # 按照规则，缺失的图片得 0 分，直接添加到scores中
                pass
        
        # 如果没有需要对比的图片对，返回0.0
        if not pairs_to_compare:
            return 0.0
        
        # 并行处理图片对比（如果只有1-2张图片，串行处理可能更快）
        scores = [0.0] * len(gt_img_paths)  # 预分配，缺失的图片已经是0.0
        
        if len(pairs_to_compare) <= 2:
            # 图片数量少，串行处理
            for i, gt_path, pred_path in pairs_to_compare:
                score = self.compare_single_pair(gt_path, pred_path)
                scores[i] = score
        else:
            # 图片数量多，并行处理
            with ThreadPoolExecutor(max_workers=min(max_workers, len(pairs_to_compare))) as executor:
                future_to_index = {
                    executor.submit(self.compare_single_pair, gt_path, pred_path): i
                    for i, gt_path, pred_path in pairs_to_compare
                }
                
                for future in as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        score = future.result()
                        scores[i] = score
                    except Exception as e:
                        logger.error(f"图片对比失败 (索引 {i}): {e}")
                        scores[i] = 0.0

        # 计算最终平均分
        return float(np.mean(scores)) if scores else 0.0
    
    async def calculate_average_similarity_async(self, gt_img_paths: List[str], pred_img_paths: List[str], max_concurrent: int = 10) -> float:
        """
        异步版本：计算平均相似度
        
        :param gt_img_paths: GT图片路径列表
        :param pred_img_paths: Pred图片路径列表
        :param max_concurrent: 最大并发数（默认10，避免API限流）
        :return: 相似度分数（0.0-1.0），如果GT无图则返回-1.0（不适用）
        """
        # --- 规则 1: GT 无图 ---
        if not gt_img_paths:
            return -1.0  # 不适用，不计入计算

        # --- 规则 2: GT 有图 ---
        if not pred_img_paths:
            return 0.0

        # 准备需要对比的图片对
        pairs_to_compare = []
        scores = [0.0] * len(gt_img_paths)  # 预分配，缺失的图片已经是0.0
        
        for i in range(len(gt_img_paths)):
            gt_path = gt_img_paths[i]
            if i < len(pred_img_paths):
                pred_path = pred_img_paths[i]
                pairs_to_compare.append((i, gt_path, pred_path))
        
        if not pairs_to_compare:
            return 0.0
        
        # 使用 asyncio 的 Semaphore 控制并发数
        import asyncio
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def compare_with_semaphore(i, gt_path, pred_path):
            async with semaphore:
                pair_start_time = time.time()
                result = i, await self.compare_single_pair_async(gt_path, pred_path)
                pair_elapsed = time.time() - pair_start_time
                logger.debug(f"图片对比完成 [{i+1}/{len(pairs_to_compare)}]: {Path(gt_path).name} vs {Path(pred_path).name}, 耗时 {pair_elapsed:.2f}秒")
                return result
        
        # 并发执行所有图片对比
        logger.info(f"开始并发对比 {len(pairs_to_compare)} 对图片（最大并发数: {max_concurrent}）")
        import time
        batch_start_time = time.time()
        tasks = [compare_with_semaphore(i, gt_path, pred_path) for i, gt_path, pred_path in pairs_to_compare]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        batch_elapsed_time = time.time() - batch_start_time
        logger.info(f"所有图片对比完成，总耗时 {batch_elapsed_time:.2f} 秒")
        
        # 处理结果
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"图片对比任务失败: {result}")
                continue
            i, score = result
            scores[i] = score

        # 计算最终平均分
        return float(np.mean(scores)) if scores else 0.0