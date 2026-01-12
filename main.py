import logging
import os
import asyncio
from pathlib import Path
from threading import Lock
from pipeline import EvaluationPipeline
from config import PATH_PAIRS, OUTPUT_DIR
from utils.report_generator import ReportGenerator
import re
import time
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logging.warning("tqdm not installed, progress bar will be disabled. Install with: pip install tqdm")

# 配置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 线程安全的计数器
result_lock = Lock()
success_counter = 0
failed_counter = 0
failed_files = []  # 记录失败的文件信息

def load_data(gt_dir, pred_dir):
    # 1. 加载文件 (同时支持目录路径和单个文件路径)
    gt_path = Path(gt_dir)
    pred_path = Path(pred_dir)
    
    # 处理GT路径
    if gt_path.is_dir():
        gt_files = {p.name: (p.read_text(encoding='utf-8'), p) for p in gt_path.glob("*.md")}
        gt_type = "目录"
    else:
        # 单个文件
        gt_files = {gt_path.name: (gt_path.read_text(encoding='utf-8'), gt_path)}
        gt_type = "文件"
    
    # 处理PRED路径
    if pred_path.is_dir():
        pred_files = {p.name: (p.read_text(encoding='utf-8'), p) for p in pred_path.glob("*.md")}
        pred_type = "目录"
    else:
        # 单个文件
        pred_files = {pred_path.name: (pred_path.read_text(encoding='utf-8'), pred_path)}
        pred_type = "文件"
    
    logging.info(f"加载数据: GT{gt_type}={gt_dir}, GT文件数={len(gt_files)}, Pred{pred_type}={pred_dir}, Pred文件数={len(pred_files)}")
    
    dataset = []
    matched_gt_names = set()
    matched_pred_names = set() 
    unmatched_gt = []
    unmatched_pred = []

    # 辅助函数：提取核心 ID
    def get_core_id(filename):
        stem = str(Path(filename).stem)
        # 移除常见的后缀变体
        if "_upright" in stem:
            return stem.split("_upright")[0]
        if "_proofread" in stem:
             return stem.split("_proofread")[0]
        if "_result" in stem:
            return stem.split("_result")[0]
        return stem

    # 建立反向索引 (为了让 GT短名 能匹配到 Pred长名)
    pred_lookup = {}
    for original_pred_name in pred_files.keys():
        core_id = get_core_id(original_pred_name)
        pred_lookup[core_id] = original_pred_name

    # 2. 遍历 GT 进行匹配
    for gt_name, (gt_content, gt_path) in gt_files.items():
        gt_core_id = get_core_id(gt_name)
        target_pred_name = None
        
        # 匹配逻辑
        if gt_name in pred_files:
            target_pred_name = gt_name
        elif gt_core_id in pred_lookup:
            target_pred_name = pred_lookup[gt_core_id]
            
        if target_pred_name:
            pred_content, pred_path = pred_files[target_pred_name]
            
            # --- 【改动点】 ---
            # 删除了之前在这里写的 re.sub / replace 等修改 content 的代码。
            # 既然 Preprocessor 已经能智能识别当前目录下的图片，
            # 这里直接传入原始的 gt_content 即可。
            
            dataset.append((gt_name, gt_content, pred_content, gt_path, pred_path))
            matched_gt_names.add(gt_name)
            matched_pred_names.add(target_pred_name)
        else:
            unmatched_gt.append(gt_name)
    
    # 3. 统计
    for name in pred_files.keys():
        if name not in matched_pred_names:
            unmatched_pred.append(name)
            
    logging.info(f"匹配的文件对数: {len(dataset)}")
    return dataset

def generate_report(results, failed_files_list=None):
    """
    生成完整的对比报告
    """
    report_generator = ReportGenerator(OUTPUT_DIR)
    report_generator.save_reports(results, failed_files_list)
    logging.info(f"报告已保存到: {OUTPUT_DIR}")

async def process_single_file_async(pipeline, filename, gt_raw, pred_raw, gt_file_path, pred_file_path):
    """
    异步处理单个文件对
    """
    global success_counter, failed_counter, failed_files
    try:
        # 运行单个评估案例（异步）
        final_context = await pipeline.run_single_case_async(
            filename=filename,
            gt_raw=gt_raw,
            pred_raw=pred_raw,
            gt_file_path=gt_file_path,
            pred_file_path=pred_file_path
        )
        
        # 检查是否对齐失败（特别是超时导致的失败）
        # 即使对齐失败，只要gt_structure_map存在（包括空字典），也应该包含在统计中（用于统计题目数）
        # 如果对齐失败且没有找到任何题目，仍然返回context以便统计题目数
        if not final_context.alignment_found and not final_context.all_question_segments:
            with result_lock:
                failed_counter += 1
                # 记录失败信息
                failure_info = {
                    "filename": filename,
                    "failure_type": "alignment_failed",
                    "reason": "未找到任何对齐的题目",
                    "gt_structure_map_exists": final_context.gt_structure_map is not None,
                    "gt_structure_map_size": len(final_context.gt_structure_map) if final_context.gt_structure_map else 0,
                    "gt_file_path": str(gt_file_path),
                    "pred_file_path": str(pred_file_path)
                }
                failed_files.append(failure_info)
            logging.warning(f"处理失败（对齐失败）: {filename} - 未找到任何对齐的题目")
            # 即使对齐失败，如果gt_structure_map存在（即使是空字典），仍然返回context以便统计题目数
            # 注意：空字典 {} 在布尔上下文中为 False，但我们需要检查它是否存在（不是 None）
            if final_context.gt_structure_map is not None:
                # 如果 gt_structure_map 为空字典，说明 analyzer 没有识别到任何题目
                if len(final_context.gt_structure_map) == 0:
                    logging.warning(f"文件 {filename} 的 gt_structure_map 为空，analyzer 未识别到任何题目")
                return final_context
            return None
        
        with result_lock:
            success_counter += 1
        return final_context
    except Exception as e:
        with result_lock:
            failed_counter += 1
            # 记录异常信息
            import traceback
            error_type = type(e).__name__
            error_msg = str(e)
            tb = traceback.format_exc()
            
            # 检查是否是常见的API错误
            is_timeout = "Timeout" in error_type or "timeout" in error_msg.lower()
            is_rate_limit = "RateLimit" in error_type or "429" in error_msg or "rate limit" in error_msg.lower()
            is_connection = "Connection" in error_type or "connection" in error_msg.lower()
            
            failure_info = {
                "filename": filename,
                "failure_type": "exception",
                "reason": error_msg,
                "exception_type": error_type,
                "traceback": tb,
                "gt_file_path": str(gt_file_path),
                "pred_file_path": str(pred_file_path),
                "is_timeout": is_timeout,
                "is_rate_limit": is_rate_limit,
                "is_connection_error": is_connection
            }
            failed_files.append(failure_info)
        
        # 根据错误类型提供不同的日志级别和建议
        if is_timeout:
            logging.error(
                f"处理失败（超时）: {filename}: {error_type}: {error_msg}\n"
                f"建议：增加超时时间 - export LLM_TIMEOUT=120"
            )
        elif is_rate_limit:
            logging.error(
                f"处理失败（API限流）: {filename}: {error_type}: {error_msg}\n"
                f"建议：降低并发数 - export MAX_CONCURRENT=6 或增加延迟 - export REQUEST_DELAY=0.2"
            )
        elif is_connection:
            logging.error(
                f"处理失败（连接错误）: {filename}: {error_type}: {error_msg}\n"
                f"建议：检查网络连接或增加重试次数"
            )
        else:
            logging.error(f"处理失败（异常）: {filename}: {error_type}: {error_msg}", exc_info=True)
        return None

async def main_async():
    # 1. 初始化 Pipeline (一次即可)
    pipeline = EvaluationPipeline()
    
    # 2. 加载数据 - 遍历所有路径对
    if not PATH_PAIRS:
        logging.warning("PATH_PAIRS 为空！请检查 config.py 中的路径配置。")
        logging.info("当前配置的路径对:")
        logging.info(f"  - 检查范围: image_1 到 image_10")
        logging.info(f"  - GT模板: {{data_dir}}/image_{{num}}/gt")
        logging.info(f"  - PRED模板: {{data_dir}}/image_{{num}}/glm-4.6_image{{num}}")
        logging.info("提示: 路径对只有在GT和PRED目录都存在时才会被添加。")
        return
    
    dataset = []
    for gt_dir, pred_dir in PATH_PAIRS:
        logging.info(f"\n处理路径对: GT={gt_dir}, PRED={pred_dir}")
        path_dataset = load_data(gt_dir, pred_dir)
        dataset.extend(path_dataset)
    
    if not dataset:
        logging.warning("没有找到任何匹配的文件对！")
        logging.info("可能的原因:")
        logging.info("  1. GT和PRED目录中没有匹配的.md文件")
        logging.info("  2. 文件名不匹配（需要相同的文件名或核心ID匹配）")
        logging.info("  3. 路径配置不正确")
        return
    
    # 3. 异步并发处理
    # 降低默认并发数，避免API限流（可根据API限流情况调整）
    # 大批量处理时建议降低并发数：export MAX_CONCURRENT=6
    max_concurrent = int(os.getenv("MAX_CONCURRENT", "4"))  # 默认4个并发（降低并发数，提高匹配质量），可通过环境变量调整
    
    # 可选：添加请求之间的延迟（秒），避免API限流
    # 设置为0表示无延迟，可以根据API限流情况调整（如0.1表示每个请求间隔0.1秒）
    # 大批量处理时建议增加延迟：export REQUEST_DELAY=0.1
    request_delay = float(os.getenv("REQUEST_DELAY", "0.05"))  # 默认0.05秒延迟（增加小延迟，提高匹配质量）
    
    logging.info(f"开始异步处理 {len(dataset)} 个文件对（最大并发数: {max_concurrent}, 请求延迟: {request_delay}秒）...")
    if len(dataset) > 100:
        logging.warning(
            f"检测到大批量处理（{len(dataset)}个文件），建议：\n"
            f"  1. 降低并发数：export MAX_CONCURRENT=6\n"
            f"  2. 增加请求延迟：export REQUEST_DELAY=0.1\n"
            f"  3. 增加超时时间：export LLM_TIMEOUT=120\n"
            f"  4. 增加重试次数：export LLM_MAX_RETRIES=5"
        )
    start_time = time.time()
    
    # 使用 asyncio.Semaphore 控制并发数
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(filename, gt_raw, pred_raw, gt_file_path, pred_file_path):
        async with semaphore:
            # 如果设置了延迟，在请求前等待
            if request_delay > 0:
                await asyncio.sleep(request_delay)
            return await process_single_file_async(
                pipeline, filename, gt_raw, pred_raw, gt_file_path, pred_file_path
            )
    
    # 创建所有任务
    tasks = [
        process_with_semaphore(filename, gt_raw, pred_raw, gt_file_path, pred_file_path)
        for filename, gt_raw, pred_raw, gt_file_path, pred_file_path in dataset
    ]
    
    # 并发执行所有任务（带进度条）
    all_results = []
    
    if HAS_TQDM:
        # 使用 tqdm 显示进度
        pbar = tqdm(total=len(tasks), desc="处理文件", unit="文件")
        processed_count = 0  # 已处理的任务数
        
        # 使用 asyncio.as_completed 来获取完成的任务
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                processed_count += 1
                if result is not None:
                    all_results.append(result)
                pbar.update(1)
                # 显示统计信息：成功数、失败数、已处理数
                pbar.set_postfix({
                    "成功": success_counter,
                    "失败": failed_counter,
                    "已处理": processed_count
                })
            except Exception as e:
                processed_count += 1
                pbar.update(1)
                logging.error(f"处理文件时发生异常: {e}", exc_info=True)
                pbar.set_postfix({
                    "成功": success_counter,
                    "失败": failed_counter,
                    "已处理": processed_count
                })
        
        pbar.close()
    else:
        # 没有tqdm，直接使用asyncio.gather
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        # 处理异常
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                filename = dataset[i][0]
                logging.error(f"处理 {filename} 时发生异常: {result}", exc_info=True)
    
    elapsed_time = time.time() - start_time
    logging.info(f"处理完成: 成功={success_counter}, 失败={failed_counter}, 总计={len(all_results)}")
    if len(dataset) > 0:
        logging.info(f"总耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟), 平均每个文件: {elapsed_time/len(dataset):.2f}秒")
    else:
        logging.warning(f"总耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟), 但没有处理任何文件")

    # 4. 生成最终报告（包含失败文件信息）
    generate_report(all_results, failed_files)

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main_async())