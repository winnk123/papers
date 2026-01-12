import os
from pathlib import Path
from dotenv import load_dotenv

# 1. 加载 .env 文件 (如果存在)
# load_dotenv() 会自动寻找当前目录及上级目录下的 .env
load_dotenv()

# 2. 读取配置
# 这里不需要抛出错误，因为 LLMClient 会再次检查
API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 3. 路径配置
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"

# 智能生成路径对的函数
def generate_path_pairs(start=1, end=10, gt_template="{data_dir}/image_{num}/gt", pred_template="{data_dir}/image_{num}/qwen_8b_image{num}"):
    """
    自动生成路径对列表
    
    Args:
        start: 起始数字 (默认: 1)
        end: 结束数字 (默认: 10)
        gt_template: GT路径模板，支持 {data_dir} 和 {num} 占位符
        pred_template: PRED路径模板，支持 {data_dir} 和 {num} 占位符
        
    Returns:
        生成的路径对列表 [(gt_path1, pred_path1), (gt_path2, pred_path2), ...]
    """
    path_pairs = []
    for num in range(start, end + 1):
        # 构建GT路径
        gt_path = Path(gt_template.format(data_dir=str(DATA_DIR), num=num))
        # 构建PRED路径
        pred_path = Path(pred_template.format(data_dir=str(DATA_DIR), num=num))
        # 只有当路径存在时才添加到列表中
        if gt_path.exists() and pred_path.exists():
            path_pairs.append((gt_path, pred_path))
    return path_pairs

# 使用函数生成路径对列表
# 1. 生成单个文件夹路径对 (示例：只处理image_8)
PATH_PAIRS = generate_path_pairs(
    start=1,       # 起始数字
    end=10,        # 结束数字，与start相同即生成单个文件夹
    gt_template="{data_dir}/image_{num}/gt",  # GT路径模板
    pred_template="{data_dir}/image_{num}/qwen_8b_image{num}"  # PRED路径模板
    ## pred_template="{data_dir}/image_{num}/Intern_8b_imag{num}"  # PRED路径模板
    ## pred_template="{data_dir}/image_{num}/qwen_30b_image{num}" # PRED路径模板
    ## pred_template="{data_dir}/image_{num}/Reslt_gemini_image{num}" # PRED路径模板
    ## pred_template="{data_dir}/image_{num}/Paddle_result_{num}" # PRED路径模板
    ## pred_template="{data_dir}/image_{num}/glm_4.6_image{num}" # PRED路径模板（注意：实际目录名是下划线，不是连字符)
    ## pred_template="{data_dir}/image_{num}/dpsk_image{num}" # PRED路径模板
    ## pred_template="{data_dir}/image_{num}/Result_doubao_image{num}" # PRED路径模板
    ## pred_template="{data_dir}/image_{num}/Result_gpt_image{num}" # PRED路径模板
    ## pred_template="{data_dir}/image_{num}/Result_MinerU_{num}" # PRED路径模板
)

OUTPUT_DIR = PROJECT_ROOT / "output"

# 新增：图片资源目录配置
IMAGE_RESOURCE_DIR = Path("/home/chenyue001/papers/Benchmark/data/image1/Rangle_image1_stage2_match_out")

# 4. 模型配置
VISION_MODEL = "qwen3-vl-plus"
TEXT_MODEL = "qwen3-max"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-2a06f833aa4141a68eb30687ec7f0042")