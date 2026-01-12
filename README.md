# MathDoc: Benchmarking Structured Extraction and Active Refusal on Noisy Mathematics Exam Papers

一个用于评估题目识别、对齐和答案准确率的综合评估系统。支持比较 Ground Truth (GT) 和 Prediction (Pred) 文件，生成详细的评估报告。

## 功能特性

- **多题型支持**：支持选择题 (Choice)、填空题 (Fill)、解答题 (Solve)
- **多维度评估**：
  - 题目文本相似度
  - 图片相似度
  - 拒答指标(precision,recall,F1)
- **智能对齐**：使用 LLM 进行题目对齐，支持直接匹配、模糊匹配和 Fallback 匹配
- **详细报告**：生成 JSON 和文本格式的评估报告，包含整体统计和按题型统计
- **批量处理**：支持批量处理多个文件或目录

## 项目结构

```
Benchmark/
├── main.py              # 主程序入口
├── pipeline.py          # 评估流水线
├── config.py            # 配置文件
├── core/                # 核心模块
│   ├── schema.py        # 数据模型定义
│   ├── preprocessor.py  # 数据预处理
│   ├── analyzer.py      # GT 分析器
│   ├── aligner.py       # 题目对齐器
│   └── evaluator.py     # 评估器
├── utils/               # 工具模块
│   ├── llm_client.py    # LLM 客户端
│   ├── report_generator.py  # 报告生成器
│   ├── text_metric.py   # 文本相似度计算
│   ├── image_sim.py     # 图片相似度计算
│   └── prompt.py        # Prompt 模板
├── data/                # 数据目录
└── output/              # 输出目录
    ├── reports/         # 评估报告
    └── segments/        # 分割的题目片段
```

## 安装

### 环境要求

- Python 3.8+
- 所需的 Python 包（见 `requirements.txt`）

### 依赖安装

```bash
pip install -r requirements.txt
```

### 环境变量配置

在项目根目录创建 `.env` 文件，配置 API 密钥：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

## 数据配置

### 数据说明

**注意**：图片数据和正确答案（Ground Truth）数据会过一段时间上传到仓库。目前仓库中只包含评估代码和配置文件。

### 数据目录结构

**重要**：GT（Ground Truth）和 Pred（Prediction）文件都需要放在 `data/` 目录下进行比较。

推荐的数据目录结构如下：

```
Benchmark/
└── data/
    ├── image_1/
    │   ├── gt/                    # GT文件目录
    │   │   ├── file1.md
    │   │   └── file2.md
    │   └── qwen_8b_image1/        # Pred文件目录（示例：qwen模型输出）
    │       ├── file1.md
    │       └── file2.md
    ├── image_2/
    │   ├── gt/
    │   └── qwen_8b_image2/
    └── ...
```

### 文件组织方式

- **GT文件**：Ground Truth（标准答案）文件，通常放在 `data/image_{num}/gt/` 目录下
- **Pred文件**：模型预测输出文件，放在 `data/image_{num}/{model_name}/` 目录下
- 支持 **目录** 和 **单个文件** 两种方式：
  - **目录方式**：目录下包含多个 `.md` 文件，系统会自动匹配同名文件
  - **单文件方式**：直接指定单个文件路径

### 文件格式说明

- **文件格式**：GT 和 Pred 文件都使用 Markdown（`.md`）格式
- **图片支持**：Markdown 文件可以包含图片，支持以下格式：
  - Markdown 图片语法：`![alt text](image_path)`
  - HTML 图片标签：`<img src="image_path" alt="alt text">`
- **图片路径要求**：
  - **重要**：如果使用相对路径，图片文件必须和 Markdown 文件放在**同一目录**下
  - 图片路径支持相对路径和绝对路径，系统会自动将相对路径转换为绝对路径（以 Markdown 文件所在目录为基准）
  - 示例：如果 Markdown 文件在 `data/image_1/gt/file1.md`，则相对路径的图片应在 `data/image_1/gt/` 目录下
- **图片处理**：系统会提取文件中的图片并计算图片相似度，用于评估模型输出的图片识别准确性

### 在 config.py 中配置路径对

编辑 `config.py` 文件，使用 `generate_path_pairs()` 函数配置 GT 和 Pred 的路径对：

```python
PATH_PAIRS = generate_path_pairs(
    start=1,       # 起始数字（如 image_1）
    end=10,        # 结束数字（如 image_10）
    gt_template="{data_dir}/image_{num}/gt",  # GT路径模板
    pred_template="{data_dir}/image_{num}/qwen_8b_image{num}"  # Pred路径模板
)
```

**路径模板说明**：
- `{data_dir}`：会自动替换为 `Benchmark/data/`
- `{num}`：会自动替换为对应的数字（start 到 end）

**示例配置**（不同模型）：

```python
# 示例1：评估 qwen 8b 模型
PATH_PAIRS = generate_path_pairs(
    start=1,
    end=10,
    gt_template="{data_dir}/image_{num}/gt",
    pred_template="{data_dir}/image_{num}/qwen_8b_image{num}"
)

# 示例2：评估 Intern 8b 模型
PATH_PAIRS = generate_path_pairs(
    start=1,
    end=10,
    gt_template="{data_dir}/image_{num}/gt",
    pred_template="{data_dir}/image_{num}/Intern_8b_imag{num}"
)

# 示例3：只处理单个数据集（如 image_3）
PATH_PAIRS = generate_path_pairs(
    start=3,
    end=3,  # start 和 end 相同即只处理单个数据集
    gt_template="{data_dir}/image_{num}/gt",
    pred_template="{data_dir}/image_{num}/qwen_8b_image{num}"
)
```

**注意事项**：
- 只有 GT 和 Pred 路径都存在时，才会被添加到路径对中
- 路径对会自动检查目录是否存在，不存在的路径会被跳过
- 确保 GT 和 Pred 目录下的 `.md` 文件名能够匹配（支持文件名变体匹配，如 `file.md` 和 `file_upright.md`）

## 配置

在 `config.py` 中还可以配置：

1. **模型配置**：
   - `VISION_MODEL`：用于图片相似度评估的视觉模型（默认：qwen3-vl-plus）
   - `TEXT_MODEL`：用于文本处理的模型（默认：qwen3-max）
2. **输出目录**：`OUTPUT_DIR` 指定报告输出位置（默认：`output/`）

## 使用方法

### 基本使用

```bash
python main.py
```

程序将：
1. 加载配置的路径对
2. 遍历所有 GT 和 Pred 文件
3. 执行评估流程
4. 生成评估报告

### 评估流程

1. **预处理**：清理文本，提取图片链接，提取答案
2. **GT 分析**：分析 GT 文件结构，识别题型和题目编号
3. **题目对齐**：将 Pred 文件中的题目与 GT 中的题目进行对齐
4. **评估**：计算各项评估指标
5. **报告生成**：生成详细的评估报告

## 输出报告

评估报告保存在 `output/reports/` 目录下：

- `summary_report.txt`：文本格式的汇总报告
- `summary_report.json`：JSON 格式的汇总报告
- `detailed_reports.json`：每个文件的详细评估报告
- `failed_files_report.txt`：处理失败的文件报告（如果有）

### 报告内容

- **整体统计**：总题目数、成功对齐数、各种匹配方式统计
- **评估指标**：文本相似度、图片相似度、拒答指标（Precision, Recall, F1）等
- **混淆矩阵**：TP、FP、TN、FN 统计
- **按题型统计**：每种题型的详细统计信息

## 注意事项
- 确保 API 密钥配置正确，否则 LLM 调用会失败
- 处理大量文件时可能需要较长时间，建议使用进度条显示（需要安装 `tqdm`）


