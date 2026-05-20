# Drilling-Parameter-Optimization-under-Data-Incomplete-Conditions-in-Deep-Sea-Operations-

A Hybrid Framework Combining Historical Case Retrieval and Conditional Generation for Drilling Parameter Optimization under Data Incomplete Conditions in Deep-Sea Operations

## 简介
本仓库实现了一个用于深海钻井参数优化的混合框架，包含：

- 模糊检索子模块（基于 HNSW++ 分支索引）用于从历史方案中检索相似案例（HNSW++-CHA.py）。
- 条件扩散生成子模块（基于改进扩散模型）用于在部分观测（缺失）条件下生成补全的钻井方案（CDMsDDIM.py）。
- 多指标评价子模块（基于 AHP + 熵权 + TOPSIS）用于对方案进行打分与可视化（topsis.py）。

本仓库所有源文件均为独立的 Python 源文件（非压缩包），符合不上传压缩文件的要求。

## 仓库结构（主要文件）

- `topsis.py`：AHP + 熵权 + TOPSIS 的实现，包含示例数据和雷达图可视化，适合作为快速测试用例（无需外部数据文件）。
- `HNSW++-CHA.py`：基于 hnswlib 的分支 HNSW 索引与模糊匹配示例，示例代码在 `if __name__ == "__main__"` 中。该示例默认从本地 Excel 读取数据，请根据实际路径修改 `excel_path`。
- `CDMsDDIM.py`：条件扩散生成器（训练 + 生成 + 测试），包含示例主函数。示例依赖 Excel 数据文件，请根据实际路径修改 `excel_path`。
- `README.md`：本文件。

## 依赖

建议在虚拟环境中安装：

pip install numpy pandas scikit-learn matplotlib openpyxl

对于扩散模型需要：

pip install torch torchvision

若要运行 HNSW 部分：

pip install hnswlib

（请根据你的 Python 和 CUDA 环境选择合适的 torch 版本：https://pytorch.org/ ）

## 快速上手 / 快速测试

1. 克隆仓库：

   git clone https://github.com/nameshou/Drilling-Parameter-Optimization-under-Data-Incomplete-Conditions-in-Deep-Sea-Operations-.git
   cd Drilling-Parameter-Optimization-under-Data-Incomplete-Conditions-in-Deep-Sea-Operations-

2. 创建并激活虚拟环境（可选但推荐）：

   python -m venv .venv
   source .venv/bin/activate  # Linux / macOS
   .\.venv\\Scripts\\activate  # Windows PowerShell

3. 安装依赖（示例）：

   pip install -r requirements.txt

如果仓库中没有 requirements.txt，请使用上面“依赖”小节列出的命令安装必要包。

4. 运行快速测试（推荐）：

   python topsis.py

该脚本包含内置示例数据，会：
- 计算 AHP 主观权重、熵权客观权重与乘法融合权重；
- 运行 TOPSIS 对样例方案打分并打印排名；
- 生成并保存若干雷达图 (radar_group*.png)，并在窗口显示可视化结果。

此测试不依赖 Excel 文件或其它外部数据，适合作为“至少包含一个快速测试或示例文件”的证明。

## 运行 HNSW++（模糊匹配）示例

HNSW++ 示例默认在 `HNSW++-CHA.py` 的 main 段里从 Excel 读取数据：

- 编辑 `excel_path`，将其指向你的数据文件（支持 Excel 格式）。
- 运行：

    python "HNSW++-CHA.py"

注意：若运行时报错找不到 hnswlib，请先安装 `pip install hnswlib`。

## 运行扩散生成器示例（CDMsDDIM.py）

- 编辑 `CDMsDDIM.py` 中的 `excel_path`，指向你的数据文件（例如 .xlsx、.xlsm）。
- 如果你没有 GPU 或不想使用 CUDA，脚本会自动选择 CPU（device='cpu'）。若要使用 GPU，请确保安装了带 CUDA 支持的 PyTorch 并且机器上有可用 GPU。
- 运行：

    python CDMsDDIM.py

该脚本会：
- 读取 Excel 中的方案数据；
- 构建并训练条件扩散模型（训练参数在脚本中可配置）；
- 生成若干补全的方案并打印输出；
- 可以通过 `measure_inference_time` 测量平均推理时间。

注意：当前示例使用绝对路径（例如 E:\\\\tool\\\\python project\\\\test\\\\钻速机器学习.xlsm）。请修改为仓库内的相对路径或将数据放到合适位置并更新路径以便他人复现。

## 安全与发布注意事项

- 仓库中未包含压缩源码包（如 .zip/.rar/.7z）。
- 请确保不上传含有敏感信息（密码、私钥、API Key 或个人隐私数据）的数据文件到公共仓库。
- 若你希望公开示例数据，建议先对数据进行脱敏或仅发布小规模、合成的数据样例。

## 建议的下一步（可选）

- 添加一个 `examples/` 目录，放置：
  - 一个小型示例数据文件（例如 `examples/sample_schemes.xlsx`），并修改 HNSW++ 和 CDMsDDIM 的示例代码以默认加载该相对路径；
  - 一个 `quick_test.py` 脚本，调用 topsis、HNSW++、CDMsDDIM 的小型端到端流程，作为自动化示例；
  - 一个 `requirements.txt` 文件列出可复现的依赖版本。 

- 将 `excel_path` 等硬编码路径替换为命令行参数或配置文件，便于他人复现与 CI 测试。

## 许可（License）

请在仓库中添加合适的 LICENSE 文件以明确源代码许可（例如 MIT、Apache-2.0 等），如果你愿意将代码开源，请选择并添加许可证。

---

如果你希望，我可以：

- 同时为你创建 `requirements.txt`、添加 `examples/sample_schemes.xlsx`（合成小样例）和一个 `quick_test.py` 来满足所有要求的演示；
- 或者我可以只更新 README（我已完成）并等待你决定是否需要我提交示例数据与测试脚本。
