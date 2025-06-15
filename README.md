# Learning to Optimize Multigrid PDE Solvers  

## 基于深度学习的多网格偏微分方程求解器优化

论文：[Learning to Optimize Multigrid PDE Solvers (arXiv:1902.10248)](https://arxiv.org/abs/1902.10248)  

本研究提出了一个用于学习多重网格求解器的创新框架，旨在解决偏微分方程(PDE)数值求解中的关键挑战。多重网格方法作为解决大规模PDE的领先技术，其核心是延拓矩阵，它连接问题的不同尺度。这种矩阵高度依赖于具体问题，其最优构造对求解器效率至关重要，但为新问题设计多重网格算法通常极具挑战性。该框架学习从参数化PDE族到延拓算子的单一映射，通过高效且无监督的损失函数对整个PDE类别进行一次性训练。在广泛的二维扩散问题类别上的实验表明，与广泛使用的黑盒多重网格方案相比，该方法实现了更快的收敛速率，证明其成功学习了构造延拓矩阵的规则。
 
## 核心思想与贡献

###  主要思想
- **局部傅里叶分析（LFA）驱动训练**：使用 LFA 指导损失函数的设计，确保模型在频域上表现最优。
- **单次训练，通用于整个 PDE 类别**：训练一个神经网络即可处理所有来自某一分布的扩散方程，而非针对单一问题训练。
- **无监督学习**：无需真实标签或精确解，仅依靠代数性质（如误差传播矩阵的谱半径）进行训练。
- **高效的块 Fourier 分析加速训练**：通过构造 block-circulant 离散矩阵，大幅降低训练复杂度。

### 创新点
- 提出一种端到端的学习策略，将多网格求解器的关键组件（插值算子）交由神经网络自动学习；
- 在多种实验设置下（包括 Dirichlet 边界条件、非均匀系数分布、大网格等）均优于经典的 Black Box 多网格方法；
- 展示了模型对不同尺度、不同边界条件以及不同 PDE 系数分布的良好泛化能力。


## 目录结构
```
MULTIGRIDSOLVERS_PADDLEPADDLE/
│── geometric_solver.py       # 几何多网格求解器的实现
├── model_dirichletBC.py      # Dirichlet 边界条件下的插值网络模型
├── model_periodicBC.py       # 周期性边界条件下的插值网络模型
├── PaddlePaddle.ipynb        # Jupyter Notebook，用于快速演示和调试
├── README.md                 # 项目说明文档
├── test.py                   # 测试脚本，评估模型性能
├── training.py               # 训练脚本，包含主训练逻辑
└── utils.py                  # 辅助工具模块，提供 LFA 分析、矩阵生成等功能
```


## 快速开始

### 1. 运行训练脚本
```bash
python training.py
```
- 默认情况下，程序会根据配置文件中的参数启动训练流程。
- 训练过程中会保存模型权重和优化器状态，并记录日志到 TensorBoard。

训练大约需要20h完成

### 2. 运行测试脚本
```bash
python test.py
```

### 或者使用 Jupyter Notebook
`PaddlePaddle.ipynb`
