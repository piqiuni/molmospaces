# Interactive-Nav-SG-nav

基于场景图（Scene Graph）的交互式导航系统，集成 AI2-THOR 仿真器和 ROS。

## 项目结构

```
Interactive-Nav-SG-nav/
├── src/
│   ├── SG_Nav_pkg/              # SG-Nav 核心模块
│   │   ├── ai2SG.py            # ROS 节点：场景图构建
│   │   ├── SG_Nav.py           # Habitat 导航脚本
│   │   ├── scenegraph.py       # 场景图核心模块
│   │   └── utils/              # 工具模块
│   │       ├── image_process.py
│   │       ├── utils_glip.py
│   │       ├── utils_fmm/      # FMM 规划器
│   │       └── utils_scenegraph/ # 场景图工具
│   └── ai2thor_pkg/            # AI2-THOR ROS 桥接
│       └── script/
│           ├── ai2thor_ros.py  # AI2-THOR ROS 主节点
│           ├── thor_sensor.py
│           └── thor_controller.py
├── data/
│   └── models/                 # 模型文件目录
│       ├── sam_vit_h_4b8939.pth
│       ├── groundingdino_swint_ogc.pth
│       └── ollama/             # Ollama 模型存储目录
├── GLIP/                       # GLIP 模型
├── GroundingDINO/              # GroundingDINO 模型
├── segment_anything/           # SAM 模型
└── tools/                      # 工具和数据文件
```

## 安装步骤

### Step 1: 环境配置

项目需要两个 conda 环境：

**1. 创建 SG_Nav 环境（用于场景图构建）：**
```bash
conda create -n SG_Nav python==3.9
conda activate SG_Nav
```

**2. 创建 smartllm 环境（用于 AI2-THOR ROS 桥接）：**
```bash
conda create -n smartllm python==3.9
conda activate smartllm
pip install -r requirements_smartllm.txt


### Step 2: 安装依赖包

**在 SG_Nav 环境中安装依赖：**
```bash
conda activate SG_Nav
conda install -c pytorch faiss-gpu=1.8.0
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

安装 PyTorch3D（推荐使用预编译版本）：
```bash
# 方式 1: 从 Anaconda 下载（推荐）
# https://anaconda.org/pytorch3d/pytorch3d/0.7.2/download

# 方式 2: 从源码安装
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Step 3: 安装 Grounded SAM

**在 SG_Nav 环境中安装：**
```bash
conda activate SG_Nav
pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO

# 下载模型权重
wget -O data/models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -O data/models/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### Step 4: 安装 GLIP 模型

**在 SG_Nav 环境中安装：**
```bash
conda activate SG_Nav
cd GLIP
python setup.py build develop --user
mkdir -p MODEL
cd MODEL
wget https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth
cd ../../
```

### Step 5: 安装 Ollama

安装 Ollama 服务：
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

下载模型到项目目录（推荐）：
```bash
cd /home/wxy/Downloads/Interactive-Nav-SG-nav
OLLAMA_MODELS="$(pwd)/data/models/ollama" ollama pull llama3.2-vision
```

### Step 6: 安装 BERT 模型

从官方下载 `bert-base-uncased`
cd
git lfs clone https://huggingface.co/google-bert/bert-base-uncased

```bash
# 搜索并替换路径
# local_bert_path="/home/xiaoji/project/bert-base-uncased"
# 替换为你的本地路径
```

## 使用方法

### 方式 1: ROS 节点模式（推荐）

**终端 1 - 启动 ROS 核心：**
```bash
roscore
```

**终端 2 - 启动 AI2-THOR ROS 桥接：**
```bash
conda activate smartllm
cd /home/wxy/Downloads/Interactive-Nav-SG-nav
source ./devel/setup.bash
python src/ai2thor_pkg/script/ai2thor_ros.py
```

**终端 3 - 启动 struct_mapping_pkg：**
```bash
cd /home/wxy/Downloads/Interactive-Nav-SG-nav
source ./devel/setup.bash
roslaunch struct_mapping_pkg slam_gmapping.launch
```

**终端 4 - 启动 semantic_mapping_pkg：**
```bash
cd /home/wxy/Downloads/Interactive-Nav-SG-nav
source ./devel/setup.bash
roslaunch semantic_mapping_pkg semantic_mapping.launch
```

```
cd /home/wxy/Downloads/Interactive-Nav-SG-nav
source ./devel/setup.bash
roslaunch explore_pkg explore_manager.launch


cd /home/wxy/Downloads/Interactive-Nav-SG-nav
source ./devel/setup.bash
roslaunch nav_pkg nav.launch
```



### 2. 启动 BLIP 服务器（如果使用 client 模式）

```bash
cd /home/wxy/Downloads/Interactive-Nav-SG-nav
cd src/semantic_mapping_pkg
bash start_blip.sh
```

### 3. 配置参数

编辑 `config/default.yaml` 设置：
- ROS 话题名称
- 建图参数（体素大小、阈值等）
- 功能开关（物体建图、场景建图）
- BLIP 服务地址（如果使用 client 模式）

### 4. 运行节点

```bash
# 单独启动语义建图节点
roslaunch semantic_mapping_pkg semantic_mapping.launch

# 或使用统一系统启动（包含其他包）
roslaunch semantic_mapping_pkg unified_system.launch

# 使用模拟模式（不需要 BLIP 服务器）
roslaunch semantic_mapping_pkg semantic_mapping.launch blip_mode:=mock
```
<!-- **终端 4 - 启动场景图构建节点：**
```bash
conda activate SG_Nav
cd /home/wxy/Downloads/Interactive-Nav-SG-nav
python src/SG_Nav_pkg/ai2SG.py
``` -->

**终端 5 - 键盘控制：**
```bash
conda activate smartllm
cd /home/wxy/Downloads/Interactive-Nav-SG-nav
   python src/ai2thor_pkg/script/keyboard_control.py
```
**终端 6 - rviz显示：**
```bash
cd /home/wxy/Downloads/Interactive-Nav-SG-nav
source ./devel/setup.bash
rviz -d $(rospack find ai2thor_pkg)/rviz/rviz.rviz
```

## 主要功能

- ✅ 将 AI2-THOR 输出的深度图、RGB 图统一打包到一个话题发布
- ✅ 写好了 SG-nav 的接口，可以实时读取话题图片信息并做出场景图关联
- ✅ 将模型检测到的 caption 投影到 SLAM 栅格地图上，实现语义地图构建
- [todo] 目前只涉及小尺度物体检测，需要加入房间属性检测
- [todo] 自主探索目标导航框架
- [todo] 场景交互后语义增量式更新

## 注意事项

1. **模型路径**：所有模型文件统一存储在 `data/models/` 目录下
2. **Ollama 模型**：可以通过 `OLLAMA_MODELS` 环境变量指定存储位置
3. **ROS 环境**：确保已正确配置 ROS 环境变量（`source /opt/ros/<version>/setup.bash`）
4. **Conda 环境**：
   - `smartllm` 环境：用于运行 `ai2thor_ros.py` 和 `keyboard_control.py`
   - `SG_Nav` 环境：用于运行 `ai2SG.py` 和 `SG_Nav.py`
5. **环境切换**：不同终端需要使用对应的 conda 环境
6. **Requirements 文件**：
   - `requirements.txt`：SG_Nav 环境的依赖
   - `requirements_smartllm.txt`：smartllm 环境的完整依赖列表

```