# 检测信息JSON格式说明

## 话题信息
- **话题名称**: `/explore_agent/result_info` (可在config中配置)
- **消息类型**: `std_msgs::String`
- **内容**: JSON字符串

## JSON格式

### 基本结构
JSON消息必须是一个**数组**，包含多个检测对象：

```json
[
  {
    "position": {...},
    "size": {...},
    "class": "...",
    "confidence": 0.0-1.0,
    "instance_id": "..."  // 可选
  },
  ...
]
```

### 单个检测对象字段说明

#### 必需字段

1. **`position`** (对象)
   - 描述: 物体在**相机坐标系**中的3D位置（米）
   - 字段:
     - `x` (double): X坐标
     - `y` (double): Y坐标
     - `z` (double): Z坐标
   - 示例:
     ```json
     "position": {
       "x": 1.5,
       "y": 0.3,
       "z": 2.0
     }
     ```

2. **`size`** (对象)
   - 描述: 物体的尺寸（米）
   - 字段:
     - `x` (double): 长度
     - `y` (double): 宽度
     - `z` (double): 高度
   - 默认值: 如果缺失，默认为 `{x: 0.1, y: 0.1, z: 0.1}`
   - 示例:
     ```json
     "size": {
       "x": 0.5,
       "y": 0.3,
       "z": 0.8
     }
     ```

3. **`class`** 或 **`semantic_class`** (字符串)
   - 描述: 物体的语义类别名称
   - 示例: `"chair"`, `"table"`, `"door"`, `"person"` 等
   - 注意: 代码会先查找 `class`，如果不存在则查找 `semantic_class`

4. **`confidence`** (数字)
   - 描述: 检测置信度，范围 [0.0, 1.0]
   - 低于 `confidence_threshold` (默认0.5) 的检测会被忽略
   - 示例: `0.85`

#### 可选字段

5. **`instance_id`** (字符串)
   - 描述: 检测器提供的实例ID，用于更精确的数据关联
   - 如果提供，会优先使用实例ID进行匹配
   - 示例: `"chair_001"`, `"person_123"`

## 完整示例

### 示例1: 单个物体检测
```json
[
  {
    "position": {
      "x": 1.2,
      "y": 0.5,
      "z": 1.8
    },
    "size": {
      "x": 0.4,
      "y": 0.4,
      "z": 0.9
    },
    "class": "chair",
    "confidence": 0.92,
    "instance_id": "chair_001"
  }
]
```

### 示例2: 多个物体检测
```json
[
  {
    "position": {"x": 1.2, "y": 0.5, "z": 1.8},
    "size": {"x": 0.4, "y": 0.4, "z": 0.9},
    "semantic_class": "chair",
    "confidence": 0.92,
    "instance_id": "chair_001"
  },
  {
    "position": {"x": 2.0, "y": 0.0, "z": 1.5},
    "size": {"x": 1.2, "y": 0.6, "z": 0.75},
    "class": "table",
    "confidence": 0.88
  },
  {
    "position": {"x": 0.5, "y": -0.3, "z": 2.1},
    "size": {"x": 0.2, "y": 0.2, "z": 1.8},
    "semantic_class": "door",
    "confidence": 0.75
  }
]
```

### 示例3: 最小格式（使用默认值）
```json
[
  {
    "position": {"x": 1.0, "y": 0.0, "z": 1.5},
    "class": "object",
    "confidence": 0.6
  }
]
```
注意: 如果缺少 `size`，会使用默认值 `{x: 0.1, y: 0.1, z: 0.1}`

## 坐标系说明

- **`position`**: 必须在**相机坐标系**中
- 代码会自动通过TF将位置转换到世界坐标系（`map`）
- 确保 `camera_frame` 和 `world_frame` 之间的TF变换可用

## 数据关联逻辑

1. **优先级1**: 如果提供 `instance_id`，优先使用实例ID匹配
2. **优先级2**: 基于位置和语义类别匹配：
   - 位置距离 < `position_threshold` (默认0.5m)
   - 尺寸差异 < `size_threshold` (默认30%)
   - 语义类别相同
   - 在时间窗口内（`time_window`，默认5秒）

## 注意事项

1. JSON必须是有效的数组格式
2. 如果JSON解析失败，会输出警告但不会崩溃
3. 置信度低于阈值的检测会被忽略
4. 位置必须在相机坐标系中，代码会自动转换到世界坐标系
5. 时间戳由节点自动添加，不需要在JSON中提供

