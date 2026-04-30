# 场景建图到14通道格式转换节点

## 功能说明

该节点将场景语义建图的输出转换为14通道格式，用于深度学习模型的输入。

## 14通道格式

根据 `channel_meaning.md` 文档：

- **通道0**: 占用图 (Occupancy Map) - 0=自由空间, 255=障碍物
- **通道1**: 探索掩码 (Explored Mask) - 0=未探索, 255=已探索
- **通道2**: 智能体位置 (Agent Position) - 0=不在, 255=在
- **通道3**: 智能体历史轨迹 (Agent History) - 0=未经过, >0=经过次数（带衰减）
- **通道4-13**: 10个语义类别概率图
  - 通道4: 客厅 (livingroom)
  - 通道5: 卧室 (bedroom)
  - 通道6: 厨房 (kitchen)
  - 通道7: 洗手间 (bathroom)
  - 通道8: 阳台 (balcony)
  - 通道9: 储藏间 (storage)
  - 通道10: 门 (door)
  - 通道11: 墙 (wall)
  - 通道12: 大门 (entrance)
  - 通道13: 外部区域 (outside)

## 输入话题

1. **占用地图** (`/struct_mapping/wall_occ_map`)
   - 类型: `nav_msgs/OccupancyGrid`
   - 用于生成通道0（占用图）和通道1（探索掩码）

2. **场景ID地图** (`/semantic_mapping/scene_id_grid`)
   - 类型: `nav_msgs/OccupancyGrid`
   - 由 `semantic_mapping_node` 发布
   - 包含每个栅格的场景ID

3. **场景置信度地图** (`/semantic_mapping/scene_confidence_grid`)
   - 类型: `nav_msgs/OccupancyGrid`
   - 由 `semantic_mapping_node` 发布
   - 包含每个栅格的置信度（0-100）

4. **TF变换** (通过 `base_link` -> `map`)
   - 用于获取智能体位置（通道2）和轨迹（通道3）

## 输出话题

节点会发布14个独立的 `nav_msgs/OccupancyGrid` 话题：

- `/semantic_mapping/14channel_map/channel_0` - 占用图
- `/semantic_mapping/14channel_map/channel_1` - 探索掩码
- `/semantic_mapping/14channel_map/channel_2` - 智能体位置
- `/semantic_mapping/14channel_map/channel_3` - 智能体历史轨迹
- `/semantic_mapping/14channel_map/channel_4` - 客厅概率图
- `/semantic_mapping/14channel_map/channel_5` - 卧室概率图
- `/semantic_mapping/14channel_map/channel_6` - 厨房概率图
- `/semantic_mapping/14channel_map/channel_7` - 洗手间概率图
- `/semantic_mapping/14channel_map/channel_8` - 阳台概率图
- `/semantic_mapping/14channel_map/channel_9` - 储藏间概率图
- `/semantic_mapping/14channel_map/channel_10` - 门概率图
- `/semantic_mapping/14channel_map/channel_11` - 墙概率图
- `/semantic_mapping/14channel_map/channel_12` - 大门概率图
- `/semantic_mapping/14channel_map/channel_13` - 外部区域概率图

## 使用方法

### 1. 启动节点

```bash
# 单独启动转换节点
roslaunch semantic_mapping_pkg scene_to_14channel.launch

# 或者指定参数
roslaunch semantic_mapping_pkg scene_to_14channel.launch \
    occupancy_grid_topic:=/struct_mapping/wall_occ_map \
    agent_frame:=base_link \
    map_frame:=map \
    publish_rate:=2.0
```

### 2. 配置场景ID到场景类型的映射（可选）

如果需要更精确的语义类别映射，可以通过参数配置场景ID到场景类型的映射：

```xml
<launch>
  <node name="scene_to_14channel_converter" ...>
    <rosparam param="scene_id_to_type_map">
      {"3": "kitchen", "4": "bedroom", "5": "livingroom"}
    </rosparam>
  </node>
</launch>
```

如果不配置，节点会使用启发式字符串匹配来推断场景类型。

### 3. 在Python代码中使用

```python
import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np

class ChannelSubscriber:
    def __init__(self):
        self.channels = [None] * 14
        for i in range(14):
            rospy.Subscriber(
                f"/semantic_mapping/14channel_map/channel_{i}",
                OccupancyGrid,
                lambda msg, idx=i: self.channel_callback(msg, idx),
                queue_size=1
            )
    
    def channel_callback(self, msg, idx):
        # 转换为numpy数组
        data = np.array(msg.data, dtype=np.int8)
        height = msg.info.height
        width = msg.info.width
        self.channels[idx] = data.reshape((height, width))
    
    def get_14channel_map(self):
        """获取14通道地图 (14, H, W)"""
        if all(ch is not None for ch in self.channels):
            return np.stack(self.channels, axis=0)  # (14, H, W)
        return None

# 使用示例
rospy.init_node('channel_subscriber')
sub = ChannelSubscriber()
rospy.spin()
```

## 注意事项

1. **数据同步**: 节点会等待所有输入数据（占用地图、场景ID地图、置信度地图）都可用后才进行转换。

2. **坐标系统一**: 确保所有输入地图使用相同的坐标系和分辨率。

3. **场景类型映射**: 如果场景类型字符串无法自动匹配到语义类别，可以通过 `scene_id_to_type_map` 参数手动配置。

4. **性能**: 节点默认以1Hz频率发布。如果地图较大，可以降低发布频率。

5. **轨迹衰减**: 智能体历史轨迹使用指数衰减，衰减因子为0.95。可以通过修改代码中的 `trajectory_decay` 参数调整。

## 依赖

- ROS (tested with ROS Noetic)
- numpy
- tf2_ros
- nav_msgs
- sensor_msgs
- geometry_msgs

## 相关文件

- `channel_meaning.md` - 14通道格式详细说明
- `scene_semantic_map.h` - 场景语义地图C++接口
- `semantic_mapping_node.cpp` - 语义建图节点（发布scene_id_grid和confidence_grid）

