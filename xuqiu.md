
---

# AI视频理解模块开发提示词

## 一、开发目标
**实现多模态视频内容解析系统**，要求：
1. 自动提取视频的视觉、听觉、时空特征
2. 生成结构化内容描述（JSON格式）
3. 输出内容摘要和分类标签

## 二、技术规范
```markdown
### 核心需求
- 输入：视频文件
- 输出：包含以下字段的JSON
  ```json
  {
    "metadata": {"duration": "", "resolution": ""},
    "visual": {
      "objects": [{"name": "", "confidence": 0.9}],
      "scene": {"label": "", "confidence": 0.9},
      "actions": [{"label": "", "time_range": []}]
    },
    "audio": {
      "transcript": "",
      "emotion": "",
      "sound_effects": []
    },
    "summary": ""
  }
  ```

### 性能指标
- 处理速度：<2分钟/10分钟视频（RTX 3060）
- 准确率要求：
  - 物体检测mAP@0.5 ≥ 0.65
  - 场景分类Top-1准确率 ≥ 0.8
  - 语音识别WER ≤ 0.25
```

## 三、模块开发提示

### 1. 视觉分析模块
```python
# 使用YOLOv8进行对象检测
def detect_objects(frame):
    """
    实现要求：
    - 输入：BGR格式的numpy数组
    - 输出：检测到的物体列表，按置信度降序排列
    - 模型：yolov8n.pt（预训练）
    - 阈值：confidence=0.5, iou=0.45
    """
    # [在此生成代码]

# 场景分类实现
class SceneClassifier:
    def __init__(self):
        """
        模型选择：
        - 使用ResNet50+Places365预训练模型
        - 输出365个场景分类
        """
    
    def classify(self, frame):
        # [在此生成预处理和推理代码]
```

### 2. 音频分析模块
```python
# 语音识别实现
def transcribe_audio(path):
    """
    要求：
    - 支持中英文自动识别
    - 输出带时间戳的文本
    - 使用Whisper medium模型
    """
    # [生成whisper异步处理代码]

# 音频事件检测
def detect_sound_events(waveform):
    """
    使用预训练的YAMNet模型：
    - 输出top5音频事件
    - 采样率16kHz
    - 输入长度0.96s
    """
    # [生成TensorFlow Lite推理代码]
```

### 3. 时空分析模块
```python
# 动作识别实现
class ActionRecognizer:
    def __init__(self):
        """
        模型选择：
        - SlowFast R50模型
        - 输入：32帧片段
        - 输出：Kinetics-400动作分类
        """
    
    def process_video(self, video_path):
        # [生成视频解码和时序采样代码]

# 镜头边界检测
def detect_shots(video_path):
    """
    要求：
    - 检测硬切和渐变转场
    - 使用OpenCV直方图差异法
    - 输出切换时间点列表
    """
    # [生成基于PySceneDetect的代码]
```

## 四、系统集成提示
```python
# 主处理流程
def analyze_video(video_path):
    # 1. 元数据提取
    cap = cv2.VideoCapture(video_path)
    metadata = {
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS),
        "resolution": f"{int(cap.get(3))}x{int(cap.get(4))}"
    }
    
    # 2. 关键帧采样（每5秒）
    key_frames = sample_key_frames(video_path, interval=5)
    
    # 3. 多线程分析
    with ThreadPoolExecutor() as executor:
        visual_future = executor.submit(analyze_visual, key_frames)
        audio_future = executor.submit(analyze_audio, extract_audio(video_path))
    
    # 4. 结果整合
    return {
        "metadata": metadata,
        "visual": visual_future.result(),
        "audio": audio_future.result(),
        "summary": generate_summary(...)
    }
```

## 五、测试案例
```python
# 测试用例1 - 足球比赛视频
输入：soccer_match.mp4
预期输出包含：
{
  "visual": {
    "objects": ["足球", "运动员", "球门"],
    "scene": "足球场",
    "actions": ["射门", "奔跑"] 
  },
  "audio": {
    "sound_effects": ["欢呼声", "哨声"],
    "emotion": "激动"
  }
}

# 测试用例2 - 烹饪教程
输入：cooking.mp4
预期包含：
{
  "visual": {"objects": ["菜刀", "西红柿"]},
  "audio": {"transcript": "首先将食材切丁..."}
}
```

## 六、开发注意事项
1. **内存管理**
   - 视频帧处理使用生成器避免内存溢出
   - 大模型按需加载（LRU缓存）

2. **错误处理**
   ```python
   try:
       analyze_video(invalid_path)
   except VideoProcessingError as e:
       logger.error(f"处理失败：{str(e)}")
       return {"error": "INVALID_FILE"}
   ```

3. **优化方向**
   - 对持续输入的视频流实现增量处理
   - 添加GPU显存监控和自动降级机制

---

将此文档输入Cursor后，可通过以下命令触发开发：
```bash
# 生成基础代码框架
/create_module visual_analysis.py --spec "视觉分析模块"

# 实现核心处理流程
/implement analyze_video() function with threading
```

建议开发顺序：
1. 物体检测 → 2. 语音识别 → 3. 场景分类 → 4. 动作识别 → 5. 系统集成

可根据实际硬件条件添加`@accelerate`装饰器进行GPU优化，或使用`@batch`处理实现批量分析。