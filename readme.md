# 视频内容搜索系统

一个基于语音识别的视频内容搜索系统，可以批量处理视频文件，提取语音内容并支持全文搜索。

## 功能特点

- 🎥 支持批量视频处理
  - 支持 MP4、AVI、MKV、MOV、FLV 等常见视频格式
  - 自动提取视频音频
  - 使用语音识别生成字幕文件
  - 支持递归处理子目录

- 🔍 强大的搜索功能
  - 全文内容搜索
  - 按视频分组显示结果
  - 精确定位到视频时间点
  - 支持视频内容预览

- 🎮 视频播放控制
  - 内置视频播放器
  - 支持跳转到指定时间点
  - 显示片段时长
  - 支持视频预加载

- 📊 系统状态监控
  - GPU 状态实时监控
  - 处理进度显示
  - 视频处理统计
  - 错误日志记录

## 系统要求

- Python 3.8+
- CUDA 支持（推荐，但不强制）
- FFmpeg（用于视频处理）
- 足够的磁盘空间用于存储处理后的文件

## 安装说明

1. 克隆仓库：
```bash
git clone [repository-url]
cd video-search
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 安装 FFmpeg：
- Windows: 下载并添加到系统环境变量
- Linux: `sudo apt install ffmpeg`
- Mac: `brew install ffmpeg`

## 使用说明

1. 启动应用：
```bash
python app.py
```

2. 打开浏览器访问：
```
http://localhost:5000
```

3. 处理视频：
   - 在"视频处理"部分输入视频目录路径
   - 点击"开始处理"
   - 等待处理完成

4. 搜索内容：
   - 在搜索框输入关键词
   - 点击搜索按钮
   - 在结果中点击时间戳可直接跳转到相应位置

## 目录结构

```
video-search/
├── app.py              # 主应用程序
├── database.py         # 数据库操作
├── video_processor.py  # 视频处理模块
├── requirements.txt    # 项目依赖
├── static/            # 静态文件
│   ├── style.css     # 样式表
│   └── script.js     # 前端脚本
├── templates/         # HTML模板
│   └── index.html    # 主页面
└── processed_videos/  # 处理后的文件
    ├── audio/        # 提取的音频
    ├── subtitles/    # 生成的字幕
    └── metadata/     # 元数据
```

## 配置说明

主要配置项在 `config.py` 中：
- 视频处理参数
- 数据库设置
- 缓存目录
- GPU 设置

## 常见问题

1. GPU 不可用？
   - 检查 CUDA 安装
   - 确认 PyTorch 版本与 CUDA 版本匹配
   - 检查显卡驱动是否最新

2. 视频处理失败？
   - 确认视频文件完整且格式支持
   - 检查 FFmpeg 是否正确安装
   - 查看日志获取详细错误信息

3. 搜索结果不准确？
   - 检查语音识别质量
   - 尝试使用更精确的关键词
   - 确认视频音频质量

## 开发计划

- [ ] 支持更多视频格式
- [ ] 添加批量导出功能
- [ ] 优化搜索算法
- [ ] 添加用户管理系统
- [ ] 支持更多语言

## 贡献指南

欢迎提交 Pull Request 或创建 Issue。

## 许可证

MIT License