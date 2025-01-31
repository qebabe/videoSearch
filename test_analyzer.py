import os
from pathlib import Path
from video_analyzer import VideoAnalyzer
import json
import traceback
from dashscope import Generation
from http import HTTPStatus
import dashscope
import logging
import torch
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置通义千问API Key
DASHSCOPE_API_KEY = "sk-dd0c57b8c14f4c7c9c6fece27411ae60"
os.environ['DASHSCOPE_API_KEY'] = DASHSCOPE_API_KEY
dashscope.api_key = DASHSCOPE_API_KEY

class SceneClassifier:
    def __init__(self):
        # 加载Places365预训练模型
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 365)
        
        # 下载预训练权重（如果不存在）
        weights_path = "resnet50_places365.pth.tar"
        if not os.path.exists(weights_path):
            logger.info("下载Places365预训练权重...")
            import urllib.request
            url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
            urllib.request.urlretrieve(url, weights_path)
        
        # 加载预训练权重
        checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # 如果有GPU则使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 加载类别名称
        self.classes = self._load_labels()
        
    def _load_labels(self):
        """加载Places365类别标签"""
        labels_path = "categories_places365.txt"
        if not os.path.exists(labels_path):
            logger.info("下载Places365类别标签...")
            url = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
            urllib.request.urlretrieve(url, labels_path)
            
        with open(labels_path, 'r') as f:
            return [line.strip().split(' ')[0][3:] for line in f.readlines()]
    
    def classify_scene(self, image):
        """分类场景"""
        # 预处理图像
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 进行预测
        with torch.no_grad():
            output = self.model(img_tensor)
            
        # 获取前3个最可能的场景
        _, indices = torch.sort(output[0], descending=True)
        scenes = [(self.classes[idx], float(output[0][idx])) for idx in indices[:3]]
        
        return scenes

def generate_summary_with_tongyi(results):
    """使用通义千问生成视频内容总结"""
    try:
        # 准备提示信息
        timeline_text = []
        for entry in results['results']:
            time_point = entry['time']
            event_parts = []
            
            # 音频内容（权重最高）
            if entry['audiotext']:
                event_parts.append(f"音频内容：{entry['audiotext']}")
                
            # 视觉内容
            if entry['visual']:
                event_parts.append(f"画面内容：{entry['visual']}")
                
            # OCR内容
            if entry['ocr']:
                event_parts.append(f"文字内容：{entry['ocr']}")
                
            if event_parts:
                timeline_text.append(f"时间点 {time_point}：\n" + "\n".join(event_parts))
        
        prompt = f"""请根据以下视频分析数据，生成一个详细的视频内容总结。请特别注意音频内容，因为这最能反映视频的主要内容。

视频基本信息：
- 时长：{results['metadata']['duration']:.2f}秒
- 分辨率：{results['metadata']['resolution']}

视频内容时间线：
{chr(10).join(timeline_text)}

请生成一个视频内容总结，要求：
1. 主要依据音频内容（权重最高）来理解视频主题和内容
2. 结合视觉识别和文字识别的内容，补充视频的场景信息
3. 按照时间顺序描述视频的主要内容发展
4. 总结视频的核心主题和要表达的内容

请用流畅的语言描述，突出重点内容。"""

        logger.info("正在调用通义千问API...")
        logger.info(f"使用的API Key: {DASHSCOPE_API_KEY[:8]}...")
        
        response = Generation.call(
            model='qwen-max',
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的视频内容分析师，擅长总结视频内容。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            result_format='message',
            max_tokens=1500,
            temperature=0.7,
            top_p=0.8
        )
        
        logger.info(f"API响应状态码: {response.status_code}")
        
        if response.status_code == HTTPStatus.OK:
            content = response.output.choices[0]['message']['content']
            logger.info("成功生成内容总结")
            return content
        else:
            error_msg = f"生成总结失败：{response.code} - {response.message}"
            logger.error(error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"生成总结时发生错误：{str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return error_msg

def main():
    # 设置输出目录
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 初始化视频分析器和场景分类器
    analyzer = VideoAnalyzer(use_gpu=True)
    scene_classifier = SceneClassifier()
    logger.info("场景分类器初始化完成")
    
    # 分析视频
    video_path = r"E:\BaiduNetdiskDownload\人民小酒环境宣传片.mp4"
    results = analyzer.analyze_video(video_path)
    
    try:
        # 使用视频MD5值作为文件名的一部分
        video_md5 = results['metadata']['md5']
        output_file = output_dir / f"{video_md5}_analysis_results.json"
        
        # 检查是否已存在分析结果文件
        existing_summary = None
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                    if 'summary' in existing_results:
                        existing_summary = existing_results['summary']
                        logger.info("找到已有的视频内容总结")
            except Exception as e:
                logger.warning(f"读取已有分析结果失败: {str(e)}")
        
        # 如果没有已有的总结，则调用通义千问生成
        if existing_summary is None:
            print("\n=== 生成视频内容总结 ===")
            summary = generate_summary_with_tongyi(results)
            results['summary'] = summary
        else:
            print("\n=== 使用已有的视频内容总结 ===")
            results['summary'] = existing_summary
            
        # 从总结中提取核心主题和表达内容
        summary_text = results['summary']
        desc = ""
        
        # 查找核心主题部分
        if "核心主题" in summary_text:
            start_idx = summary_text.find("核心主题")
            end_idx = summary_text.find("\n\n", start_idx)
            if end_idx == -1:  # 如果是最后一段
                end_idx = len(summary_text)
            desc = summary_text[start_idx:end_idx].strip()
        
        # 添加描述到元数据
        results['metadata']['desc'] = desc
        
        # 保存分析结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        # 打印分析报告
        print("\n=== 视频分析统计 ===")
        metadata = results['metadata']
        print(f"视频时长: {metadata['duration']:.2f}秒")
        print(f"视频分辨率: {metadata['resolution']}")
        print(f"视频MD5: {metadata['md5']}")
        print(f"视频路径: {metadata['file_path']}")
        print(f"视频描述: {metadata['desc']}")
        print(f"分析结果已保存至: {output_file}")
        
        # 打印总结
        print("\n=== 视频内容总结 ===")
        print(results['summary'])
        
        # 打印时间线分析
        print("\n=== 时间线分析 ===")
        for entry in results['results']:
            print(f"\n时间点: {entry['time']}")
            if entry['visual']:
                print(f"检测到的物体: {entry['visual']}")
            if entry['ocr']:
                print(f"识别到的文字: {entry['ocr']}")
            if entry['audiotext']:
                print(f"音频内容: {entry['audiotext']}")
                
    except Exception as e:
        print(f"错误：视频分析失败 - {str(e)}")
        traceback.print_exc()
        
        # 即使失败，也保存已有的分析结果
        video_md5 = results['metadata']['md5']
        output_file = output_dir / f"{video_md5}_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 