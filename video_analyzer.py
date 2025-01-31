import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import whisper
import openai
import json
import logging
from datetime import datetime
from tqdm import tqdm
import os
import sys
import subprocess
import hashlib
import dashscope
from http import HTTPStatus
from queue import Queue, Full
from threading import Thread
import easyocr
import traceback

class VideoAnalyzer:
    def __init__(self, use_gpu=True, openai_api_key=None, dashscope_api_key=None):
        """初始化视频分析器
        
        Args:
            use_gpu (bool): 是否使用GPU加速
            openai_api_key (str): OpenAI API密钥
            dashscope_api_key (str): 通义千问 API密钥
        """
        self.logger = self._setup_logger()
        self.openai_api_key = openai_api_key
        self.dashscope_api_key = dashscope_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
        if dashscope_api_key:
            dashscope.api_key = dashscope_api_key
        
        # 创建缓存目录
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # 创建临时目录
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # 设置环境变量
        if use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
            # 诊断信息
            self.logger.info("\nPyTorch环境诊断:")
            self.logger.info(f"Python版本: {sys.version}")
            self.logger.info(f"PyTorch版本: {torch.__version__}")
            
            # 检查CUDA是否可用
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.logger.info(f"GPU型号: {gpu_name}")
                self.logger.info(f"GPU总内存: {gpu_memory:.2f}GB")
            else:
                self.logger.warning("未检测到可用的GPU，将使用CPU处理")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            
        # 初始化模型
        self._init_models()
        
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger("VideoAnalyzer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
        
    def _init_models(self):
        """初始化所有需要的模型"""
        self.logger.info("\n开始加载模型...")
        
        # 加载YOLOv8模型用于物体检测
        try:
            self.object_detector = YOLO('yolov8n.pt')
            if isinstance(self.device, torch.device) and self.device.type == "cuda":
                self.object_detector.to(self.device)
            self.logger.info(f"YOLOv8模型加载成功，设备: {self.device}")
        except Exception as e:
            self.logger.error(f"YOLOv8模型加载失败: {str(e)}")
            self.object_detector = None
            
        # 加载EasyOCR模型用于文字识别
        try:
            use_gpu = isinstance(self.device, torch.device) and self.device.type == "cuda"
            self.ocr = easyocr.Reader(['ch_sim', 'en'], gpu=use_gpu)
            self.logger.info(f"EasyOCR模型加载成功，设备: {'GPU' if use_gpu else 'CPU'}")
        except Exception as e:
            self.logger.error(f"EasyOCR模型加载失败: {str(e)}")
            self.ocr = None
            
        # 加载Whisper模型用于音频识别
        try:
            # 设置模型路径
            model_name = "medium"
            cache_dir = os.path.expanduser("~/.cache/whisper")
            if os.name == 'nt':  # Windows系统
                cache_dir = os.path.expanduser("~\\.cache\\whisper")
            
            # 确保缓存目录存在
            os.makedirs(cache_dir, exist_ok=True)
            
            # 检查模型文件是否存在
            model_path = os.path.join(cache_dir, f"{model_name}.pt")
            if not os.path.exists(model_path):
                self.logger.warning(f"Whisper模型文件不存在: {model_path}")
                self.logger.info(f"请手动下载模型文件并放置在: {model_path}")
                self.logger.info("下载链接:")
                self.logger.info("- tiny: https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt")
                self.logger.info("- base: https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt")
                self.logger.info("- small: https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt")
                self.logger.info("- medium: https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt")
                self.logger.info("- large: https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt")
            
            # 加载模型
            self.audio_model = whisper.load_model(model_name)
            if isinstance(self.device, torch.device) and self.device.type == "cuda":
                self.audio_model.to(self.device)
            self.logger.info(f"Whisper模型({model_name})加载成功")
        except Exception as e:
            self.logger.error(f"Whisper模型加载失败: {str(e)}")
            self.audio_model = None
            
    def analyze_video(self, video_path):
        """分析视频内容"""
        try:
            # 获取视频基本信息
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 计算视频文件的MD5
            md5_hash = self._calculate_file_md5(video_path)
            
            # 准备结果数据结构
            results = {
                'metadata': {
                    'duration': duration,
                    'resolution': f"{width}x{height}",
                    'md5': md5_hash,
                    'file_path': str(video_path)
                },
                'results': []
            }
            
            # 检查是否存在缓存的分析结果
            cache_dir = Path("cache")
            cache_dir.mkdir(exist_ok=True)
            
            visual_cache_file = cache_dir / f"{md5_hash}_visual_results.json"
            audio_cache_file = cache_dir / f"{md5_hash}_audio_results.json"
            
            # 分析视觉内容
            visual_results = []
            if visual_cache_file.exists():
                self.logger.info("发现视觉分析缓存，正在加载...")
                with open(visual_cache_file, 'r', encoding='utf-8') as f:
                    visual_results = json.load(f)
            else:
                visual_results = self._analyze_visual_content(video_path, frame_count)
                with open(visual_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(visual_results, f, ensure_ascii=False, indent=2)
            
            # 分析音频内容
            audio_results = []
            if audio_cache_file.exists():
                self.logger.info("使用缓存的音频分析结果")
                with open(audio_cache_file, 'r', encoding='utf-8') as f:
                    audio_results = json.load(f)
            else:
                audio_results = self._analyze_audio_content(video_path)
                with open(audio_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(audio_results, f, ensure_ascii=False, indent=2)
            
            # 合并结果
            results['results'] = self._merge_timeline_results(visual_results, audio_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"视频分析失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        
    def _extract_metadata(self, video_path: str) -> dict:
        """提取视频元数据"""
        cap = cv2.VideoCapture(video_path)
        metadata = {
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            "resolution": f"{int(cap.get(3))}x{int(cap.get(4))}"
        }
        cap.release()
        return metadata
        
    def _get_cache_path(self, video_path: str, analysis_type: str) -> Path:
        """获取缓存文件路径
        
        Args:
            video_path (str): 视频文件路径
            analysis_type (str): 分析类型 ('visual' 或 'audio')
            
        Returns:
            Path: 缓存文件路径
        """
        video_md5 = self._calculate_file_md5(video_path)
        return self.cache_dir / f"{video_md5}_{analysis_type}_results.json"

    def _load_cache(self, cache_path: Path) -> dict:
        """加载缓存文件"""
        try:
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"加载缓存失败: {str(e)}")
        return None

    def _save_cache(self, cache_path: Path, data: dict):
        """保存缓存文件"""
        try:
            # 转换数据为JSON可序列化格式
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: convert_to_serializable(value) for key, value in obj.items()}
                return obj
            
            # 转换数据
            serializable_data = convert_to_serializable(data)
            
            # 保存到文件
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存缓存失败: {str(e)}")
            
    def _analyze_visual(self, video_path: str) -> dict:
        """分析视频的视觉内容"""
        # 初始化空结果
        empty_results = {
            "objects": [],
            "texts": [],
            "metadata": {
                "total_frames": 0,
                "fps": 0,
                "width": 0,
                "height": 0,
                "sample_interval": 0
            }
        }
        
        # 检查缓存
        cache_path = self._get_cache_path(video_path, "visual")
        if cache_path.exists():
            self.logger.info("发现视觉分析缓存，正在加载...")
            cached_results = self._load_cache(cache_path)
            if cached_results:
                return cached_results
            else:
                self.logger.warning("缓存加载失败，将重新分析")
                
        self.logger.info("开始视觉内容分析...")
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error("无法打开视频文件")
            return empty_results
            
        # 获取视频基本信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 设置帧采样间隔（1帧/秒）
        sample_interval = max(1, int(fps))
        
        # 创建显示窗口
        cv2.namedWindow('Video Analysis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video Analysis', 1280, 720)
        
        # 初始化结果
        results = {
            "objects": [],
            "texts": [],
            "metadata": {
                "total_frames": int(total_frames),
                "fps": float(fps),
                "width": int(frame_width),
                "height": int(frame_height),
                "sample_interval": int(sample_interval)
            }
        }
        
        try:
            frame_count = 0
            with tqdm(total=total_frames, desc="分析帧", position=1) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    pbar.update(1)
                    
                    # 每秒采样一帧进行分析
                    if frame_count % sample_interval != 0:
                        continue
                        
                    # 物体检测
                    if self.object_detector:
                        try:
                            detections = self.object_detector(frame)[0]
                            for det in detections.boxes.data.tolist():
                                x1, y1, x2, y2, conf, cls = det
                                class_name = detections.names[int(cls)]
                                results["objects"].append({
                                    "frame": int(frame_count),
                                    "time": float(frame_count / fps),
                                    "class": str(class_name),
                                    "confidence": float(conf),
                                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                                })
                                
                                # 在帧上绘制检测框
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.putText(frame, f"{class_name}: {conf:.2f}", 
                                          (int(x1), int(y1) - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        except Exception as e:
                            self.logger.error(f"物体检测失败: {str(e)}")
                    
                    # 文字识别
                    if self.ocr:
                        try:
                            ocr_results = self.ocr.readtext(frame)
                            for detection in ocr_results:
                                bbox, text, conf = detection
                                if conf > 0.5:  # 仅保留置信度大于0.5的结果
                                    # 确保边界框坐标是原生Python类型
                                    bbox = [[float(x), float(y)] for x, y in bbox]
                                    results["texts"].append({
                                        "frame": int(frame_count),
                                        "time": float(frame_count / fps),
                                        "text": str(text),
                                        "confidence": float(conf),
                                        "bbox": bbox
                                    })
                                    
                                    # 在帧上绘制文字框
                                    pts = np.array(bbox, np.int32)
                                    cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                                    cv2.putText(frame, text, 
                                              (int(bbox[0][0]), int(bbox[0][1]) - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        except Exception as e:
                            self.logger.error(f"文字识别失败: {str(e)}")
                    
                    # 显示处理后的帧
                    cv2.imshow('Video Analysis', frame)
                    
                    # 按'q'键退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        except Exception as e:
            self.logger.error(f"视频分析过程中发生错误: {str(e)}")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # 保存缓存（即使发生错误也保存已分析的结果）
            if results["objects"] or results["texts"]:
                self._save_cache(cache_path, results)
            
            return results
            
    def _analyze_audio(self, video_path: str) -> dict:
        """使用Whisper分析视频的音频内容"""
        # 检查缓存
        cache_path = self._get_cache_path(video_path, 'audio')
        cached_results = self._load_cache(cache_path)
        if cached_results:
            self.logger.info("使用缓存的音频分析结果")
            return cached_results

        self.logger.info("\n开始音频分析")
        
        try:
            # 计算视频文件的MD5值
            video_md5 = self._calculate_file_md5(video_path)
            audio_path = self.temp_dir / f"{video_md5}.wav"
            
            # 提取音频
            if not audio_path.exists():
                self.logger.info("正在提取音频...")
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-vn',
                    '-acodec', 'pcm_s16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-y',
                    str(audio_path)
                ]
                
                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8',
                        errors='ignore'
                    )
                    _, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        self.logger.error(f"FFmpeg错误: {stderr}")
                        raise subprocess.CalledProcessError(process.returncode, cmd, stderr)
                        
                    self.logger.info(f"音频提取成功: {audio_path}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"FFmpeg错误: {e.stderr}")
                    raise
            else:
                self.logger.info(f"使用已存在的音频文件: {audio_path}")
            
            # 使用Whisper进行音频识别
            if self.audio_model:
                result = self.audio_model.transcribe(
                    str(audio_path),
                    language='zh',  # 使用'zh'来指定简体中文
                    task='transcribe',
                    initial_prompt="这是一段中文广告宣传片的解说词。请忽略背景音乐，只输出清晰的人声内容。",
                    temperature=0.0,  # 完全消除随机性
                    best_of=5,  # 生成多个候选结果并选择最佳
                    beam_size=5,  # 使用波束搜索
                    condition_on_previous_text=True,  # 考虑前文语境
                    no_speech_threshold=0.6,  # 提高非语音检测阈值
                    logprob_threshold=-1.0,  # 提高置信度要求
                    compression_ratio_threshold=2.4,  # 控制输出文本的压缩比
                    word_timestamps=True  # 启用词级时间戳以更好地过滤噪音
                )
                
                # 清理和过滤文本
                def clean_text(text):
                    # 移除音乐符号和特殊字符
                    text = text.replace('♪', '').replace('♫', '').replace('', '')
                    text = ''.join(char for i, char in enumerate(text) if char != text[i-1:i])
                    
                    # 移除重复的字母
                    import re
                    text = re.sub(r'([A-Za-z])\1{2,}', r'\1', text)
                    
                    # 修正常见的错误识别
                    corrections = {
                        '沿博春': '严伯村',
                        '地球额博': '严伯村',
                        '小锅': '小国',
                        '声动': '生动',
                        '举示': '举世',
                        '分阅': '分岸'
                    }
                    for wrong, right in corrections.items():
                        text = text.replace(wrong, right)
                    
                    # 规范化标点符号
                    text = text.replace('。。', '。').replace(',,', ',').replace(',.', '。')
                    text = text.replace('?,', '。').replace('? ', '。').replace('.,', '。')
                    
                    # 确保句子以合适的标点结束
                    if text and not text[-1] in '。！？':
                        text += '。'
                    
                    return text.strip()
                
                # 过滤和清理文本段落
                filtered_segments = []
                prev_text = ""
                for segment in result["segments"]:
                    text = clean_text(segment["text"])
                    # 如果当前文本不是上一个的重复，且长度合适，且不全是标点符号
                    if (text != prev_text and 
                        len(text) > 1 and 
                        len(text) < 100 and  # 避免过长的段落
                        any(c.isalnum() for c in text)):  # 确保包含实际文字
                        
                        # 检查是否应该与前一个段落合并
                        if (filtered_segments and 
                            filtered_segments[-1]["end"] + 0.3 >= segment["start"] and
                            len(filtered_segments[-1]["text"] + text) < 100):
                            # 合并相邻的短句
                            filtered_segments[-1]["text"] += text
                            filtered_segments[-1]["end"] = segment["end"]
                        else:
                            filtered_segments.append({
                                "text": text,
                                "start": segment["start"],
                                "end": segment["end"]
                            })
                        prev_text = text
                
                # 合并最终文本
                transcript = " ".join(segment["text"] for segment in filtered_segments)
                segments = filtered_segments
                
                # 清理临时文件
                if audio_path.exists():
                    audio_path.unlink()
                
                results = {
                    "transcript": transcript,
                    "segments": segments
                }
                
                # 保存结果到缓存
                self._save_cache(cache_path, results)
                return results
            else:
                self.logger.error("Whisper模型未正确加载")
                return {
                    "transcript": "",
                    "segments": []
                }
            
        except Exception as e:
            self.logger.error(f"音频分析失败: {str(e)}")
            return {
                "transcript": "",
                "segments": []
            }
    
    def _calculate_file_md5(self, file_path: str, chunk_size=8192) -> str:
        """计算文件的MD5值"""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
            
    def _merge_timeline_results(self, visual_results: dict, audio_results: dict) -> list:
        """合并视觉和音频结果到时间线格式"""
        # 创建时间线字典，用于合并同一时间点的事件
        timeline_dict = {}
        
        # 处理视觉检测结果
        for obj in visual_results.get("objects", []):
            time_str = f"{int(obj['time'] // 60):02d}:{int(obj['time'] % 60):02d}"
            if time_str not in timeline_dict:
                timeline_dict[time_str] = {"time": time_str, "visual": [], "ocr": [], "audiotext": ""}
            timeline_dict[time_str]["visual"].append(obj["class"])
            
        # 处理OCR结果
        for text in visual_results.get("texts", []):
            time_str = f"{int(text['time'] // 60):02d}:{int(text['time'] % 60):02d}"
            if time_str not in timeline_dict:
                timeline_dict[time_str] = {"time": time_str, "visual": [], "ocr": [], "audiotext": ""}
            timeline_dict[time_str]["ocr"].append(text["text"])
            
        # 处理音频结果
        if audio_results and "segments" in audio_results:
            for segment in audio_results["segments"]:
                time_str = f"{int(segment['start'] // 60):02d}:{int(segment['start'] % 60):02d}"
                if time_str not in timeline_dict:
                    timeline_dict[time_str] = {"time": time_str, "visual": [], "ocr": [], "audiotext": ""}
                timeline_dict[time_str]["audiotext"] = segment["text"]
        
        # 转换字典为列表并去重
        timeline_list = []
        for time_point in sorted(timeline_dict.keys()):
            entry = timeline_dict[time_point]
            # 去重并转换列表为字符串
            entry["visual"] = ", ".join(sorted(set(entry["visual"]))) if entry["visual"] else ""
            entry["ocr"] = ", ".join(sorted(set(entry["ocr"]))) if entry["ocr"] else ""
            timeline_list.append(entry)
            
        return timeline_list

    def _generate_summary(self, visual_results: dict, audio_results: dict) -> str:
        """使用通义千问或OpenAI API生成内容摘要"""
        try:
            if self.dashscope_api_key:
                return self._generate_tongyi_summary(visual_results, audio_results)
            elif self.openai_api_key:
                return self._generate_openai_summary(visual_results, audio_results)
            else:
                return self._generate_basic_summary(visual_results, audio_results)
                
        except Exception as e:
            self.logger.error(f"生成摘要失败: {str(e)}")
            return self._generate_basic_summary(visual_results, audio_results)

    def _generate_tongyi_summary(self, visual_results: dict, audio_results: dict) -> str:
        """使用通义千问API生成内容摘要"""
        try:
            # 生成详细的时间线数据
            timeline_data = []
            
            # 添加视觉事件
            for obj in visual_results["objects"]:
                timeline_data.append({
                    "timestamp": obj["time"],
                    "type": "visual",
                    "content": f"检测到{obj['class']}（置信度：{obj['confidence']:.2f}）"
                })
            
            # 添加音频事件
            if audio_results and "segments" in audio_results:
                for segment in audio_results["segments"]:
                    timeline_data.append({
                        "timestamp": segment["start"],
                        "type": "audio",
                        "content": segment["text"]
                    })
            
            # 按时间排序
            timeline_data.sort(key=lambda x: x["timestamp"])
            
            # 生成时间线文本
            timeline_text = []
            current_second = 0
            events_in_second = []
            
            for event in timeline_data:
                second = int(event["timestamp"])
                if second != current_second and events_in_second:
                    # 合并同一秒内的事件
                    timeline_text.append(f"第{current_second}秒：{'；'.join(events_in_second)}")
                    events_in_second = []
                current_second = second
                if event["type"] == "visual":
                    events_in_second.append(f"画面中{event['content']}")
                else:
                    events_in_second.append(f"音频：{event['content']}")
            
            # 添加最后一秒的事件
            if events_in_second:
                timeline_text.append(f"第{current_second}秒：{'；'.join(events_in_second)}")

            # 准备提示信息
            prompt = f"""请根据以下视频分析数据，生成一个详细的视频内容时序描述报告。

视频基本信息：
- 视频总时长：{visual_results['metadata']['total_frames'] / visual_results['metadata']['fps']:.1f}秒
- 分析帧数：{visual_results['metadata']['total_frames']}帧
- 检测到的总物体数：{len(visual_results['objects'])}个
- 物体类别数：{len(set(obj['class'] for obj in visual_results['objects']))}种

详细时间线：
{chr(10).join(timeline_text[:50])}  # 限制前50个时间点避免超出token限制

请生成一个详细的视频内容分析报告，包含以下内容：
1. 按照时间顺序描述视频中的主要场景变化和重要事件
2. 重点说明每个场景中出现的主要人物和物体，以及他们的互动
3. 结合音频内容，解释视频画面与旁白/对话的关系
4. 总结视频的整体叙事结构和主要内容主题

请用流畅的语言描述，突出时间顺序，让读者能清晰理解视频的内容发展脉络。"""

            # 调用通义千问API
            self.logger.info("正在调用通义千问API生成分析报告...")
            
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的视频内容分析助手，擅长生成详细的视频内容分析报告。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            try:
                import requests.exceptions
                from urllib3.exceptions import ReadTimeoutError
                from socket import timeout
                
                # 设置超时时间为30秒
                response = dashscope.Generation.call(
                    model='qwen-max',
                    messages=messages,
                    result_format='message',
                    timeout=30,  # 设置超时时间
                    max_tokens=1500,  # 限制生成的文本长度
                    temperature=0.7,  # 控制生成文本的创造性
                    top_p=0.8  # 控制生成文本的多样性
                )
                
                if response.status_code == HTTPStatus.OK:
                    return response.output.choices[0]['message']['content']
                else:
                    error_msg = f"通义千问API调用失败: {response.code} - {response.message}"
                    self.logger.error(error_msg)
                    return self._generate_basic_summary(visual_results, audio_results)
                    
            except (requests.exceptions.Timeout, ReadTimeoutError, timeout, KeyboardInterrupt) as e:
                self.logger.error(f"通义千问API调用超时或被中断: {str(e)}")
                return self._generate_basic_summary(visual_results, audio_results)
                
            except Exception as e:
                self.logger.error(f"通义千问API调用异常: {str(e)}")
                return self._generate_basic_summary(visual_results, audio_results)
                
        except Exception as e:
            self.logger.error(f"生成摘要时发生错误: {str(e)}")
            return self._generate_basic_summary(visual_results, audio_results)

    def _format_object_statistics(self, objects: list) -> str:
        """格式化物体检测统计信息"""
        object_counts = {}
        for obj in objects:
            key = obj["class"]
            if key not in object_counts:
                object_counts[key] = {
                    "count": 0,
                    "timestamps": []
                }
            object_counts[key]["count"] += 1
            object_counts[key]["timestamps"].append(obj["time"])
        
        stats = []
        for obj, data in sorted(object_counts.items(), key=lambda x: x[1]["count"], reverse=True):
            timestamps = sorted(data["timestamps"])
            first_appear = min(timestamps)
            last_appear = max(timestamps)
            stats.append(f"- {obj}: 出现{data['count']}次，首次出现于{first_appear:.1f}秒，最后出现于{last_appear:.1f}秒")
        
        return "\n".join(stats)

    def _format_timeline(self, visual_objects: list, audio_segments: list) -> str:
        """格式化时间线
        
        Args:
            visual_objects (list): 视觉对象列表
            audio_segments (list): 音频片段列表
            
        Returns:
            str: 格式化的时间线
        """
        timeline = []
        
        # 添加视觉检测结果到时间线
        for obj in visual_objects:
            timeline.append({
                "timestamp": obj["time"],
                "type": "visual",
                "content": f"检测到{obj['class']}（置信度：{obj['confidence']:.2f}）"
            })
            
        # 添加文字识别结果到时间线
        for text in visual_objects.get("texts", []):
            timeline.append({
                "timestamp": text["time"],
                "type": "text",
                "content": f"识别到文字：{text['text']}（置信度：{text['confidence']:.2f}）"
            })
        
        # 添加音频片段到时间线
        for segment in audio_segments:
            timeline.append({
                "timestamp": segment["start"],
                "type": "audio",
                "content": segment["text"]
            })
            
        # 按时间戳排序
        timeline.sort(key=lambda x: x["timestamp"])
        
        # 格式化输出
        formatted_timeline = []
        for event in timeline:
            timestamp = event["timestamp"]
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            formatted_time = f"{minutes:02d}:{seconds:02d}"
            
            if event["type"] == "visual":
                prefix = "👁"
            elif event["type"] == "text":
                prefix = "📝"
            else:
                prefix = "🔊"
                
            formatted_timeline.append(f"{formatted_time} {prefix} {event['content']}")
            
        return "\n".join(formatted_timeline)

    def _generate_openai_summary(self, visual_results: dict, audio_results: dict) -> str:
        """使用OpenAI API生成内容摘要"""
        try:
            detected_objects = set(obj["class"] for obj in visual_results["objects"])
            prompt = f"""请根据以下信息生成一个简洁的视频内容摘要：

视觉内容：
- 检测到的物体：{', '.join(detected_objects)}
- 分析的帧数：{visual_results['metadata']['total_frames']}

音频内容：
{audio_results['transcript']}

请生成一个简短的摘要，描述视频的主要内容和重要事件。"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的视频内容分析助手。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI摘要生成失败: {str(e)}")
            return self._generate_basic_summary(visual_results, audio_results)

    def _generate_basic_summary(self, visual_results: dict, audio_results: dict) -> str:
        """生成基本的内容摘要"""
        try:
            if not visual_results or not isinstance(visual_results, dict):
                visual_results = {
                    "objects": [],
                    "texts": [],
                    "metadata": {
                        "total_frames": 0,
                        "fps": 0,
                        "width": 0,
                        "height": 0
                    }
                }
                
            if not audio_results or not isinstance(audio_results, dict):
                audio_results = {
                    "segments": []
                }
            
            # 统计物体检测结果
            object_stats = self._format_object_statistics(visual_results.get("objects", []))
            
            # 生成时间线
            timeline = self._format_timeline(visual_results, audio_results.get("segments", []))
            
            # 生成摘要文本
            summary = f"""视频分析报告：

基本信息：
- 视频时长：{visual_results['metadata']['total_frames'] / visual_results['metadata']['fps']:.1f}秒
- 视频分辨率：{visual_results['metadata']['width']}x{visual_results['metadata']['height']}
- 检测到的物体数：{len(visual_results.get('objects', []))}
- 识别到的文字数：{len(visual_results.get('texts', []))}
- 音频片段数：{len(audio_results.get('segments', []))}

物体检测统计：
{object_stats}

详细时间线：
{timeline}
"""
            return summary
            
        except Exception as e:
            self.logger.error(f"生成基本摘要时发生错误: {str(e)}")
            return "无法生成视频分析摘要。"

    def _time_to_seconds(self, time_str):
        """将时间字符串（MM:SS）转换为秒数"""
        try:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds
        except Exception as e:
            self.logger.error(f"时间转换失败: {str(e)}")
            return 0 