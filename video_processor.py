import os
import subprocess
from tqdm import tqdm
from pathlib import Path
import whisper
import torch
import warnings
import time
from datetime import datetime
import hashlib
import json

# 过滤掉特定的警告
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

class VideoProcessor:
    def __init__(self, model_size="base", use_gpu=True):
        """初始化视频处理器
        
        Args:
            model_size (str): Whisper模型大小 ("tiny", "base", "small", "medium", "large")
            use_gpu (bool): 是否使用GPU加速
        """
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 初始化视频处理器")
        print(f"模型大小: {model_size}")
        
        # 创建必要的目录结构
        self.output_base_dir = os.path.abspath("processed_videos")
        self.audio_dir = os.path.join(self.output_base_dir, "audio")
        self.srt_dir = os.path.join(self.output_base_dir, "subtitles")
        self.metadata_dir = os.path.join(self.output_base_dir, "metadata")
        
        for directory in [self.audio_dir, self.srt_dir, self.metadata_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 检查是否有可用的GPU
        print("\nGPU检查:")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        print(f"当前CUDA版本: {torch.version.cuda}")
        
        # 尝试初始化CUDA
        if use_gpu and torch.cuda.is_available():
            try:
                # 强制初始化CUDA
                torch.cuda.init()
                # 设置当前设备
                torch.cuda.set_device(0)
                # 预热GPU
                dummy_tensor = torch.zeros(1).cuda()
                del dummy_tensor
                torch.cuda.empty_cache()
                
                print(f"当前CUDA设备: {torch.cuda.current_device()}")
                print(f"GPU型号: {torch.cuda.get_device_name(0)}")
                self.device = "cuda"
                
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"使用GPU: {gpu_name}")
                print(f"GPU内存: {gpu_memory:.2f}GB")
                print("GPU初始化成功")
            except Exception as e:
                print(f"GPU初始化失败: {str(e)}")
                self.device = "cpu"
        else:
            self.device = "cpu"
            print("使用CPU处理")
        
        print("\n正在加载Whisper模型...")
        start_time = time.time()
        try:
            # 加载模型到指定设备
            self.model = whisper.load_model(model_size)
            if self.device == "cuda":
                self.model = self.model.cuda()
            print(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")
            print(f"模型设备: {next(self.model.parameters()).device}")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise
        
    def calculate_file_md5(self, file_path: str, chunk_size=8192) -> str:
        """计算文件的MD5值
        
        Args:
            file_path (str): 文件路径
            chunk_size (int): 读取块大小
            
        Returns:
            str: MD5哈希值
        """
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
        
    def get_video_metadata(self, video_path: str) -> dict:
        """获取视频的元数据
        
        Args:
            video_path (str): 视频文件路径
            
        Returns:
            dict: 视频元数据
        """
        video_md5 = self.calculate_file_md5(video_path)
        original_filename = os.path.basename(video_path)
        file_size = os.path.getsize(video_path)
        
        metadata = {
            "md5": video_md5,
            "original_path": video_path,
            "original_filename": original_filename,
            "file_size": file_size,
            "process_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存元数据
        metadata_path = os.path.join(self.metadata_dir, f"{video_md5}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        return metadata
        
    def extract_audio(self, video_path: str, video_md5: str) -> str:
        """从视频中提取音频
        
        Args:
            video_path (str): 视频文件路径
            video_md5 (str): 视频文件的MD5值
            
        Returns:
            str: 输出的音频文件路径
        """
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始提取音频")
        print(f"处理视频: {video_path}")
        print(f"视频MD5: {video_md5}")
        
        audio_path = os.path.join(self.audio_dir, f"{video_md5}.wav")
        
        # 如果音频文件已存在，直接返回
        if os.path.exists(audio_path):
            print(f"音频文件已存在，跳过提取: {audio_path}")
            return audio_path
        
        try:
            # 获取视频信息
            info_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-hide_banner'
            ]
            try:
                result = subprocess.run(info_cmd, capture_output=True, text=True)
                print("\n视频信息:")
                print(result.stderr)
            except:
                print("无法获取视频信息")
            
            start_time = time.time()
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                '-y',
                audio_path
            ]
            
            print("\n执行FFmpeg命令:")
            print(" ".join(cmd))
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            duration = time.time() - start_time
            print(f"音频提取完成，耗时: {duration:.2f}秒")
            print(f"输出文件: {audio_path}")
            
            file_size = os.path.getsize(audio_path) / (1024*1024)
            print(f"音频文件大小: {file_size:.2f}MB")
            
            return audio_path
        except subprocess.CalledProcessError as e:
            print(f"\n错误：提取音频失败")
            print(f"错误信息: {str(e)}")
            if e.stderr:
                print(f"FFmpeg错误输出:\n{e.stderr}")
            return None
        except Exception as e:
            print(f"\n错误：处理视频时发生意外错误")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            return None
            
    def transcribe_audio(self, audio_path: str, video_md5: str) -> dict:
        """将音频转换为文字
        
        Args:
            audio_path (str): 音频文件路径
            video_md5 (str): 视频文件的MD5值
            
        Returns:
            dict: 包含转写结果和时间戳的字典
        """
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始音频转写")
        print(f"处理音频: {audio_path}")
        
        srt_path = os.path.join(self.srt_dir, f"{video_md5}.srt")
        
        # 如果字幕文件已存在，直接返回
        if os.path.exists(srt_path):
            print(f"字幕文件已存在，跳过转写: {srt_path}")
            return {
                "srt_path": srt_path,
                "video_md5": video_md5
            }
        
        try:
            start_time = time.time()
            print("正在使用Whisper进行转写...")
            result = self.model.transcribe(
                audio_path,
                fp16=self.device=="cuda",
                language="zh",
                task="transcribe",
                initial_prompt="以下是简体中文的转写："
            )
            duration = time.time() - start_time
            print(f"转写完成，耗时: {duration:.2f}秒")
            
            print(f"\n正在保存字幕文件: {srt_path}")
            segments_count = len(result["segments"])
            print(f"识别到的片段数: {segments_count}")
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(result["segments"], 1):
                    start = self._format_timestamp(segment["start"])
                    end = self._format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    
                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{text}\n\n")
            
            print("\n转写内容示例（前3段）:")
            for i, segment in enumerate(result["segments"][:3], 1):
                print(f"{i}. [{self._format_timestamp(segment['start'])} --> {self._format_timestamp(segment['end'])}]")
                print(f"   {segment['text'].strip()}")
            
            return {
                "text": result["text"],
                "segments": result["segments"],
                "srt_path": srt_path,
                "video_md5": video_md5
            }
        except Exception as e:
            print(f"\n错误：音频转写失败")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            return None
    
    def process_video(self, video_path: str) -> dict:
        """处理单个视频文件
        
        Args:
            video_path (str): 视频文件路径
            
        Returns:
            dict: 处理结果
        """
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始处理视频")
        print(f"视频文件: {video_path}")
        
        start_time = time.time()
        
        # 获取视频元数据
        metadata = self.get_video_metadata(video_path)
        video_md5 = metadata["md5"]
        print(f"视频MD5: {video_md5}")
        
        # 提取音频
        audio_path = self.extract_audio(video_path, video_md5)
        if not audio_path:
            print("音频提取失败，跳过后续处理")
            return None
            
        # 转写音频
        result = self.transcribe_audio(audio_path, video_md5)
        if not result:
            print("音频转写失败，跳过后续处理")
            return None
            
        # 添加元数据信息
        result.update(metadata)
        
        total_duration = time.time() - start_time
        print(f"\n处理完成")
        print(f"总耗时: {total_duration:.2f}秒")
        print(f"{'='*80}\n")
        
        return result
    
    def process_directory(self, input_dir: str) -> list:
        """处理目录中的所有视频文件
        
        Args:
            input_dir (str): 输入目录
            
        Returns:
            list: 所有处理结果的列表
        """
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv'}
        results = []
        
        for root, _, files in os.walk(input_dir):
            for file in tqdm(files, desc="Processing videos"):
                if Path(file).suffix.lower() in video_extensions:
                    video_path = os.path.join(root, file)
                    result = self.process_video(video_path)
                    if result:
                        results.append(result)
        
        return results
        
    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """将秒数转换为SRT时间戳格式
        
        Args:
            seconds (float): 秒数
            
        Returns:
            str: 格式化的时间戳 (HH:MM:SS,mmm)
        """
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}" 