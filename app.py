import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from video_processor import VideoProcessor
from database import Database
import threading
import torch
import torch.cuda
import multiprocessing
from pathlib import Path

# 全局变量用于存储处理器实例
global_processor = None

def initialize_cuda():
    """初始化CUDA环境"""
    print("\n正在初始化CUDA环境...")
    if torch.cuda.is_available():
        try:
            # 强制初始化CUDA
            torch.cuda.init()
            # 预热GPU
            dummy_tensor = torch.zeros(1).cuda()
            del dummy_tensor
            torch.cuda.empty_cache()
            print("CUDA环境初始化成功")
            print(f"PyTorch版本: {torch.__version__}")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"当前GPU: {torch.cuda.get_device_name(0)}")
            return True
        except Exception as e:
            print(f"CUDA初始化失败: {str(e)}")
            return False
    else:
        print("未检测到CUDA设备")
        return False

# 在主线程中初始化CUDA
cuda_available = initialize_cuda()

app = Flask(__name__)

# 初始化数据库
db = Database()

# 存储处理任务的状态
processing_tasks = {}

def check_gpu():
    """检查GPU状态"""
    if torch.cuda.is_available():
        return {
            'available': True,
            'device_name': torch.cuda.get_device_name(0),
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'memory_allocated': f"{torch.cuda.memory_allocated(0)/1024**3:.2f}GB",
            'memory_cached': f"{torch.cuda.memory_reserved(0)/1024**3:.2f}GB"
        }
    return {'available': False}

def process_directory_task(directory_path, task_id):
    """后台处理目录的任务"""
    try:
        # 检查GPU状态
        gpu_info = check_gpu()
        processing_tasks[task_id] = {
            'status': 'initializing',
            'message': f"正在初始化处理器... GPU状态: {'可用' if gpu_info['available'] else '不可用'}"
        }
        
        # 初始化处理器
        global global_processor
        if global_processor is None:
            global_processor = VideoProcessor(use_gpu=cuda_available)
        
        # 更新状态为处理中
        processing_tasks[task_id] = {
            'status': 'processing',
            'message': f"正在处理目录... 使用设备: {global_processor.device}"
        }
        
        # 获取视频文件列表，使用集合去重
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv'}
        video_files = set()  # 使用集合存储唯一的文件路径
        
        # 递归查找视频文件
        for ext in video_extensions:
            # 转换为字符串路径并添加到集合中
            video_files.update(str(p) for p in Path(directory_path).glob(f'**/*{ext}'))
            video_files.update(str(p) for p in Path(directory_path).glob(f'**/*{ext.upper()}'))
        
        # 转换为列表并排序，确保处理顺序一致
        video_files = sorted(list(video_files))
        total_files = len(video_files)
        processed_files = 0
        failed_files = []
        
        print(f"\n找到 {total_files} 个视频文件:")
        for file in video_files:
            print(f"- {file}")
        print()
        
        for video_path in video_files:
            try:
                print(f"\n开始处理视频: {video_path}")
                result = global_processor.process_video(video_path)
                if result:
                    db.add_video_result(result)
                    processed_files += 1
                    print(f"处理成功: {video_path}")
                else:
                    failed_files.append(video_path)
                    print(f"处理失败: {video_path}")
                
                # 更新进度
                processing_tasks[task_id] = {
                    'status': 'processing',
                    'message': f'已处理: {processed_files}/{total_files}',
                    'progress': {
                        'current': processed_files,
                        'total': total_files,
                        'failed': len(failed_files)
                    }
                }
            except Exception as e:
                failed_files.append(video_path)
                print(f"处理文件失败: {video_path}, 错误: {str(e)}")
        
        # 完成处理
        if failed_files:
            processing_tasks[task_id] = {
                'status': 'completed_with_errors',
                'message': f'处理完成，成功: {processed_files}, 失败: {len(failed_files)}',
                'failed_files': failed_files
            }
        else:
            processing_tasks[task_id] = {
                'status': 'completed',
                'message': f'处理完成，共处理 {processed_files} 个文件'
            }
            
    except Exception as e:
        processing_tasks[task_id] = {
            'status': 'error',
            'message': f'错误: {str(e)}'
        }

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/gpu-status')
def gpu_status():
    """获取GPU状态"""
    return jsonify(check_gpu())

@app.route('/process', methods=['POST'])
def process_directory():
    """处理视频目录"""
    directory = request.json.get('directory', '')
    if not directory or not os.path.isdir(directory):
        return jsonify({'error': '无效的目录路径'}), 400
    
    # 创建处理任务
    task_id = str(hash(directory))
    processing_tasks[task_id] = {
        'status': 'initializing',
        'message': '正在初始化...'
    }
    
    # 启动后台处理线程
    thread = threading.Thread(
        target=process_directory_task,
        args=(directory, task_id)
    )
    thread.start()
    
    return jsonify({
        'message': '开始处理目录',
        'task_id': task_id
    })

@app.route('/task/<task_id>')
def get_task_status(task_id):
    """获取任务状态"""
    if task_id in processing_tasks:
        return jsonify(processing_tasks[task_id])
    return jsonify({'error': '任务不存在'}), 404

@app.route('/search')
def search():
    """搜索视频内容"""
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    
    results = db.search_videos(query)
    return jsonify(results)

@app.route('/processed-videos')
def get_processed_videos():
    """获取已处理的视频列表"""
    try:
        videos = db.get_processed_videos()
        return jsonify(videos)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear-database', methods=['POST'])
def clear_database():
    """清空数据库"""
    try:
        # 打印当前工作目录和processed_videos目录路径
        print(f"\n当前工作目录: {os.getcwd()}")
        processed_dir = db.get_processed_videos_dir()
        print(f"处理目录路径: {processed_dir}")
        print(f"处理目录是否存在: {os.path.exists(processed_dir)}")
        
        if os.path.exists(processed_dir):
            for subdir in ['audio', 'subtitles', 'temp']:
                dir_path = os.path.join(processed_dir, subdir)
                if os.path.exists(dir_path):
                    print(f"子目录存在: {dir_path}")
                    print(f"子目录中的文件: {os.listdir(dir_path)}")
        
        db.clear_database()
        return jsonify({'message': '数据库已清空'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video/<path:filename>')
def serve_video(filename):
    """提供视频流服务"""
    try:
        # 对路径进行安全检查
        if '..' in filename or filename.startswith('/'):
            return jsonify({'error': '无效的文件路径'}), 400
            
        # 获取视频文件的完整路径
        video_path = os.path.abspath(filename)
        if not os.path.exists(video_path):
            return jsonify({'error': '视频文件不存在'}), 404
            
        # 使用 send_file 提供视频流，支持范围请求
        return send_from_directory(
            os.path.dirname(video_path),
            os.path.basename(video_path),
            conditional=True  # 启用范围请求支持
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 启动时打印GPU状态
    gpu_info = check_gpu()
    print("\nGPU 状态:")
    if gpu_info['available']:
        print(f"GPU可用: {gpu_info['device_name']}")
        print(f"GPU数量: {gpu_info['device_count']}")
        print(f"当前设备: {gpu_info['current_device']}")
        print(f"已分配内存: {gpu_info['memory_allocated']}")
        print(f"缓存内存: {gpu_info['memory_cached']}")
    else:
        print("GPU不可用，将使用CPU处理")
    print("\n")
    
    # 在主线程中初始化处理器
    global_processor = VideoProcessor(use_gpu=cuda_available)
    
    # 禁用调试模式运行
    app.run(debug=False, port=5000, use_reloader=False, threaded=False) 