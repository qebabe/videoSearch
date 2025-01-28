import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QPushButton, QLineEdit, QTextEdit, QFileDialog,
                           QProgressBar, QLabel, QMessageBox, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from video_processor import VideoProcessor
from database import Database

class ProcessingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool)
    
    def __init__(self, input_dir, use_gpu=True):
        super().__init__()
        self.input_dir = input_dir
        self.use_gpu = use_gpu
        
    def run(self):
        try:
            processor = VideoProcessor(use_gpu=self.use_gpu)
            db = Database()
            
            results = processor.process_directory(self.input_dir)
            
            for result in results:
                db.add_video_result(result)
                self.progress.emit(f"处理完成: {result['original_filename']} (MD5: {result['md5']})")
                
            self.finished.emit(True)
        except Exception as e:
            self.progress.emit(f"处理出错: {str(e)}")
            self.finished.emit(False)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频内容搜索")
        self.setMinimumSize(800, 600)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # 添加控件
        self.setup_ui(layout)
        
        # 初始化数据库
        self.db = Database()
        
    def setup_ui(self, layout):
        # 视频处理部分
        process_label = QLabel("视频处理")
        process_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(process_label)
        
        # 选择输入目录
        self.input_dir_btn = QPushButton("选择视频目录")
        self.input_dir_btn.clicked.connect(self.select_input_dir)
        layout.addWidget(self.input_dir_btn)
        
        self.input_dir_label = QLabel("未选择目录")
        layout.addWidget(self.input_dir_label)
        
        # GPU选项
        self.use_gpu_checkbox = QCheckBox("使用GPU加速（如果可用）")
        self.use_gpu_checkbox.setChecked(True)
        layout.addWidget(self.use_gpu_checkbox)
        
        # 处理按钮
        self.process_btn = QPushButton("开始处理")
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn)
        
        # 进度显示
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(100)
        layout.addWidget(self.progress_text)
        
        # 搜索部分
        search_label = QLabel("视频搜索")
        search_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(search_label)
        
        # 搜索框
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入搜索关键词")
        self.search_input.returnPressed.connect(self.search_videos)
        layout.addWidget(self.search_input)
        
        # 搜索按钮
        self.search_btn = QPushButton("搜索")
        self.search_btn.clicked.connect(self.search_videos)
        layout.addWidget(self.search_btn)
        
        # 搜索结果
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
        
    def select_input_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择视频目录")
        if dir_path:
            self.input_dir_label.setText(dir_path)
            
    def start_processing(self):
        input_dir = self.input_dir_label.text()
        
        if input_dir == "未选择目录":
            QMessageBox.warning(self, "警告", "请先选择输入目录")
            return
            
        self.process_btn.setEnabled(False)
        self.progress_text.clear()
        
        # 创建处理线程
        self.processing_thread = ProcessingThread(
            input_dir,
            use_gpu=self.use_gpu_checkbox.isChecked()
        )
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.start()
        
    def update_progress(self, message):
        self.progress_text.append(message)
        
    def processing_finished(self, success):
        self.process_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "完成", "视频处理完成")
        else:
            QMessageBox.warning(self, "错误", "处理过程中出现错误")
            
    def search_videos(self):
        query = self.search_input.text().strip()
        if not query:
            return
            
        results = self.db.search_videos(query)
        self.results_text.clear()
        
        if not results:
            self.results_text.append("未找到匹配的内容")
            return
            
        for result in results:
            self.results_text.append(f"视频: {result['original_filename']}")
            self.results_text.append(f"MD5: {result['video_md5']}")
            self.results_text.append(f"时间: {self._format_time(result['start_time'])} - {self._format_time(result['end_time'])}")
            self.results_text.append(f"内容: {result['text']}")
            self.results_text.append("-" * 50)
            
    @staticmethod
    def _format_time(seconds):
        """格式化时间为 HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 