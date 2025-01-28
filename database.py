from sqlalchemy import create_engine, Column, Integer, String, Float, Text, func, case, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
import os

Base = declarative_base()

class VideoSegment(Base):
    __tablename__ = 'video_segments'
    
    id = Column(Integer, primary_key=True)
    video_md5 = Column(String(32))  # MD5哈希值
    original_filename = Column(String(500))  # 原始文件名
    original_path = Column(String(500))  # 原始文件路径
    start_time = Column(Float)
    end_time = Column(Float)
    text = Column(Text)
    srt_path = Column(String(500))
    file_size = Column(Integer)  # 文件大小（字节）
    created_at = Column(String(50))  # 创建时间
    process_time = Column(String(50))  # 处理时间

class Database:
    def __init__(self):
        """初始化数据库连接"""
        db_path = os.path.join(os.path.dirname(__file__), 'video_search.db')
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_video_result(self, result):
        """添加视频处理结果到数据库"""
        session = self.Session()
        try:
            if 'segments' not in result:
                raise ValueError("结果中缺少segments字段")
                
            for segment in result['segments']:
                video_segment = VideoSegment(
                    video_md5=result['video_md5'],
                    original_filename=result['original_filename'],
                    original_path=result['original_path'],
                    start_time=segment['start'],
                    end_time=segment['end'],
                    text=segment['text'],
                    srt_path=result['srt_path'],
                    file_size=result['file_size'],
                    created_at=result.get('created_at', ''),
                    process_time=result.get('process_time', '')
                )
                session.add(video_segment)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error adding video result to database: {str(e)}")
            raise
        finally:
            session.close()
            
    def search_videos(self, query: str) -> list:
        """搜索视频内容"""
        session = self.Session()
        try:
            # 如果查询以 video: 开头，则按视频MD5搜索
            if query.startswith('video:'):
                video_md5 = query[6:].strip()
                results = session.query(VideoSegment).filter(
                    VideoSegment.video_md5 == video_md5
                ).order_by(VideoSegment.start_time).all()
            else:
                # 否则按内容搜索
                results = session.query(VideoSegment).filter(
                    VideoSegment.text.like(f"%{query}%")
                ).order_by(
                    VideoSegment.video_md5,
                    VideoSegment.start_time
                ).all()
            
            return [{
                'video_md5': result.video_md5,
                'original_filename': result.original_filename,
                'original_path': result.original_path,
                'start_time': result.start_time,
                'end_time': result.end_time,
                'text': result.text,
                'srt_path': result.srt_path,
                'file_size': result.file_size,
                'process_time': result.process_time
            } for result in results]
        finally:
            session.close()
            
    def get_all_videos(self) -> list:
        """获取所有已处理的视频信息
        
        Returns:
            list: 视频信息列表
        """
        session = self.Session()
        try:
            results = session.query(
                VideoSegment.video_md5,
                VideoSegment.original_filename,
                VideoSegment.original_path
            ).distinct().all()
            return [{
                'md5': result[0],
                'filename': result[1],
                'path': result[2]
            } for result in results]
        finally:
            session.close()

    def get_processed_videos(self):
        """获取已处理的视频列表"""
        session = self.Session()
        try:
            # 先获取所有唯一的视频MD5和对应的最新记录
            subquery = (
                session.query(
                    VideoSegment.video_md5,
                    VideoSegment.original_filename,
                    VideoSegment.created_at,
                    func.row_number().over(
                        partition_by=VideoSegment.video_md5,
                        order_by=VideoSegment.created_at.desc()
                    ).label('rn')
                )
            ).subquery()

            # 只选择每个视频的最新记录
            results = session.query(
                subquery.c.original_filename,
                subquery.c.video_md5,
                subquery.c.created_at.label('process_time'),
                case(
                    (subquery.c.video_md5.isnot(None), 'success'),
                    else_='failed'
                ).label('status')
            ).filter(subquery.c.rn == 1).all()

            videos = []
            for row in results:
                videos.append({
                    'original_filename': row[0],
                    'video_md5': row[1],
                    'process_time': row[2],
                    'status': row[3]
                })

            print(f"查询到 {len(videos)} 个唯一视频")
            for video in videos:
                print(f"- {video['original_filename']} ({video['video_md5']})")

            return videos
        except Exception as e:
            print(f"获取视频列表失败: {str(e)}")
            return []
        finally:
            session.close()

    def clear_database(self):
        """清空数据库中的所有数据和相关文件"""
        session = self.Session()
        try:
            # 使用get_processed_videos_dir获取正确的路径
            processed_dir = self.get_processed_videos_dir()
            print(f"准备清空处理目录: {processed_dir}")
            
            # 清空processed_videos下的所有子文件夹
            if os.path.exists(processed_dir):
                print(f"处理目录存在: {processed_dir}")
                # 扩展子目录列表，包含所有可能的目录
                subdirs = ['audio', 'subtitles', 'temp', 'metadata']
                
                # 清理所有子目录
                for subdir in subdirs:
                    dir_path = os.path.join(processed_dir, subdir)
                    print(f"检查子目录: {dir_path}")
                    if os.path.exists(dir_path):
                        print(f"正在清空目录: {dir_path}")
                        files = os.listdir(dir_path)
                        print(f"目录 {subdir} 中的文件数量: {len(files)}")
                        for file in files:
                            file_path = os.path.join(dir_path, file)
                            try:
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                                    print(f"已删除文件: {file_path}")
                                elif os.path.isdir(file_path):
                                    # 如果是子目录，递归删除其中的文件
                                    for root, dirs, files in os.walk(file_path, topdown=False):
                                        for name in files:
                                            try:
                                                os.remove(os.path.join(root, name))
                                                print(f"已删除文件: {os.path.join(root, name)}")
                                            except Exception as e:
                                                print(f"删除文件失败 {os.path.join(root, name)}: {str(e)}")
                                else:
                                    print(f"不是文件或目录，跳过: {file_path}")
                            except Exception as e:
                                print(f"删除失败 {file_path}: {str(e)}")
                    else:
                        print(f"目录不存在: {dir_path}")
                
                # 清理processed_videos目录下的其他文件（如果有）
                for file in os.listdir(processed_dir):
                    file_path = os.path.join(processed_dir, file)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                            print(f"已删除根目录文件: {file_path}")
                        except Exception as e:
                            print(f"删除文件失败 {file_path}: {str(e)}")
            else:
                print(f"处理目录不存在: {processed_dir}")

            # 删除数据库记录
            count = session.query(VideoSegment).delete()
            session.commit()
            print(f"数据库记录已清空，共删除 {count} 条记录")
            print("数据库和相关文件已清空完成")
        except Exception as e:
            session.rollback()
            print(f"清空数据库失败: {str(e)}")
            raise
        finally:
            session.close()

    def get_processed_videos_dir(self):
        """获取processed_videos目录的路径"""
        # 使用当前工作目录，这是应用程序运行的目录
        workspace_dir = os.getcwd()
        processed_dir = os.path.join(workspace_dir, 'processed_videos')
        return processed_dir 