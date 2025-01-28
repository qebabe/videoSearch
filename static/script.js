document.addEventListener('DOMContentLoaded', function() {
    const processForm = document.getElementById('processForm');
    const processStatus = document.getElementById('processStatus');
    const processList = document.getElementById('processList');
    const searchInput = document.getElementById('searchInput');
    const searchButton = document.getElementById('searchButton');
    const searchResults = document.getElementById('searchResults');
    const gpuStatus = document.getElementById('gpuStatus');
    const browseBtn = document.getElementById('browseBtn');
    const videoDirInput = document.getElementById('videoDir');
    const refreshVideosBtn = document.getElementById('refreshVideosBtn');
    const clearDatabaseBtn = document.getElementById('clearDatabaseBtn');
    const processedVideosList = document.getElementById('processedVideosList');

    // 初始化加载
    loadProcessedVideos();

    // 刷新按钮点击事件
    refreshVideosBtn.addEventListener('click', loadProcessedVideos);

    // 清空数据库按钮点击事件
    clearDatabaseBtn.addEventListener('click', async function() {
        if (!confirm('确定要清空数据库吗？此操作不可恢复！')) {
            return;
        }

        try {
            const response = await fetch('/clear-database', {
                method: 'POST'
            });
            const data = await response.json();

            if (response.ok) {
                showStatus('数据库已清空', 'success');
                loadProcessedVideos(); // 刷新视频列表
            } else {
                throw new Error(data.error || '清空数据库失败');
            }
        } catch (error) {
            showStatus('清空数据库失败: ' + error.message, 'danger');
        }
    });

    // 加载已处理的视频列表
    async function loadProcessedVideos() {
        try {
            const response = await fetch('/processed-videos');
            const videos = await response.json();
            
            if (response.ok) {
                displayProcessedVideos(videos);
            } else {
                throw new Error(videos.error || '获取视频列表失败');
            }
        } catch (error) {
            processedVideosList.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center text-danger">
                        加载视频列表失败: ${error.message}
                    </td>
                </tr>
            `;
        }
    }

    // 显示已处理的视频列表
    function displayProcessedVideos(videos) {
        // 更新统计信息
        const totalVideos = videos ? videos.length : 0;
        const successVideos = videos ? videos.filter(v => v.status === 'success').length : 0;
        const failedVideos = totalVideos - successVideos;
        const lastProcessed = videos && videos.length > 0 ? 
            new Date(Math.max(...videos.map(v => new Date(v.process_time)))).toLocaleString() : '-';

        document.getElementById('totalVideos').textContent = totalVideos;
        document.getElementById('successVideos').textContent = successVideos;
        document.getElementById('failedVideos').textContent = failedVideos;
        document.getElementById('lastProcessed').textContent = lastProcessed;

        // 显示视频列表
        if (!videos || videos.length === 0) {
            processedVideosList.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center">
                        暂无已处理的视频
                    </td>
                </tr>
            `;
            return;
        }

        // 按处理时间降序排序
        videos.sort((a, b) => new Date(b.process_time) - new Date(a.process_time));

        processedVideosList.innerHTML = videos.map(video => `
            <tr>
                <td>${video.original_filename}</td>
                <td>${new Date(video.process_time).toLocaleString()}</td>
                <td>
                    <span class="badge bg-${video.status === 'success' ? 'success' : 'warning'}">
                        ${video.status === 'success' ? '处理成功' : '处理失败'}
                    </span>
                </td>
                <td>
                    <button class="btn btn-sm btn-outline-primary" 
                            onclick="searchVideo('${video.video_md5}')">
                        查看内容
                    </button>
                </td>
            </tr>
        `).join('');
    }

    // 搜索特定视频的内容
    window.searchVideo = function(videoMd5) {
        searchInput.value = `video:${videoMd5}`;
        performSearch();
    }

    // 检查GPU状态
    async function checkGPUStatus() {
        try {
            const response = await fetch('/gpu-status');
            const data = await response.json();
            
            if (data.available) {
                gpuStatus.innerHTML = `
                    <div class="alert alert-success mb-0">
                        <strong>GPU可用</strong><br>
                        设备: ${data.device_name}<br>
                        GPU数量: ${data.device_count}<br>
                        已分配内存: ${data.memory_allocated}<br>
                        缓存内存: ${data.memory_cached}
                    </div>
                `;
            } else {
                gpuStatus.innerHTML = `
                    <div class="alert alert-warning mb-0">
                        <strong>GPU不可用</strong><br>
                        将使用CPU进行处理（处理速度可能较慢）
                    </div>
                `;
            }
        } catch (error) {
            gpuStatus.innerHTML = `
                <div class="alert alert-danger mb-0">
                    检查GPU状态时发生错误: ${error.message}
                </div>
            `;
        }
    }

    // 定期检查GPU状态
    checkGPUStatus();
    setInterval(checkGPUStatus, 30000);  // 每30秒检查一次

    // 处理目录选择
    browseBtn.addEventListener('click', function() {
        // 由于浏览器安全限制，我们不能直接访问文件系统
        // 这里只是提示用户手动输入路径
        alert('出于浏览器安全限制，请直接在输入框中粘贴目录路径。');
    });

    // 处理目录提交
    processForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const directory = videoDirInput.value.trim();
        if (!directory) {
            showStatus('请输入视频目录路径', 'danger');
            return;
        }
        
        try {
            showStatus('正在开始处理...', 'info');
            
            const response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ directory: directory })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                const taskId = data.task_id;
                showStatus('开始处理目录中的视频文件...', 'info');
                pollTaskStatus(taskId);
            } else {
                showStatus(data.error || '处理失败', 'danger');
            }
        } catch (error) {
            showStatus('处理过程中发生错误: ' + error.message, 'danger');
        }
    });

    // 轮询任务状态
    async function pollTaskStatus(taskId) {
        try {
            const response = await fetch(`/task/${taskId}`);
            const data = await response.json();
            
            if (response.ok) {
                if (data.status === 'completed') {
                    showStatus(data.message, 'success');
                    checkGPUStatus();
                    loadProcessedVideos(); // 刷新视频列表
                } else if (data.status === 'completed_with_errors') {
                    showStatus(data.message, 'warning');
                    loadProcessedVideos(); // 刷新视频列表
                    if (data.failed_files && data.failed_files.length > 0) {
                        processList.innerHTML = `
                            <div class="alert alert-warning">
                                <h6>处理失败的文件：</h6>
                                <ul>
                                    ${data.failed_files.map(file => `<li>${file}</li>`).join('')}
                                </ul>
                            </div>
                        `;
                    }
                } else if (data.status === 'error') {
                    showStatus('处理失败: ' + data.message, 'danger');
                } else {
                    showStatus(data.message, 'info');
                    if (data.progress) {
                        const progress = data.progress;
                        const percent = ((progress.current / progress.total) * 100).toFixed(1);
                        processList.innerHTML = `
                            <div class="progress mb-2">
                                <div class="progress-bar" role="progressbar" style="width: ${percent}%">
                                    ${percent}%
                                </div>
                            </div>
                            <div class="text-muted">
                                已处理: ${progress.current}/${progress.total}
                                ${progress.failed > 0 ? `(失败: ${progress.failed})` : ''}
                            </div>
                        `;
                    }
                    setTimeout(() => pollTaskStatus(taskId), 2000);
                }
            } else {
                showStatus('获取任务状态失败', 'danger');
            }
        } catch (error) {
            showStatus('检查任务状态时发生错误: ' + error.message, 'danger');
        }
    }

    // 处理搜索
    searchButton.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            performSearch();
        }
    });

    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) {
            return;
        }
        
        try {
            searchResults.innerHTML = '<div class="text-center">正在搜索...</div>';
            
            const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
            const results = await response.json();
            
            if (results.length === 0) {
                searchResults.innerHTML = '<div class="alert alert-info">未找到匹配的内容</div>';
                return;
            }
            
            // 按视频分组显示结果
            const groupedResults = {};
            results.forEach(result => {
                if (!groupedResults[result.video_md5]) {
                    groupedResults[result.video_md5] = {
                        filename: result.original_filename,
                        path: result.original_path,
                        segments: []
                    };
                }
                groupedResults[result.video_md5].segments.push(result);
            });
            
            searchResults.innerHTML = Object.entries(groupedResults).map(([md5, video], index) => `
                <div class="video-result mb-4">
                    <div class="card">
                        <div class="card-header collapsed" 
                             data-bs-toggle="collapse" 
                             data-bs-target="#video-collapse-${md5}" 
                             aria-expanded="false" 
                             aria-controls="video-collapse-${md5}">
                            <div class="d-flex justify-content-between align-items-center">
                                <div class="d-flex align-items-center">
                                    <i class="collapse-icon bi bi-chevron-down me-2"></i>
                                    <h5 class="mb-0">${video.filename}</h5>
                                </div>
                                <span class="badge bg-primary">${video.segments.length} 个片段</span>
                            </div>
                        </div>
                        <div id="video-collapse-${md5}" class="collapse" data-bs-parent="#searchResults">
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-12 mb-3">
                                        <div class="video-container">
                                            <video id="video-${md5}" class="w-100" controls preload="metadata">
                                                <source src="/video/${encodeURIComponent(video.path)}" type="video/mp4">
                                                您的浏览器不支持视频播放
                                            </video>
                                        </div>
                                    </div>
                                </div>
                                <div class="segments-container">
                                    ${video.segments.map(segment => `
                                        <div class="segment-item p-2 border">
                                            <div class="d-flex justify-content-between align-items-center">
                                                <div class="timestamp">
                                                    <button class="btn btn-sm btn-outline-primary" 
                                                            onclick="seekVideo('${md5}', ${segment.start_time})">
                                                        ▶ ${formatTime(segment.start_time)}
                                                    </button>
                                                    - ${formatTime(segment.end_time)}
                                                </div>
                                                <div class="duration badge bg-secondary">
                                                    时长: ${formatTime(segment.end_time - segment.start_time)}
                                                </div>
                                            </div>
                                            <div class="text mt-2">
                                                ${segment.text}
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `).join('');
            
        } catch (error) {
            searchResults.innerHTML = `
                <div class="alert alert-danger">
                    搜索过程中发生错误: ${error.message}
                </div>
            `;
        }
    }

    // 跳转视频到指定时间点
    window.seekVideo = function(videoMd5, time) {
        const video = document.getElementById(`video-${videoMd5}`);
        if (video) {
            video.currentTime = time;
            video.play().catch(e => {
                console.error('视频播放失败:', e);
                alert('视频播放失败。请确保视频文件存在且可访问。');
            });
        }
    }

    // 辅助函数
    function showStatus(message, type = 'info') {
        processStatus.className = `alert alert-${type}`;
        processStatus.classList.remove('d-none');
        processStatus.textContent = message;
    }

    function formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }
}); 