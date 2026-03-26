# 智能视觉分析与考勤系统 (Smart Vision Attendance)

基于 MediaPipe 和 Face_Recognition 开发的轻量级课堂/会议监控与考勤打卡系统。采用 PyQt6 构建现代化双屏 GUI，提供实时的抬头率监测、无感人脸录入与签到管理功能。

## ✨ 核心功能
- **实时抬头率检测**：利用 MediaPipe Face Mesh 进行 3D 头部姿态解算，支持一键 0° 基准校准。
- **动态人脸库管理**��支持锁定画幅最大人脸进行录入，自动截取特征照片。
- **多人考勤打卡**：自动提取人脸特征比对，支持远距离多人同时打卡。
- **本地 JSON 数据库**：自动记录打卡次数与最新签到时间，配合精美数据表格实时刷新。

## 🛠️ 安装与运行

1. 克隆本仓库：
   ```bash
   git clone https://github.com/Not-a-Different-Name/Intelligent-Visual-Attendance-System.git
   cd Intelligent-Visual-Attendance-System
   ```
2. 安装依赖 (推荐使用 Python 3.10+)：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行程序：
   ```bash
   python main.py
   ```