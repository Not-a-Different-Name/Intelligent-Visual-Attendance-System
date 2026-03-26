import cv2
import numpy as np
import os
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from vision_core import VisionProcessor
from face_recognizer import FaceRecognizerLogger

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_data_signal = pyqtSignal(int, int, float)
    
    def __init__(self, vision_processor):
        super().__init__()
        self.vision = vision_processor
        self.running = True
        self.threshold = -5
        self.current_frame = None

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                processed_frame, count, rate, up_count = self.vision.process_frame(frame, self.threshold)
                self.current_frame = frame.copy() 
                
                self.change_pixmap_signal.emit(processed_frame)
                self.update_data_signal.emit(count, up_count, rate)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class RosterDialog(QDialog):
    def __init__(self, recognizer, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self.setWindowTitle("人脸数据库图录与管理")
        self.resize(750, 500)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QHBoxLayout(self)
        
        # 左侧列表和删除按钮组合区
        left_layout = QVBoxLayout()
        
        self.list_widget = QListWidget()
        self.list_widget.setFixedWidth(200)
        self.list_widget.setFont(QFont("Microsoft YaHei", 12))
        self.list_widget.itemClicked.connect(self.display_photo)
        left_layout.addWidget(self.list_widget)
        
        # 【新增排版】：删除按钮
        self.btn_delete = QPushButton("🗑️ 删除选中人员")
        self.btn_delete.setStyleSheet("background-color: #F44336; color: white; font-weight: bold; padding: 8px; border-radius: 4px;")
        self.btn_delete.clicked.connect(self.delete_selected_user)
        left_layout.addWidget(self.btn_delete)
        
        layout.addLayout(left_layout)
        
        # 右侧照片区
        self.photo_label = QLabel("请在左边选择人员以查看录入时的抓拍")
        self.photo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        layout.addWidget(self.photo_label, stretch=1)
        
        self.load_data()
        
    def load_data(self):
        self.list_widget.clear()
        self.db_data = self.recognizer.get_attendance_data()
        for name in self.db_data.keys():
            if name != "Unknown":
                self.list_widget.addItem(name)
            
    def display_photo(self, item):
        name = item.text()
        photo_filename = self.db_data.get(name, {}).get("photo", "")
        if photo_filename:
            photo_path = os.path.join(self.recognizer.faces_dir, photo_filename)
            if os.path.exists(photo_path):
                pixmap = QPixmap(photo_path)
                pixmap = pixmap.scaled(500, 450, Qt.AspectRatioMode.KeepAspectRatio)
                self.photo_label.setPixmap(pixmap)
                return
        self.photo_label.clear()
        self.photo_label.setText("找不到该人员的照片数据")

    # 【新增功能】：删除交互逻辑
    def delete_selected_user(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "提示", "请先在上方列表中点击选择要删除的人员！")
            return
            
        name = selected_items[0].text()
        
        # 二次确认弹窗
        reply = QMessageBox.question(
            self, "确认删除", 
            f"确定要彻底删除人员 【{name}】 吗？\n删除后照片特征与所有签到记录将不可恢复！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            success, msg = self.recognizer.delete_user(name)
            if success:
                QMessageBox.information(self, "删除成功", msg)
                self.photo_label.clear()
                self.photo_label.setText("请在左边选择人员以查看录入时的抓拍")
                self.load_data() # 刷新当前窗口列表
                
                # 同步刷新主界面的数据表格
                if self.parent() and hasattr(self.parent(), "control_window"):
                    if self.parent().control_window:
                        self.parent().control_window.refresh_table()
            else:
                QMessageBox.warning(self, "删除失败", msg)

class ControlWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("数据分析与操作控制台")
        self.resize(400, 700)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title_font = QFont("Microsoft YaHei", 14, QFont.Weight.Bold)
        content_font = QFont("Microsoft YaHei", 12)

        lbl_title1 = QLabel("实时数据监控")
        lbl_title1.setFont(title_font)
        layout.addWidget(lbl_title1)

        self.lbl_count = QLabel("画面人数: 0")
        self.lbl_count.setFont(content_font)
        self.lbl_up = QLabel("抬头人数: 0")
        self.lbl_up.setFont(content_font)
        self.lbl_rate = QLabel("抬头率: 0.0%")
        self.lbl_rate.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        self.lbl_rate.setStyleSheet("color: #1976D2;")
        
        layout.addWidget(self.lbl_count)
        layout.addWidget(self.lbl_up)
        layout.addWidget(self.lbl_rate)
        
        layout.addWidget(self.create_line())

        self.lbl_slider = QLabel("抬头判定角度阈值: -5")
        self.lbl_slider.setFont(QFont("Microsoft YaHei", 10))
        layout.addWidget(self.lbl_slider)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(-50)
        self.slider.setMaximum(50)
        self.slider.setValue(-5)
        self.slider.valueChanged.connect(self.on_slider_change)
        layout.addWidget(self.slider)

        btn_calibrate = QPushButton("🎯 将当前平视姿态设为 0° 基准")
        btn_calibrate.setMinimumHeight(35)
        btn_calibrate.setStyleSheet("background-color: #607D8B; color: white; border-radius: 5px;")
        btn_calibrate.clicked.connect(self.main_window.calibrate_baseline)
        layout.addWidget(btn_calibrate)

        layout.addWidget(self.create_line())

        lbl_title2 = QLabel("人脸数据库管理")
        lbl_title2.setFont(title_font)
        layout.addWidget(lbl_title2)

        btn_layout = QHBoxLayout()
        btn_register = QPushButton("📸 录入新面孔")
        btn_register.setMinimumHeight(40)
        btn_register.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; border-radius: 5px;")
        btn_register.clicked.connect(self.main_window.register_face)
        
        btn_view_list = QPushButton("📋 查看/管理名录")
        btn_view_list.setMinimumHeight(40)
        btn_view_list.setStyleSheet("background-color: #00BCD4; color: white; font-weight: bold; border-radius: 5px;")
        btn_view_list.clicked.connect(self.main_window.view_registered_faces)
        
        btn_layout.addWidget(btn_register)
        btn_layout.addWidget(btn_view_list)
        layout.addLayout(btn_layout)

        layout.addWidget(self.create_line())

        btn_recognize = QPushButton("✅ 截屏并考勤打卡")
        btn_recognize.setMinimumHeight(50)
        btn_recognize.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; font-weight: bold; border-radius: 5px;")
        btn_recognize.clicked.connect(self.main_window.trigger_recognition)
        layout.addWidget(btn_recognize)

        lbl_title3 = QLabel("JSON签到统计")
        lbl_title3.setFont(title_font)
        layout.addWidget(lbl_title3)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["姓名", "次数", "最新签到时间"])
        
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)
        
        self.refresh_table()

    def create_line(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #E0E0E0;")
        return line

    def on_slider_change(self, value):
        self.lbl_slider.setText(f"抬头判定角度阈值: {value}")
        if hasattr(self.main_window, 'video_thread'):
            self.main_window.video_thread.threshold = value

    def refresh_table(self):
        data = self.main_window.recognizer.get_attendance_data()
        self.table.setRowCount(len([k for k in data.keys() if k != "Unknown"]))
        row = 0
        for name, info in data.items():
            if name == "Unknown": continue 
            self.table.setItem(row, 0, QTableWidgetItem(str(name)))
            item_count = QTableWidgetItem(str(info.get("count", 0)))
            item_count.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, item_count)
            self.table.setItem(row, 2, QTableWidgetItem(str(info.get("last_sign_in", ""))))
            row += 1

    def closeEvent(self, event):
        self.main_window.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能视觉分析系统 - 视频监控")
        self.resize(800, 600)
        
        self.vision = VisionProcessor()
        self.recognizer = FaceRecognizerLogger()
        self.control_window = None
        self.video_thread = None
        
        self.setup_welcome_ui()

    def setup_welcome_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_title = QLabel("智能视觉分析系统")
        lbl_title.setFont(QFont("Microsoft YaHei", 28, QFont.Weight.Bold))
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_title)
        
        lbl_sub = QLabel("点击下方按钮分离出控制台与视频窗口")
        lbl_sub.setFont(QFont("Microsoft YaHei", 12))
        lbl_sub.setStyleSheet("color: #666;")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_sub)
        
        layout.addSpacing(50)
        
        btn_start = QPushButton("启动系统 (双窗口)")
        btn_start.setFixedSize(250, 60)
        btn_start.setStyleSheet("background-color: #2196F3; color: white; font-size: 18px; border-radius: 8px;")
        btn_start.clicked.connect(self.start_system)
        layout.addWidget(btn_start, alignment=Qt.AlignmentFlag.AlignCenter)

    def start_system(self):
        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.video_label)
        
        self.control_window = ControlWindow(self)
        self.control_window.show()
        
        self.video_thread = VideoThread(self.vision)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_data_signal.connect(self.update_data)
        self.video_thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        
        scaled_img = qt_img.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_img))

    @pyqtSlot(int, int, float)
    def update_data(self, count, up_count, rate):
        if self.control_window:
            self.control_window.lbl_count.setText(f"画面人数: {count}")
            self.control_window.lbl_up.setText(f"抬头人数: {up_count}")
            self.control_window.lbl_rate.setText(f"抬头率: {rate:.1f}%")

    def calibrate_baseline(self):
        if not self.video_thread or self.video_thread.current_frame is None:
            QMessageBox.warning(self, "提示", "画面未准备好！")
            return
            
        success = self.vision.calibrate_baseline(self.video_thread.current_frame)
        if success:
            QMessageBox.information(self, "校准成功", "已将画面中检测到的人脸平视姿态设为 0 度基准。")
        else:
            QMessageBox.warning(self, "校准失败", "未能检测到人脸，请面对摄像头重试！")

    def register_face(self):
        if not self.video_thread or self.video_thread.current_frame is None:
            QMessageBox.warning(self, "提示", "摄像头画面未准备好！")
            return
            
        name, ok = QInputDialog.getText(self, "录入新面孔", "请输入需要录入的姓名：\n(若画面有���人，只锁定面部最大者提取特征)")
        if ok and name.strip():
            frame_to_save = self.video_thread.current_frame.copy()
            success, msg = self.recognizer.register_new_face(frame_to_save, name.strip())
            
            if success:
                QMessageBox.information(self, "成功", msg)
                if self.control_window:
                    self.control_window.refresh_table()
            else:
                QMessageBox.critical(self, "录入失败", msg)

    def view_registered_faces(self):
        dialog = RosterDialog(self.recognizer, self)
        dialog.exec()

    def trigger_recognition(self):
        if not self.video_thread or self.video_thread.current_frame is None:
            return
            
        frame_to_recognize = self.video_thread.current_frame.copy()
        names = self.recognizer.recognize_and_log(frame_to_recognize)
        
        valid_names = [n for n in names if n != "Unknown"]
        
        if not valid_names:
            QMessageBox.information(self, "打卡报告", "未能成功识别到已录入的人员！")
        else:
            QMessageBox.information(self, "打卡报告", f"打卡成功:\n{', '.join(valid_names)}\n\n签到记录已更新至 JSON")
            if self.control_window:
                self.control_window.refresh_table()

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        if self.control_window:
            self.control_window.close()
        event.accept()