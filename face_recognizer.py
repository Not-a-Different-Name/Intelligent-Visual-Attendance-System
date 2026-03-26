import cv2
import os
import datetime
import face_recognition
import numpy as np
import json

class FaceRecognizerLogger:
    def __init__(self, log_dir="data/logs", faces_dir="data/known_faces"):
        self.log_dir = log_dir
        self.faces_dir = faces_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.faces_dir, exist_ok=True)
        
        self.json_path = os.path.join(self.log_dir, "attendance_db.json")
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
    def load_known_faces(self):
        self.known_face_encodings.clear()
        self.known_face_names.clear()
        for filename in os.listdir(self.faces_dir):
            if filename.lower().endswith((".jpg", ".png")) and not filename.endswith("_display.jpg"):
                filepath = os.path.join(self.faces_dir, filename)
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(os.path.splitext(filename)[0])

    def get_attendance_data(self):
        if os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # 【新增方法】：删除用户的所有数据
    def delete_user(self, name):
        """删除指定人员的照片特征和 JSON 签到记录"""
        success = False
        
        # 1. 尝试删除本地两张照片
        raw_path = os.path.join(self.faces_dir, f"{name}.jpg")
        display_path = os.path.join(self.faces_dir, f"{name}_display.jpg")
        
        if os.path.exists(raw_path):
            os.remove(raw_path)
            success = True
        if os.path.exists(display_path):
            os.remove(display_path)
            
        # 2. 从 JSON 中移除数据
        db_data = self.get_attendance_data()
        if name in db_data:
            del db_data[name]
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(db_data, f, ensure_ascii=False, indent=4)
            success = True
            
        # 3. 重新加载内存特征库，防止程序继续识别出已删除的人
        self.load_known_faces()
        
        if success:
            return True, f"已成功彻底删除人员：{name}"
        else:
            return False, f"未找到人员 {name} 的相关数据。"

    def register_new_face(self, frame, name):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)
        
        if len(face_locations) == 0:
            return False, "未检测到人脸，请再靠���或正对摄像头一些。"
        
        largest_face = max(face_locations, key=lambda loc: (loc[2]-loc[0])*(loc[1]-loc[3]))
        top, right, bottom, left = largest_face
            
        h, w, _ = frame.shape
        margin_y = int((bottom - top) * 0.3)
        margin_x = int((right - left) * 0.3)
        top_m = max(0, top - margin_y)
        bottom_m = min(h, bottom + margin_y)
        left_m = max(0, left - margin_x)
        right_m = min(w, right + margin_x)
        
        face_crop = frame[top_m:bottom_m, left_m:right_m]
        raw_path = os.path.join(self.faces_dir, f"{name}.jpg")
        cv2.imwrite(raw_path, face_crop)
        
        annotated_frame = frame.copy()
        cv2.rectangle(annotated_frame, (left, top), (right, bottom), (255, 152, 0), 3)
        cv2.putText(annotated_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 152, 0), 2)
        display_photo_name = f"{name}_display.jpg"
        display_path = os.path.join(self.faces_dir, display_photo_name)
        cv2.imwrite(display_path, annotated_frame)
        
        db_data = self.get_attendance_data()
        if name not in db_data:
            db_data[name] = {"count": 0, "history": [], "last_sign_in": "", "photo": display_photo_name}
        else:
            db_data[name]["photo"] = display_photo_name
            
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(db_data, f, ensure_ascii=False, indent=4)

        self.load_known_faces() 
        return True, f"成功锁定提取并录入面孔：{name}"

    def recognize_and_log(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized_names = []
        for face_encoding in face_encodings:
            name = "Unknown"
            if len(self.known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.5:
                    name = self.known_face_names[best_match_index]
            recognized_names.append(name)
            
        if recognized_names:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_path = os.path.join(self.log_dir, "attendance_log.csv")
            with open(log_path, "a", encoding="utf-8") as f:
                for name in recognized_names:
                    f.write(f"{timestamp},{name}\n")
                    
            db_data = self.get_attendance_data()
            for name in recognized_names:
                if name != "Unknown":
                    if name not in db_data:
                        db_data[name] = {"count": 0, "history": [], "last_sign_in": "", "photo": ""}
                    db_data[name]["count"] += 1
                    db_data[name]["history"].append(timestamp)
                    db_data[name]["last_sign_in"] = timestamp
            
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(db_data, f, ensure_ascii=False, indent=4)
                
        return recognized_names