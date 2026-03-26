import cv2
import mediapipe as mp
import numpy as np

class VisionProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.face_3d = np.array([
            [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
        ], dtype=np.float64)
        
        self.baseline_pitch = 0.0 

    def calibrate_baseline(self, frame):
        """一键校准：计算画面中第一个人的当前角度，并设为 0 度基准"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            img_h, img_w, _ = frame.shape
            face_2d = []
            landmark_indices = [1, 199, 33, 263, 61, 291] 
            for idx in landmark_indices:
                lm = results.multi_face_landmarks[0].landmark[idx]
                face_2d.append([int(lm.x * img_w), int(lm.y * img_h)])
            
            face_2d = np.array(face_2d, dtype=np.float64)
            cam_matrix = np.array([[img_w, 0, img_w / 2], [0, img_w, img_h / 2], [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            
            success, rot_vec, trans_vec = cv2.solvePnP(self.face_3d, face_2d, cam_matrix, dist_matrix)
            if success:
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                x_angle = angles[0]
                if x_angle > 180: x_angle -= 360
                elif x_angle < -180: x_angle += 360
                self.baseline_pitch = x_angle
                return True
        return False

    def process_frame(self, frame, pitch_threshold=-5):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        person_count = 0
        head_up_count = 0
        
        if results.multi_face_landmarks:
            person_count = len(results.multi_face_landmarks)
            img_h, img_w, _ = frame.shape
            
            for face_landmarks in results.multi_face_landmarks:
                face_2d = []
                x_max, y_max = 0, 0
                x_min, y_min = img_w, img_h
                
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    if x < x_min: x_min = x
                    if x > x_max: x_max = x
                    if y < y_min: y_min = y
                    if y > y_max: y_max = y
                    
                box_color = (0, 255, 0)
                landmark_indices = [1, 199, 33, 263, 61, 291] 
                for idx in landmark_indices:
                    lm = face_landmarks.landmark[idx]
                    face_2d.append([int(lm.x * img_w), int(lm.y * img_h)])
                
                face_2d = np.array(face_2d, dtype=np.float64)
                cam_matrix = np.array([[img_w, 0, img_w / 2], [0, img_w, img_h / 2], [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                
                success, rot_vec, trans_vec = cv2.solvePnP(self.face_3d, face_2d, cam_matrix, dist_matrix)
                if success:
                    rmat, jac = cv2.Rodrigues(rot_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                    
                    raw_angle = angles[0]
                    if raw_angle > 180: raw_angle -= 360
                    elif raw_angle < -180: raw_angle += 360
                    
                    display_angle = raw_angle - self.baseline_pitch
                    if display_angle > 180: display_angle -= 360
                    elif display_angle < -180: display_angle += 360
                    
                    # 【核心修复】：加上负号，翻转坐标系！让抬头变为正数，低头变为负数！
                    display_angle = -display_angle
                    
                    if display_angle > pitch_threshold: 
                        head_up_count += 1
                        box_color = (0, 0, 255) 
                    
                    cv2.putText(frame, f"Ang:{display_angle:.1f}", (x_min, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                        
        head_up_rate = (head_up_count / person_count * 100) if person_count > 0 else 0
        return frame, person_count, head_up_rate, head_up_count