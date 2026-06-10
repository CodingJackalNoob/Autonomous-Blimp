#!/usr/bin/env python3
import cv2
import numpy as np
from pupil_apriltags import Detector
import time
import math
from typing import List, Tuple, Dict

class AprilTagDetector:
    def __init__(self, camera_id: int = 0, tag_size: float = 0.05):
        self.detector = Detector(
            families="tag36h11",
            nthreads=4,               # Increased to 4 threads for faster processing
            quad_decimate=1.5,        # Decreased slightly from 2.0 to improve corner detection accuracy
            quad_sigma=0.8,           # Added slight blur to smooth out camera sensor pixel noise
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        self.tag_size = tag_size
        self.camera = self._open_camera(camera_id)
        if not self.camera or not self.camera.isOpened():
            raise RuntimeError("No camera available.")
        self.setup_camera()

        # Camera intrinsic configurations
        self.cam_params = [640.0, 640.0, 320.0, 240.0]
        self.K = np.array([
            [self.cam_params[0], 0,                  self.cam_params[2]],
            [0,                  self.cam_params[1], self.cam_params[3]],
            [0,                  0,                  1]
        ], dtype=np.float32)

        # --- Smoothing Filter States ---
        # alpha controls smoothing (0.0 = frozen, 1.0 = raw/no smoothing). 
        # 0.25 removes high-frequency jitter while keeping low latency.
        self.alpha = 0.25 
        self.prev_R = None
        self.prev_t = None
        self.prev_yaw = None

    def _open_camera(self, camera_id: int):
        for candidate in [camera_id, 0, 1]:
            try:
                cap = cv2.VideoCapture(candidate)
                if cap is not None and cap.isOpened():
                    return cap
            except Exception:
                continue
        return None
        
    def setup_camera(self):
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        # Force the buffer size down to prevent historical frame pile-up lag
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
    def detect_tags(self, frame) -> List[Dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detections = self.detector.detect(
            gray, 
            estimate_tag_pose=True, 
            camera_params=self.cam_params,
            tag_size=self.tag_size
        )
        
        tags_info = []
        for detection in detections:
            r = detection.pose_R
            t = detection.pose_t
            
            # Calculate raw yaw angle
            raw_yaw = math.degrees(math.atan2(-r[2, 0], math.sqrt(r[2, 1]**2 + r[2, 2]**2)))

            # Apply Low-Pass Smoothing Filter if we have historical frames
            if self.prev_R is not None and self.prev_t is not None:
                r = self.alpha * r + (1.0 - self.alpha) * self.prev_R
                t = self.alpha * t + (1.0 - self.alpha) * self.prev_t
                raw_yaw = self.alpha * raw_yaw + (1.0 - self.alpha) * self.prev_yaw
                
                # Re-orthogonalize the filtered rotation matrix to keep 3D math clean
                u, _, vt = np.linalg.svd(r)
                r = np.dot(u, vt)

            # Store current states for the next frame
            self.prev_R = r
            self.prev_t = t
            self.prev_yaw = raw_yaw

            tags_info.append({
                'id': detection.tag_id,
                'center': (detection.center[0], detection.center[1]),
                'corners': detection.corners,
                'pose_R': r,    
                'pose_t': t,    
                'yaw': raw_yaw
            })
            
        # Clear filter history if the tag completely leaves the view
        if not detections:
            self.prev_R = None
            self.prev_t = None
            self.prev_yaw = None
            
        return tags_info
    
    def draw_3d_box(self, frame, tag) -> np.ndarray:
        R = tag['pose_R']
        t = tag['pose_t']
        
        s = self.tag_size / 2.0
        # Box points: Extruded along the Z axis (out of the tag surface face)
        box_3d = np.array([
            [-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0],         # Base
            [-s, -s, -self.tag_size], [s, -s, -self.tag_size],       # Top face
            [s, s, -self.tag_size], [-s, s, -self.tag_size]
        ], dtype=np.float32)
        
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.astype(np.float32)
        
        img_pts, _ = cv2.projectPoints(box_3d, rvec, tvec, self.K, distCoeffs=None)
        img_pts = img_pts.reshape(-1, 2).astype(int)
        
        # Render visual cube layers
        cv2.polylines(frame, [img_pts[0:4]], True, (255, 0, 0), 2)
        cv2.polylines(frame, [img_pts[4:8]], True, (0, 255, 255), 2)
        for i in range(4):
            cv2.line(frame, tuple(img_pts[i]), tuple(img_pts[i+4]), (0, 255, 0), 2)
            
        return frame

    def draw_rotation_bar(self, frame, tags_info) -> np.ndarray:
        height, width = frame.shape[:2]
        center_x = width // 2
        
        # Render HUD backgrounds
        cv2.rectangle(frame, (50, 20), (width - 50, 45), (30, 30, 30), -1)
        cv2.rectangle(frame, (50, 20), (width - 50, 45), (70, 70, 70), 1)
        cv2.line(frame, (center_x, 15), (center_x, 50), (200, 200, 200), 1)

        if tags_info:
            yaw = tags_info[0]['yaw']
            clipped_yaw = max(-90.0, min(90.0, yaw))
            max_bar_width = (width - 100) // 2
            offset = int((clipped_yaw / 90.0) * max_bar_width)
            
            bar_color = (0, 220, 0) if abs(yaw) < 20 else (0, 140, 255) if abs(yaw) < 45 else (0, 0, 240)
            cv2.rectangle(frame, (center_x, 23), (center_x + offset, 42), bar_color, -1)
            cv2.putText(frame, f"Rotation: {yaw:.1f} deg", (55, 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No Target Locked", (55, 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
        return frame
        
    def run(self, display: bool = True):
        print("Running stabilized tracking loop. Press 'q' to close.")
        
        try:
            while True:
                # Flush the OpenCV buffer structure by grabbing the absolute latest frame
                # This stops the video stream from presenting delayed historical data
                if self.camera.get(cv2.CAP_PROP_BUFFERSIZE) > 0:
                    self.camera.grab()
                
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                tags_info = self.detect_tags(frame)
                
                if display:
                    for tag in tags_info:
                        frame = self.draw_3d_box(frame, tag)
                        center = tag['center']
                        cv2.putText(frame, f"ID: {tag['id']}", 
                                   (int(center[0]) - 20, int(center[1]) - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    frame = self.draw_rotation_bar(frame, tags_info)
                    cv2.imshow('Seamless AprilTag Tracking', frame)
                
                if display and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = AprilTagDetector(camera_id=0, tag_size=0.05)
    detector.run(display=True)
