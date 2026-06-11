#!/usr/bin/env python3
import cv2
import numpy as np
from pupil_apriltags import Detector
import time
import math
import subprocess
from typing import List, Dict

class AprilTagSSHDetector:
    def __init__(self, tag_size: float = 0.05):
        # Tracking configurations tuned for 320x240 resolution
        self.detector = Detector(
            families="tag36h11",
            nthreads=4,               
            quad_decimate=1.0,        # Set to 1.0 since resolution is smaller (keeps corners sharp)
            quad_sigma=0.4,           # Reduced blur to match lower resolution textures
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        self.tag_size = tag_size
        self.pipe_process = None
        
        self.camera = self._open_ssh_stream()
        if not self.camera or not self.camera.isOpened():
            self.cleanup()
            raise RuntimeError("Could not establish low-latency stream link with the Pi.")

        # Updated intrinsic camera matrix tuned directly for 320x240 stream metrics
        self.cam_params = [320.0, 320.0, 160.0, 120.0]
        self.K = np.array([
            [self.cam_params[0], 0,                  self.cam_params[2]],
            [0,                  self.cam_params[1], self.cam_params[3]],
            [0,                  0,                  1]
        ], dtype=np.float32)

        # Filters for smoothing spatial variance
        self.alpha = 0.35             # Slightly higher alpha makes responses snappier for flight control
        self.prev_R = None
        self.prev_t = None
        self.prev_yaw = None

    def _open_ssh_stream(self):
        print("Launching ultra-low-latency MJPEG stream over SSH...")
        
        # FIXED STREAM TUNING PIPELINE:
        # 1. Pi encodes to raw mjpeg frames over SSH stdout pipeline.
        # 2. Local ffmpeg reads raw 'mjpeg' protocol payload (-f mjpeg) and pipes it locally to port 5001.
        # 3. Setting output to '-f mjpeg' stops the local machine from using h264 decoders.
        cmd = (
            'bash -c \'ssh x11@x11.local "source ~/myenv/bin/activate && '
            'rpicam-vid -t 0 --inline --width 320 --height 240 --framerate 30 '
            '--codec mjpeg -n -o -" | ffmpeg -fflags nobuffer -flags low_delay '
            '-f mjpeg -i - -vcodec copy -f mjpeg udp://127.0.0.1:5001?pkt_size=1316\''
        )
        
        # Spin up network process pipe
        self.pipe_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Let network handshakes complete quickly
        time.sleep(1.0)
        
        print("Connecting tracking engine to zero-buffer loopback...")
        cap = cv2.VideoCapture("udp://127.0.0.1:5001", cv2.CAP_FFMPEG)
        
        # Clear out historical hardware buffer queues inside OpenCV backend
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return cap

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
            
            raw_yaw = math.degrees(math.atan2(-r[2, 0], math.sqrt(r[2, 1]**2 + r[2, 2]**2)))

            if self.prev_R is not None and self.prev_t is not None:
                r = self.alpha * r + (1.0 - self.alpha) * self.prev_R
                t = self.alpha * t + (1.0 - self.alpha) * self.prev_t
                raw_yaw = self.alpha * raw_yaw + (1.0 - self.alpha) * self.prev_yaw
                
                u, _, vt = np.linalg.svd(r)
                r = np.dot(u, vt)

            self.prev_R = r
            self.prev_t = t
            self.prev_yaw = raw_yaw

            pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
            roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))

            tags_info.append({
                'id': detection.tag_id,
                'center': (detection.center[0], detection.center[1]),
                'corners': detection.corners,
                'pose_R': r,    
                'pose_t': t,    
                'yaw': raw_yaw,
                'pitch': pitch,
                'roll': roll
            })
            
        if not detections:
            self.prev_R = None
            self.prev_t = None
            self.prev_yaw = None
            
        return tags_info
    
    def draw_3d_box(self, frame, tag) -> np.ndarray:
        R = tag['pose_R']
        t = tag['pose_t']
        
        s = self.tag_size / 2.0
        box_3d = np.array([
            [-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0],         
            [-s, -s, -self.tag_size], [s, -s, -self.tag_size],       
            [s, s, -self.tag_size], [-s, s, -self.tag_size]
        ], dtype=np.float32)
        
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.astype(np.float32)
        
        img_pts, _ = cv2.projectPoints(box_3d, rvec, tvec, self.K, distCoeffs=None)
        img_pts = img_pts.reshape(-1, 2).astype(int)
        
        cv2.polylines(frame, [img_pts[0:4]], True, (255, 0, 0), 2)
        cv2.polylines(frame, [img_pts[4:8]], True, (0, 255, 255), 2)
        for i in range(4):
            cv2.line(frame, tuple(img_pts[i]), tuple(img_pts[i+4]), (0, 255, 0), 2)
            
        return frame

    def draw_telemetry_hud(self, frame, tags_info, fps: float) -> np.ndarray:
        box_x, box_y = 10, 10
        box_w, box_h = 230, 160
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (70, 70, 70), 1)
        
        title_color, label_color, val_color = (0, 255, 255), (170, 170, 170), (255, 255, 255)
        font, font_scale, spacing = cv2.FONT_HERSHEY_SIMPLEX, 0.38, 17
        
        status_text = "TRACKING" if tags_info else "SCANNING..."
        status_color = (0, 255, 0) if tags_info else (0, 165, 255)
        cv2.putText(frame, f"HUD LINK: {status_text}", (box_x + 8, box_y + 18), font, font_scale, status_color, 1, cv2.LINE_AA)
        cv2.line(frame, (box_x + 8, box_y + 24), (box_x + box_w - 8, box_y + 24), (45, 45, 45), 1)
        
        current_y = box_y + 40
        
        if tags_info:
            tag = tags_info[0]
            t = tag['pose_t'].flatten()
            distance_cm = math.sqrt(t[0]**2 + t[1]**2 + t[2]**2) * 100
            x_cm, y_cm, z_cm = t[0]*100, t[1]*100, t[2]*100
            
            telemetry_lines = [
                ("Target ID:", f"{tag['id']}", title_color),
                ("Distance:", f"{distance_cm:.1f} cm", (0, 255, 0)),
                ("Pos X (L/R):", f"{x_cm:+.1f} cm", val_color),
                ("Pos Y (U/D):", f"{y_cm:+.1f} cm", val_color),
                ("Pos Z (Dth):", f"{z_cm:.1f} cm", val_color),
                ("Yaw Angle:", f"{tag['yaw']:+.1f} deg", title_color),
                ("Pitch/Roll:", f"P:{tag['pitch']:+.0f} R:{tag['roll']:+.0f}", val_color)
            ]
            
            for label, val, color in telemetry_lines:
                cv2.putText(frame, label, (box_x + 8, current_y), font, font_scale, label_color, 1, cv2.LINE_AA)
                cv2.putText(frame, val, (box_x + 105, current_y), font, font_scale, color, 1, cv2.LINE_AA)
                current_y += spacing
        else:
            cv2.putText(frame, "Awaiting valid stream sync...", (box_x + 8, current_y), font, font_scale, (90, 90, 90), 1, cv2.LINE_AA)
            
        cv2.putText(frame, f"Low-Latency Loop: {fps:.1f} FPS", (box_x + 8, box_y + box_h - 8), font, 0.32, (100, 100, 100), 1, cv2.LINE_AA)
        return frame
        
    def run(self, display: bool = True):
        frame_count = 0
        start_time = time.time()
        current_fps = 0.0
        
        try:
            while True:
                # Forcefully clear system stream buffers before fetching image data
                # Ensures OpenCV doesn't read sequential backlog lines
                for _ in range(2):
                    self.camera.grab()
                
                ret, frame = self.camera.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                tags_info = self.detect_tags(frame)
                
                frame_count += 1
                if frame_count % 15 == 0:
                    current_fps = frame_count / (time.time() - start_time)
                
                if display:
                    for tag in tags_info:
                        frame = self.draw_3d_box(frame, tag)
                    frame = self.draw_telemetry_hud(frame, tags_info, current_fps)
                    cv2.imshow('AprilTag Sync - Remote Pi Cam', frame)
                
                if display and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("\nShutting down stream connection...")
        finally:
            self.cleanup()
            
    def cleanup(self):
        if self.camera and self.camera.isOpened():
            self.camera.release()
        if self.pipe_process:
            self.pipe_process.terminate()
            self.pipe_process.wait()
        cv2.destroyAllWindows()
        print("Disconnected and cleaned up resources.")

if __name__ == "__main__":
    detector = AprilTagSSHDetector(tag_size=0.05)
    detector.run(display=True)
