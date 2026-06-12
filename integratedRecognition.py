from __future__ import annotations
import sys
import cv2
import numpy as np
import subprocess
from collections import Counter, deque
from dataclasses import dataclass, field

# --- 1. FRAME CONFIGURATIONS ---
WIDTH = 640
HEIGHT = 480
FRAME_SIZE = WIDTH * HEIGHT * 3

# --- 2. GESTURE ENGINE PARAMETERS ---
CALIBRATION_FRAMES = 30
BACKGROUND_WEIGHT = 0.5
OBJECT_THRESHOLD = 23  
GESTURE_HISTORY = 12          
MIN_HAND_AREA = 2500

ROI_TOP = 40
ROI_BOTTOM = 360
ROI_LEFT = 300
ROI_RIGHT = 620

CONFIRMATION_THRESHOLD = 30   
LOCK_THRESHOLD = 120          

BOX_COLOR = (0, 255, 0)         
HUD_BG_COLOR = (20, 20, 20)     
HUD_TEXT_COLOR = (255, 255, 255)
HUD_ACCENT_COLOR = (0, 255, 0)  
HUD_LOCK_COLOR = (0, 165, 255)  
DEBUG_COLOR = (0, 255, 255)    

# --- 3. DATA STRUCTURES & GLOBAL TRACKERS ---
@dataclass
class HandData:
    top: tuple[int, int]
    bottom: tuple[int, int]
    left: tuple[int, int]
    right: tuple[int, int]
    center_x: int
    fingers: int = 0
    is_in_frame: bool = True
    gesture_history: deque[int] = field(default_factory=lambda: deque(maxlen=GESTURE_HISTORY))

    def update(self, top: tuple[int, int], bottom: tuple[int, int], left: tuple[int, int], right: tuple[int, int]) -> None:
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.center_x = int((left[0] + right[0]) / 2)
        self.is_in_frame = True

    def update_fingers(self, finger_count: int) -> None:
        self.gesture_history.append(finger_count)
        if len(self.gesture_history) == self.gesture_history.maxlen:
            self.fingers = Counter(self.gesture_history).most_common(1)[0][0]

front_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

background: np.ndarray | None = None
hand: HandData | None = None
frames_elapsed = 0
people_count = 0  

last_face = None       
lost_frames_count = 0  
MAX_LOST_FRAMES = 10     
SMOOTHING_FACTOR = 0.25  

confirmed_gesture = "NONE"
confirmed_fingers = 0
confirmation_counter = 0
lock_counter = 0

# --- 4. GESTURE PROCESSING FUNCTIONS ---
def get_region(frame: np.ndarray) -> np.ndarray:
    region = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    region = cv2.GaussianBlur(region, (5, 5), 0)
    return region

def update_background(region: np.ndarray) -> None:
    global background
    if background is None:
        background = region.copy().astype("float")
        return
    cv2.accumulateWeighted(region, background, BACKGROUND_WEIGHT)

def segment_hand(region: np.ndarray, face_rect: tuple[int, int, int, int] | None) -> tuple[np.ndarray, np.ndarray] | None:
    global hand
    if background is None:
        return None

    diff = cv2.absdiff(background.astype(np.uint8), region)
    threshold = cv2.threshold(diff, OBJECT_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    # --- FIX: FACE MASKING LOGIC ---
    # If a face is tracked, mask it out of the hand threshold so hair/shoulders don't trigger fingers
    if face_rect is not None:
        fx, fy, fw, fh = face_rect
        # Pad the box upward and slightly right to capture hair outlines safely
        mask_x1 = max(0, fx - ROI_LEFT)
        mask_y1 = max(0, (fy - 40) - ROI_TOP)
        mask_x2 = min(ROI_RIGHT - ROI_LEFT, (fx + fw + 20) - ROI_LEFT)
        mask_y2 = min(ROI_BOTTOM - ROI_TOP, (fy + fh + 40) - ROI_TOP)
        
        # Draw a black cutout rectangle over the face region within the ROI matrix
        cv2.rectangle(threshold, (mask_x1, mask_y1), (mask_x2, mask_y2), 0, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_ERODE, kernel, iterations=1)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= MIN_HAND_AREA]

    if not contours:
        if hand is not None:
            hand.is_in_frame = False
        return None

    return threshold, max(contours, key=cv2.contourArea)

def update_hand_data(threshold: np.ndarray, contour: np.ndarray) -> None:
    global hand
    hull = cv2.convexHull(contour)
    top = tuple(hull[hull[:, :, 1].argmin()][0])
    bottom = tuple(hull[hull[:, :, 1].argmax()][0])
    left = tuple(hull[hull[:, :, 0].argmin()][0])
    right = tuple(hull[hull[:, :, 0].argmax()][0])
    center_x = int((left[0] + right[0]) / 2)

    if hand is None:
        hand = HandData(top=top, bottom=bottom, left=left, right=right, center_x=center_x)
    else:
        hand.update(top, bottom, left, right)

    hand.update_fingers(count_fingers(threshold))

def count_fingers(threshold: np.ndarray) -> int:
    if hand is None:
        return 0

    hand_height = hand.bottom[1] - hand.top[1]
    hand_width = hand.right[0] - hand.left[0]
    if hand_height <= 0 or hand_width <= 0:
        return 0

    line_height = int(hand.top[1] + 0.30 * hand_height)
    line_height = max(0, min(line_height, threshold.shape[0] - 1))

    line_mask = np.zeros(threshold.shape[:2], dtype=np.uint8)
    cv2.line(line_mask, (0, line_height), (threshold.shape[1], line_height), 255, 1)
    intersections = cv2.bitwise_and(threshold, threshold, mask=line_mask)

    contours, _ = cv2.findContours(intersections.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    fingers = 0
    max_finger_width = 0.75 * hand_width
    for contour in contours:
        _, _, width, _ = cv2.boundingRect(contour)
        if 5 < width < max_finger_width:
            fingers += 1

    return min(fingers, 5)

# --- 5. LATCH CONTROLLER STATE ENGINE ---
def process_temporal_latch() -> tuple[str, int, bool]:
    global confirmation_counter, lock_counter, confirmed_gesture, confirmed_fingers
    
    if lock_counter > 0:
        lock_counter -= 1
        if lock_counter == 0:
            confirmed_gesture = "NONE"
            confirmed_fingers = 0
        return confirmed_gesture, confirmed_fingers, True

    if hand is None or not hand.is_in_frame:
        confirmation_counter = 0
        confirmed_gesture = "NONE"
        confirmed_fingers = 0
        return "NONE", 0, False

    raw_fingers = hand.fingers
    hand_height = hand.bottom[1] - hand.top[1]
    hand_width = hand.right[0] - hand.left[0]
    aspect_ratio = hand_width / max(1, hand_height)

    if raw_fingers == 1: 
        raw_gesture = "POINTING"
    elif raw_fingers == 2: 
        raw_gesture = "VICTORY / PEACE"
    elif raw_fingers == 3: 
        raw_gesture = "THREE COUNT"
    elif raw_fingers >= 4 and aspect_ratio > 0.65: 
        raw_gesture = "STOP (Open Hand)"
    else: 
        raw_gesture = "COME HERE (Fist)"
        raw_fingers = 0

    if raw_gesture != "NONE":
        confirmation_counter += 1
        if confirmation_counter >= CONFIRMATION_THRESHOLD:
            confirmed_gesture = raw_gesture
            confirmed_fingers = raw_fingers
            lock_counter = LOCK_THRESHOLD
            confirmation_counter = 0  
    else:
        confirmation_counter = max(0, confirmation_counter - 1) 

    return confirmed_gesture, confirmed_fingers, False

# --- 6. HUD & DEBUG TRACING RENDER ENGINE ---
def draw_hud_dashboard(frame: np.ndarray) -> None:
    box_x1, box_y1 = 10, 10
    box_x2, box_y2 = 280, 115
    
    display_gesture, display_fingers, is_locked = process_temporal_latch()
    accent = HUD_LOCK_COLOR if is_locked else HUD_ACCENT_COLOR
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), HUD_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame) 
    cv2.rectangle(frame, (box_x1, box_y1), (box_x1 + 4, box_y2), accent, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if frames_elapsed < CALIBRATION_FRAMES:
        cv2.putText(frame, f"People Detected: {people_count}", (box_x1 + 15, box_y1 + 25), font, 0.55, HUD_TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame, "Fingers: WAIT", (box_x1 + 15, box_y1 + 55), font, 0.55, HUD_TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame, "Gesture: CALIBRATING...", (box_x1 + 15, box_y1 + 85), font, 0.55, accent, 1, cv2.LINE_AA)
    else:
        f_str = f"Fingers: {display_fingers}"
        g_str = f"Gesture: {display_gesture}"
        
        if is_locked:
            pct = lock_counter / LOCK_THRESHOLD
            cv2.rectangle(frame, (box_x1 + 15, box_y2 - 8), (box_x1 + 15 + int(240 * pct), box_y2 - 4), HUD_LOCK_COLOR, -1)
        elif confirmation_counter > 0 and (hand is not None and hand.is_in_frame):
            pct = confirmation_counter / CONFIRMATION_THRESHOLD
            cv2.rectangle(frame, (box_x1 + 15, box_y2 - 8), (box_x1 + 15 + int(240 * pct), box_y2 - 4), HUD_ACCENT_COLOR, -1)

        cv2.putText(frame, f"People Detected: {people_count}", (box_x1 + 15, box_y1 + 25), font, 0.55, HUD_TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame, f_str, (box_x1 + 15, box_y1 + 55), font, 0.55, HUD_TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame, g_str, (box_x1 + 15, box_y1 + 85), font, 0.55, accent, 1, cv2.LINE_AA)

def draw_debug_traces(frame: np.ndarray, contour: np.ndarray) -> None:
    if hand is None or not hand.is_in_frame:
        return

    shifted_contour = contour + np.array([ROI_LEFT, ROI_TOP])
    cv2.drawContours(frame, [shifted_contour], -1, DEBUG_COLOR, 1)

    hand_height = hand.bottom[1] - hand.top[1]
    line_y = int(ROI_TOP + hand.top[1] + 0.30 * hand_height)
    cv2.line(frame, (ROI_LEFT, line_y), (ROI_RIGHT, line_y), DEBUG_COLOR, 1, cv2.LINE_4)
    cv2.putText(frame, "SCAN LINE", (ROI_LEFT + 5, line_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, DEBUG_COLOR, 1, cv2.LINE_AA)

    cv2.circle(frame, (ROI_LEFT + hand.top[0], ROI_TOP + hand.top[1]), 4, (0, 0, 255), -1)

def draw_hand_box(frame: np.ndarray, contour: np.ndarray) -> None:
    x, y, w, h = cv2.boundingRect(contour)
    top_left = (ROI_LEFT + x, ROI_TOP + y)
    bottom_right = (ROI_LEFT + x + w, ROI_TOP + y + h)
    cv2.rectangle(frame, top_left, bottom_right, BOX_COLOR, 2)

def reset_calibration() -> None:
    global background, hand, frames_elapsed, confirmation_counter, lock_counter, confirmed_gesture, confirmed_fingers
    background = None
    hand = None
    frames_elapsed = 0
    confirmation_counter = 0
    lock_counter = 0
    confirmed_gesture = "NONE"
    confirmed_fingers = 0

# --- 7. MAIN FFMPEG PIPELINE EXECUTION ---
print("Initializing local FFMPEG decoder pipeline...")

command = [
    'ffmpeg',
    '-f', 'h264',
    '-fflags', 'nobuffer',
    '-flags', 'low_delay',
    '-probesize', '32',
    '-analyzeduration', '0',
    '-i', 'pipe:0',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    'pipe:1'
]

proc = subprocess.Popen(command, stdin=sys.stdin.buffer, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
print("Listening for video frames... Press 'q' to quit, 'r' to recalibrate hand frame.")

while True:
    raw_frame = proc.stdout.read(FRAME_SIZE)
    if len(raw_frame) != FRAME_SIZE:
        print("End of stream or connection lost.")
        break

    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.flip(frame, 1)

    region = get_region(frame)

    # Prioritize face/head cascading so we can immediately feed it into the gesture masking stage
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    faces = front_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(90, 90))

    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(90, 90))
        if len(faces) == 0:
            flipped_gray = cv2.flip(gray, 1)
            flipped_faces = profile_cascade.detectMultiScale(flipped_gray, scaleFactor=1.05, minNeighbors=6, minSize=(90, 90))
            for (x, y, w, h) in flipped_faces:
                faces = [[WIDTH - x - w, y, w, h]]

    final_faces = []
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            is_inside_another = False
            for (ox, oy, ow, oh) in faces:
                if w * h < ow * oh:
                    if x >= ox and y >= oy and (x + w) <= (ox + ow) and (y + h) <= (oy + oh):
                        is_inside_another = True
                        break
            if not is_inside_another:
                final_faces.append((x, y, w, h))

    people_count = len(final_faces)

    if people_count > 0:
        final_faces = sorted(final_faces, key=lambda f: f[2] * f[3], reverse=True)
        target_face = final_faces[0]
        lost_frames_count = 0
        
        if last_face is None:
            last_face = target_face
        else:
            old_area = last_face[2] * last_face[3]
            new_area = target_face[2] * target_face[3]
            
            if new_area < (old_area * 0.4) or new_area > (old_area * 2.5):
                last_face = target_face
            else:
                nx = int(last_face[0] + SMOOTHING_FACTOR * (target_face[0] - last_face[0]))
                ny = int(last_face[1] + SMOOTHING_FACTOR * (target_face[1] - last_face[1]))
                nw = int(last_face[2] + SMOOTHING_FACTOR * (target_face[2] - last_face[2]))
                nh = int(last_face[3] + SMOOTHING_FACTOR * (target_face[3] - last_face[3]))
                last_face = (nx, ny, nw, nh)
    else:
        lost_frames_count += 1
        if lost_frames_count > MAX_LOST_FRAMES:
            last_face = None

    # Run background modeling processing and pass face coordinates for masking
    if frames_elapsed < CALIBRATION_FRAMES:
        update_background(region)
        segmented = None
    else:
        segmented = segment_hand(region, last_face)
        if segmented is not None:
            threshold, contour = segmented
            update_hand_data(threshold, contour)

    # --- RENDER UI ELEMENTS ---
    cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (255, 255, 255), 2)
    
    if last_face is not None:
        (fx, fy, fw, fh) = last_face
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), BOX_COLOR, 2)
        cv2.putText(frame, "Face", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 2)

    if frames_elapsed >= CALIBRATION_FRAMES and segmented is not None:
        draw_hand_box(frame, contour)
        draw_debug_traces(frame, contour) 

    draw_hud_dashboard(frame)

    cv2.imshow("Unified Face Tracking & Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        print("Recalibrating background model...")
        reset_calibration()
        continue

    frames_elapsed += 1

proc.terminate()
cv2.destroyAllWindows()
