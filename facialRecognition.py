import sys
import cv2
import numpy as np
import subprocess

# Frame configuration
WIDTH = 640
HEIGHT = 480
FRAME_SIZE = WIDTH * HEIGHT * 3

# Load detectors
front_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

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
print("Listening for video frames... Press 'q' to quit.")

# --- SMOOTHING & SIZE VARIABLES ---
last_face = None       
lost_frames_count = 0  
MAX_LOST_FRAMES = 8    # Increased slightly to ride out brief profile orientation drops
SMOOTHING_FACTOR = 0.20 # Sweeter, smoother transition factor to minimize jumping

while True:
    raw_frame = proc.stdout.read(FRAME_SIZE)
    if len(raw_frame) != FRAME_SIZE:
        print("End of stream or connection lost.")
        break

    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ACCURACEY TWEAK 1: Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # This is much more advanced than standard equalizeHist. It prevents harsh lighting/shadows 
    # from breaking face templates when your head tilts away from primary light sources.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # --- ACCURACY TWEAK 2: TUNED DETECTION PARAMS ---
    # Dropped scaleFactor to 1.05: Scans the image at much finer increments (5% steps instead of 15%).
    # This drastically improves multi-distance detection, especially when you step far back.
    # Dropped minNeighbors to 4: Makes detection more eager to catch tilts, while our nested filter handles cleaning.
    # Dropped minSize to (80, 80): Allows the system to catch your face when you are further away.
    faces = front_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(80, 80))

    # Fallback to Profiles (Side-views & Tilts)
    if len(faces) == 0:
        # Profile cascades are looser, tuned similarly to pick up angles quickly
        faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(80, 80))
        if len(faces) == 0:
            flipped_gray = cv2.flip(gray, 1)
            flipped_faces = profile_cascade.detectMultiScale(flipped_gray, scaleFactor=1.05, minNeighbors=4, minSize=(80, 80))
            for (x, y, w, h) in flipped_faces:
                faces = [[WIDTH - x - w, y, w, h]]

    # Filter nested boxes (Ensures features like eyes/nose don't steal the box from the whole head)
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

    # --- SMOOTHING & RESIZE TRACKING ---
    if len(final_faces) > 0:
        # Sort faces by size so the closest/largest face is always prioritized
        final_faces = sorted(final_faces, key=lambda f: f[2] * f[3], reverse=True)
        target_face = final_faces[0]
        lost_frames_count = 0
        
        if last_face is None:
            last_face = target_face
        else:
            old_area = last_face[2] * last_face[3]
            new_area = target_face[2] * target_face[3]
            
            # Adjusted tolerance for size jumps to allow smoother expansion/contraction
            if new_area < (old_area * 0.3) or new_area > (old_area * 3.0):
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

    # Draw the final head-sized box
    if last_face is not None:
        (x, y, w, h) = last_face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Zero-Latency Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

proc.terminate()
cv2.destroyAllWindows()
