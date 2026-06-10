import sys
import cv2
import numpy as np
import subprocess

# Frame configuration
WIDTH = 640
HEIGHT = 480
FRAME_SIZE = WIDTH * HEIGHT * 3

# Load BOTH frontal and profile (tilted/side) face detectors
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

while True:
    raw_frame = proc.stdout.read(FRAME_SIZE)
    if len(raw_frame) != FRAME_SIZE:
        print("End of stream or connection lost.")
        break

    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()

    # Convert to grayscale for speed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Equalize histogram to stabilize lighting variations (prevents false positives from harsh shadows)
    gray = cv2.equalizeHist(gray)

    # 1. Detect Frontal Faces
    # Raised minNeighbors to 8 (makes it highly selective, fixing the nose-box issue)
    faces = front_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(40, 40))

    # 2. Fallback: Detect Profile/Tilted Faces if no frontal face is found
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
        # If it still finds nothing, try flipping the frame to find profile faces looking the other way
        if len(faces) == 0:
            flipped_gray = cv2.flip(gray, 1)
            flipped_faces = profile_cascade.detectMultiScale(flipped_gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
            # Convert coordinates back to original non-flipped space
            for (x, y, w, h) in flipped_faces:
                faces = [[WIDTH - x - w, y, w, h]]

    # --- FILTER OUT DOUBLE BOXES (Nose inside Head) ---
    final_faces = []
    for (x, y, w, h) in faces:
        is_inside_another = False
        for (ox, oy, ow, oh) in faces:
            # Check if current box is smaller and completely inside another box
            if w * h < ow * oh:
                if x >= ox and y >= oy and (x + w) <= (ox + ow) and (y + h) <= (oy + oh):
                    is_inside_another = True
                    break
        if not is_inside_another:
            final_faces.append((x, y, w, h))

    # Draw the clean, filtered boxes
    for (x, y, w, h) in final_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Zero-Latency Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

proc.terminate()
cv2.destroyAllWindows()
