import sys
import cv2
import numpy as np
import face_recognition
import subprocess

# Frame configuration (must match the Pi's output resolution)
WIDTH = 640
HEIGHT = 480
FRAME_SIZE = WIDTH * HEIGHT * 3 # 3 bytes per pixel (RGB)

print("Initializing local FFMPEG decoder pipeline...")

# Spin up an internal ffmpeg process on your Mac to parse the incoming stdin stream
command = [
    'ffmpeg',
    '-f', 'h264',               # Input is raw h264 bytes
    '-fflags', 'nobuffer',      # Minimize latency
    '-flags', 'low_delay',
    '-probesize', '32',
    '-analyzeduration', '0',
    '-i', 'pipe:0',             # Read from the main terminal pipe
    '-f', 'rawvideo',           # Output raw, uncompressed video frames
    '-pix_fmt', 'bgr24',        # Direct conversion to OpenCV's native BGR format
    'pipe:1'                    # Push clean frames to python
]

# Open the pipe
proc = subprocess.Popen(command, stdin=sys.stdin.buffer, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

print("Listening for video frames... Press 'q' to quit.")

frame_count = 0
face_locations = []

while True:
    # Read exactly one raw frame worth of bytes from ffmpeg's stdout
    raw_frame = proc.stdout.read(FRAME_SIZE)
    if len(raw_frame) != FRAME_SIZE:
        print("End of stream or connection lost.")
        break

    frame_count += 1

    # Reshape the raw bytes into a standard image array and make a writable COPY
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()

    # --- LATENCY FIX: Only run face recognition every 5th frame ---
    if frame_count % 5 == 0:
        # Convert from BGR to face_recognition's RGB format
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Find all face locations
        face_locations = face_recognition.face_locations(rgb)

    # Draw the boxes using the most recent face coordinates (keeps video smooth)
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the video output window
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

proc.terminate()
cv2.destroyAllWindows()
