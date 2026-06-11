"""
webcam_face_detect.py

Uses the laptop's built-in webcam to detect AND recognize faces in real time.

Detection is performed with OpenCV's pre-trained Haar cascade (fast, runs on
every frame). For each detected face, a 128-d facial embedding is computed
with the `face_recognition` library (built on dlib) and compared against
embeddings of known faces loaded from the `known_faces/` directory.

    Known face  -> green box, label is the person's name
    Unknown     -> red box,   label is "Unknown"

Requirements:
    pip install opencv-python face_recognition numpy

    `face_recognition` requires dlib, which in turn needs CMake and a C++
    compiler. On Windows, the easiest install is:
        pip install cmake
        pip install dlib
        pip install face_recognition

    On Python 3.13 you also need to pin setuptools<81 so face_recognition's
    legacy `pkg_resources` import works:
        pip install --force-reinstall "setuptools<81"

Enrolling a new face:
    1. Create a folder named `known_faces` in the same directory as this
       script (one already exists in this repo).
    2. Drop in a photo of the person. The filename (without extension) is
       used as their name.
           known_faces/alice.jpg   -> recognized as "alice"
           known_faces/bob.png     -> recognized as "bob"
    3. Run this script again.

       NOTE: very large images (e.g. 4000x3000 phone photos) are
       automatically resized to a manageable size before dlib looks for
       faces. 1024 px on the long side keeps plenty of detail for the
       128-d encoder while making detection fast and reliable.

Run:
    python webcam_face_detect.py
    (or `py webcam_face_detect.py` on Windows)

Press 'q' to quit.
"""

import os
import sys

import cv2
import face_recognition
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Folder that contains one image per known person. The file name (without
# extension) is used as the label that gets drawn on the frame.
KNOWN_FACES_DIR = "known_faces"

# How strict the match is. face_recognition compares the Euclidean distance
# between two 128-d embeddings: smaller distance == more similar. The defaultq
# tolerance of 0.6 is what the dlib examples use; lower it (e.g. 0.45) to be
# more strict, raise it (e.g. 0.55) to be more permissive.
MATCH_TOLERANCE = 0.5

# Haar cascade detector parameters (used for the live webcam).
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (30, 30)

# dlib HOG detector parameters used when loading known faces.
#   * We resize every enrolled image so its longest side is at most
#     ENROLL_MAX_DIM pixels. 12 MP photos from a phone are 4000x3000 and
#     make dlib's HOG detector painfully slow / unreliable. 1024 keeps
#     plenty of detail for the 128-d encoder.
#   * number_of_times_to_upsample=1 lets dlib find smaller / farther faces
#     in the resized image. Default is 0, which often misses faces in busy
#     photos.
ENROLL_MAX_DIM = 1024
ENROLL_UPSAMPLE = 1

# dlib HOG detector parameters used for the live webcam.
LIVE_UPSAMPLE = 1


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resize_long_side(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Return `image` rescaled so its longest side is at most `max_dim`."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    # INTER_AREA is the right choice for shrinking images.
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Loading known faces
# ---------------------------------------------------------------------------

def _find_faces_in_enrollment(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Run dlib's HOG face detector on `rgb` with a couple of fallbacks.

    Returns the list of `(top, right, bottom, left)` locations, or `[]` if
    no face is found. We try HOG with the configured upsample first, then
    with a more aggressive upsample, then as a last resort ask
    `face_recognition.face_encodings` (which itself runs HOG) to also
    detect by passing an empty location list.
    """
    # First attempt: configured upsample.
    locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=ENROLL_UPSAMPLE, model="hog")
    if locs:
        return locs
    # Fallback: more aggressive upsample (helps on small/far faces).
    locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=ENROLL_UPSAMPLE + 1, model="hog")
    if locs:
        return locs
    # Final fallback: even more aggressive upsample.
    return face_recognition.face_locations(rgb, number_of_times_to_upsample=ENROLL_UPSAMPLE + 2, model="hog")


def load_known_faces(directory: str):
    """Load facial embeddings for every image in `directory`.

    Returns
    -------
    known_encodings : list[np.ndarray]
        128-d embeddings, one per successfully loaded face.
    known_names : list[str]
        The corresponding person names (file name without extension).
    """
    known_encodings: list[np.ndarray] = []
    known_names: list[str] = []

    if not os.path.isdir(directory):
        print(f"[warn] '{directory}' directory not found. "
              "Starting with an empty database -- every face will be 'Unknown'.")
        return known_encodings, known_names

    print(f"Loading known faces from '{directory}' ...")
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)

        # Skip non-files (sub-folders, .DS_Store, etc.).
        if not os.path.isfile(path):
            continue

        # Only try common image extensions.
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue

        try:
            # Use OpenCV (libjpeg) to decode -- more reliable than PIL on
            # some Windows machines and won't choke on large images.
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"  [skip] {filename}: could not decode (unsupported / corrupt).")
                continue

            orig_h, orig_w = bgr.shape[:2]
            # Resize large images so dlib's HOG detector runs in a sane
            # amount of time and doesn't hang on noisy backgrounds.
            bgr = _resize_long_side(bgr, ENROLL_MAX_DIM)
            new_h, new_w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            locs = _find_faces_in_enrollment(rgb)
            if not locs:
                print(f"  [skip] {filename}: no face found "
                      f"(orig {orig_w}x{orig_h}, processed {new_w}x{new_h}).")
                continue

            # Encode the first face we found. If multiple faces are in
            # the photo we only keep the largest one -- enroll each
            # person in a separate image to be safe.
            locs_sorted = sorted(
                locs,
                key=lambda b: (b[2] - b[0]) * (b[1] - b[3]),
                reverse=True,
            )
            encodings = face_recognition.face_encodings(rgb, known_face_locations=[locs_sorted[0]])
            if not encodings:
                print(f"  [skip] {filename}: face detected but encoding failed.")
                continue

            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])
            print(f"  [ok]   loaded '{filename}' as '{known_names[-1]}' "
                  f"({orig_w}x{orig_h} -> {new_w}x{new_h}, "
                  f"{len(locs)} face(s) found, kept largest).")
        except Exception as exc:  # noqa: BLE001 - we want to keep going
            print(f"  [skip] {filename}: failed ({type(exc).__name__}: {exc})")
            continue

    print(f"Loaded {len(known_encodings)} known face(s).")
    return known_encodings, known_names


# ---------------------------------------------------------------------------
# Recognition helpers
# ---------------------------------------------------------------------------

def recognize_face(
    face_encoding: np.ndarray,
    known_encodings: list[np.ndarray],
    known_names: list[str],
    tolerance: float = MATCH_TOLERANCE,
) -> tuple[str, float]:
    """Compare `face_encoding` against every known embedding.

    Returns
    -------
    name : str
        The best-matching name, or "Unknown" if no match is within tolerance.
    distance : float
        The Euclidean distance to the best match (lower is better). Returns
        ``np.inf`` if there are no known encodings to compare against.
    """
    if not known_encodings:
        return "Unknown", np.inf

    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])

    if best_distance <= tolerance:
        return known_names[best_idx], best_distance

    return "Unknown", best_distance


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Load Haar cascade -------------------------------------------------
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(
            f"Failed to load Haar cascade from: {cascade_path}\n"
            "Make sure opencv-python is installed correctly."
        )

    # ---- Load known faces --------------------------------------------------
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
    if not known_encodings:
        print("No known faces loaded. The script will still run and label "
              "every detection as 'Unknown' until you add images to "
              f"'{KNOWN_FACES_DIR}/' and restart.")

    # ---- Open the webcam ---------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check that no other application "
            "is using it and that your OS has granted camera permissions."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "Face Recognition - press 'q' to quit"
    print(f"Webcam opened. {window_name}")

    # Counters for the on-screen overlay.
    known_count = 0
    unknown_count = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab a frame from the webcam.")
            break

        total_frames += 1

        # Haar cascades need grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Detection ----------------------------------------------------
        haar_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE,
        )

        # The face_recognition library expects RGB (dlib was trained on RGB).
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Build a list of (top, right, bottom, left) tuples for every Haar
        # detection. Haar gives us (x, y, w, h); we convert to the
        # (top, right, bottom, left) format that face_recognition wants.
        face_locations = []
        for (x, y, w, h) in haar_faces:
            face_locations.append((y, x + w, y + h, x))

        # ---- Recognition --------------------------------------------------
        # If Haar didn't find anything this frame, fall back to dlib's
        # HOG detector (with upsample) before giving up. This makes
        # recognition more robust to lighting / angle changes that Haar
        # sometimes misses.
        if not face_locations:
            face_locations = face_recognition.face_locations(
                rgb,
                number_of_times_to_upsample=LIVE_UPSAMPLE,
                model="hog",
            )

        # Compute embeddings for every detected face. We pass our
        # locations in to skip face_recognition's own (slower) detector
        # and only run the encoder.
        face_encodings = face_recognition.face_encodings(
            rgb,
            known_face_locations=face_locations,
        )

        # Reset per-frame counters.
        known_count = 0
        unknown_count = 0

        # ---- Draw results -------------------------------------------------
        # Iterate over detections and encodings together. The lists have the
        # same length because face_encodings was built from face_locations.
        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
            name, distance = recognize_face(
                face_encoding, known_encodings, known_names
            )

            if name == "Unknown":
                box_color = (0, 0, 255)   # red
                unknown_count += 1
            else:
                box_color = (0, 255, 0)   # green
                known_count += 1

            # Bounding box.
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                color=box_color,
                thickness=2,
            )

            # Label background -- improves readability over varied scenes.
            label = name if name == "Unknown" else f"{name} ({distance:.2f})"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (left, top - text_h - baseline - 4),
                (left + text_w, top),
                box_color,
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                label,
                (left, top - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # white text on colored background
                2,
            )

        # ---- HUD overlay --------------------------------------------------
        # Top line: tallies.
        info = f"Known: {known_count}   Unknown: {unknown_count}   Total: {known_count + unknown_count}"
        cv2.putText(
            frame,
            info,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        # Bottom line: instructions / database size.
        footer = f"DB: {len(known_encodings)} known face(s)  |  tolerance={MATCH_TOLERANCE}  |  press 'q' to quit"
        cv2.putText(
            frame,
            footer,
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Released webcam and closed windows. Processed {total_frames} frame(s).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Allow Ctrl+C to exit cleanly.
        sys.exit(0)
