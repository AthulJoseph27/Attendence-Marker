import numpy as np
import cv2


def extractFace(detector, frame: np.ndarray) -> np.ndarray:
    faces = detector.detect_faces(frame)

    if len(faces) == 0:
        return []

    if faces[0]["confidence"] < 0.9:
        return []

    (x, y, w, h) = faces[0]["box"]
    try:
        a = max(w, h)
        dx = (a - w) // 2
        dy = (a - h) // 2
        croppedFace = np.array(frame[(y - dy) : (y - dy) + a, (x - dx) : (x - dx) + a])
    except Exception as _:
        croppedFace = np.array(frame[y : y + h, x : x + w])

    croppedFace = cv2.resize(croppedFace, (224, 224))
    return croppedFace


def extractAllFaceFromFrame(
    detector, frame: np.ndarray
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    faces = detector.detect_faces(frame)

    if len(faces) == 0:
        return ([], faces)

    croppedFaces = []

    for i, face in enumerate(faces):
        if face["confidence"] < 0.9:
            continue

        (x, y, w, h) = face["box"]
        try:
            a = max(w, h)
            dx = (a - w) // 2
            dy = (a - h) // 2
            croppedFace = np.array(
                frame[(y - dy) : (y - dy) + a, (x - dx) : (x - dx) + a]
            )
            faces[i] = (x - dx, y - dy, a, a)
        except Exception as _:
            croppedFace = np.array(frame[y : y + h, x : x + w])

        croppedFace = cv2.resize(croppedFace, (224, 224))
        croppedFaces.append(croppedFace)

    return (croppedFaces, faces)
