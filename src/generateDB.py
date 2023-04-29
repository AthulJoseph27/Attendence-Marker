import cv2
import csv
import os
import random
import numpy as np
from tqdm import tqdm
from mtcnn import MTCNN
from student import Student
from helperFunctions import extractFace

STUDENTS_CSV_FILE = "fvc.csv"
MAX_FACE_PER_PERSON = 12000
detector = MTCNN()


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rotMat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rotMat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def translateImage(image, tx, ty):
    translationMatrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    translatedImage = cv2.warpAffine(
        src=image, M=translationMatrix, dsize=image.shape[1::-1]
    )
    return translatedImage


def loadDB(filename="fvc.csv") -> list[Student]:
    filename = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "datasets", filename
    )
    with open(filename, "r") as file:
        csvFile = csv.reader(file)

        rows = []
        for line in csvFile:
            rows.append(line)

        nameIndex = rows[0].index("Name")
        rollNoIndex = rows[0].index("Roll Number")
        videoUrlIndex = rows[0].index("Video Link")

        rows.pop(0)

        students = []
        for row in rows:
            students.append(
                Student(
                    name=row[nameIndex],
                    rollNo=row[rollNoIndex],
                    videoUrl=os.path.join(
                        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
                        "datasets",
                        "videos",
                        f"{row[videoUrlIndex]}",
                    ),
                )
            )

        return students


def saveExtractedFaceFromFrames(student: Student, path: str):
    cap = cv2.VideoCapture(student.videoUrl)

    frameCount = 0
    savedFaceCount = 0

    while savedFaceCount < MAX_FACE_PER_PERSON:
        extract, frame = cap.read()

        if not extract:
            break

        frameCount += 1

        try:
            face = extractFace(detector=detector, frame=frame)
        except Exception as e:
            continue

        if len(face) == 0:
            print(f"Warning: No face detected in frame {frameCount} of {student.name}")
            continue

        try:
            face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            continue

        facePath = os.path.join(path, f"{frameCount}.jpg")
        cv2.imwrite(facePath, face)

        flippedFace = cv2.flip(face, 1)
        facePath = os.path.join(path, f"{frameCount}_f.jpg")
        cv2.imwrite(facePath, flippedFace)

        savedFaceCount += 2

        for angle in range(-10, 10, 1):
            if angle == 0:
                continue
            index = random.choice([0, 1])
            fs = [face, flippedFace]
            translatedFace = translateImage(
                fs[index], random.randint(-20, 20), random.randint(-20, 20)
            )
            rotatedFace = rotateImage(translatedFace, angle)
            facePath = os.path.join(path, f"{frameCount}_r{angle}.jpg")
            cv2.imwrite(facePath, rotatedFace)
            savedFaceCount += 1

    cap.release()


def saveFaces(students: list[Student]):
    parentDir = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "tmp", "training"
    )

    for student in tqdm(students):
        newDir = f"{student.name}_{student.rollNo}"
        path = os.path.join(os.path.curdir, parentDir, newDir)
        os.makedirs(path, exist_ok=True)
        saveExtractedFaceFromFrames(student=student, path=path)


def main() -> list[Student]:
    students = loadDB(filename=STUDENTS_CSV_FILE)
    saveFaces(students=students)

    return students


if __name__ == "__main__":
    main()
