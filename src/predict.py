import tensorflow as tf
from keras.models import load_model
import numpy as np
import pandas as pd
import cv2
import time
import datetime
import argparse
import threading
import csv
import os
from mtcnn import MTCNN
from helperFunctions import extractAllFaceFromFrame

STUDENTS_CSV_FILE = "fvc.csv"
 
ATTENDENCE = {}
FACE_RECOGNITION_CONFIDENCE_THRESHOLD = 0.95
FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.95

faceDetector = MTCNN(min_face_size=100)

outputFolder = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "outputs"
)


class Student:
    def __init__(self, name: str, rollNo: str, videoUrl: str):
        self.name = name
        self.rollNo = rollNo
        self.videoUrl = videoUrl

    def __hash__(self) -> int:
        return hash(self.rollNo)

    def __eq__(self, other) -> bool:
        return self.rollNo == other.rollNo

    def __repr__(self) -> str:
        return f"name: {self.name}, rollNo: {self.rollNo}, videoUrl: {self.videoUrl}"

    def __str__(self) -> str:
        return f"name: {self.name}, rollNo: {self.rollNo}, videoUrl: {self.videoUrl}"


def loadDB() -> list[Student]:
    filename = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        "datasets",
        STUDENTS_CSV_FILE,
    )
    with open(filename, "r") as file:
        csvFile = csv.reader(file)

        rows = []
        for line in csvFile:
            rows.append(line)

        nameIndex = rows[0].index("Name")
        rollNoIndex = rows[0].index("Roll Number")

        rows.pop(0)

        students = []
        for row in rows:
            students.append(
                Student(
                    name=row[nameIndex],
                    rollNo=row[rollNoIndex],
                    videoUrl="",
                )
            )

        students.sort(key=lambda x: x.name)
        return students


def predict(
    faceRecognizer,
    faces: np.ndarray,
    studentId: int,
    faceIdentified: list[list[int]],
    faceConfidence: list[list[float]],
) -> None:
    results = faceRecognizer(faces, training=False)  # 0: Known Face, 1: Unknown
    confidences = [max(r) for r in results]
    results = [np.argmax(r) for r in results]

    for j in range(len(faces)):
        if results[j] == 0:
            if confidences[j] < FACE_RECOGNITION_CONFIDENCE_THRESHOLD:
                continue

            faceIdentified[studentId][j] = studentId
            faceConfidence[studentId][j] = confidences[j]


def markAttendenceFromFrame(
    frame: np.ndarray, students: list[Student], faceRecognizers: list
) -> np.ndarray:
    faces, facesCoords = extractAllFaceFromFrame(detector=faceDetector, frame=frame)
    faces = np.array(faces)
    markedFrame = frame.copy()
    color = (0, 255, 0)

    if len(faces) == 0:
        return markedFrame

    predictions: list[list[int]] = [
        [None for _ in range(len(faces))] for _ in range(len(faceRecognizers))
    ]
    confidences: list[list[float]] = [
        [1 for _ in range(len(faces))] for _ in range(len(faceRecognizers))
    ]

    threads = []

    for i, faceRecognizer in enumerate(faceRecognizers):
        if faceRecognizer == None:
            continue

        t = threading.Thread(
            target=predict, args=(faceRecognizer, faces, i, predictions, confidences)
        )
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    faceIdentified: list[int] = [None for _ in range(len(faces))]
    faceConfidence: list[float] = [1 for _ in range(len(faces))]

    for i in range(len(faces)):
        l = []
        c = []
        for j in range(len(predictions)):
            if predictions[j][i] == None:
                continue
            l.append(predictions[j][i])
            c.append(confidences[j][i])

        l = list(set(l))
        if len(l) == 1:
            faceIdentified[i] = l[0]
            faceConfidence[i] = max(c)

    for i, (studentIndex, confidence, (x, y, w, h)) in enumerate(
        zip(faceIdentified, faceConfidence, facesCoords)
    ):
        if studentIndex == None or studentIndex == -1:
            cv2.rectangle(markedFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            continue
        student = students[studentIndex]
        ATTENDENCE[student.rollNo] = True
        cv2.rectangle(markedFrame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            markedFrame,
            f"{student.name}_{student.rollNo}_{int(confidence * 100)}%",
            (x + w, y + h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    return markedFrame


def markAttendence(
    videofile: str, students: list[Student], faceRecognizers, preview=False
) -> None:
    cap = cv2.VideoCapture(videofile)
    prevFrameTime = 0
    curFrameTime = 0

    output = None

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            if output == None and (not preview):
                size = frame.shape[:2][::-1]
                output = cv2.VideoWriter(
                    os.path.join(outputFolder, f"{int(time.time())}.mp4"),
                    cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                    10,
                    size,
                )
            try:
                processedFrame = markAttendenceFromFrame(
                    frame=frame, students=students, faceRecognizers=faceRecognizers
                )
                curFrameTime = time.time()
                fps = round(1 / (curFrameTime - prevFrameTime), 2)
                prevFrameTime = curFrameTime

                cv2.putText(
                    processedFrame,
                    str(fps),
                    (7, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (100, 255, 0),
                    3,
                    cv2.LINE_AA,
                )
                if preview:
                    cv2.imshow("Output", processedFrame)
            except Exception as e:
                print(e)

            if preview:
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
            else:
                output.write(processedFrame)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    if output != None:
        output.release()


def exportAttendenceAsExcel(students: list[Student]) -> None:
    for student in students:
        if not (student.rollNo in ATTENDENCE):
            ATTENDENCE[student.rollNo] = False

    data = {"Roll No": list(ATTENDENCE.keys()), "PRESENT": list(ATTENDENCE.values())}
    df = pd.DataFrame(data=data)
    df.to_excel(
        os.path.join(outputFolder, f'{datetime.date.today().strftime("%d-%m-%Y")}.xlsx')
    )


def loadModels(students: list[Student]) -> list:
    baseFolder = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "models"
    )

    models = []
    for student in students:
        try:
            models.append(
                load_model(
                    os.path.join(
                        baseFolder, f"{student.name}_{student.rollNo}_model.h5"
                    )
                )
            )
        except Exception as _:
            models.append(None)

    return models


# def blockPrint():
#     sys.stdout = open(os.devnull, 'w')


def main(videoFilename="test.mp4", preview=False):
    students = loadDB()
    models = loadModels(students=students)
    os.makedirs(outputFolder, exist_ok=True)
    markAttendence(
        videofile=videoFilename,
        students=students,
        faceRecognizers=models,
        preview=preview,
    )
    exportAttendenceAsExcel(students=students)


parser = argparse.ArgumentParser(description="Video file name")
parser.add_argument("--file", metavar="file", type=str, help="path to the test video")
parser.add_argument(
    "--preview",
    metavar="preview",
    type=int,
    help="live video processing, by default the processed video would be saved.",
)

args = parser.parse_args()
main(videoFilename=args.file, preview=(args.preview == 1))
