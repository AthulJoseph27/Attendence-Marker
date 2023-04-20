import cv2
import csv
import os
import datetime
import time
import face_recognition
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

ATTENDENCE = {} # rollNo: Studnet , if not present: ATTENDENCE[rollNo] -> None

MAX_FACE_PER_PERSON = 5

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
        return f'name: {self.name}, rollNo: {self.rollNo}, videoUrl: {self.videoUrl}'
    
    def __str__(self) -> str:
        return f'name: {self.name}, rollNo: {self.rollNo}, videoUrl: {self.videoUrl}'

class CustomImageEncoding:

    def __init__(self, student: Student, encoding: np.ndarray):
        self.student = student
        self.encoding = encoding

class StudentEncoding:

    def __init__(self, customEncoding: list[CustomImageEncoding]):
        self.students = []
        self.encodings = []

        for ce in customEncoding:
            self.students.append(ce.student)
            self.encodings.append(ce.encoding)
        
        self.encodings = np.array(self.encodings)

def loadDB() -> list[Student]:
    with open('fvc.csv', 'r') as file:
        csvFile = csv.reader(file)

        rows = []
        for line in csvFile:
            rows.append(line)
        
        nameIndex = rows[0].index('Name')
        rollNoIndex = rows[0].index('Roll Number')
        videoUrlIndex = rows[0].index('Video Link')

        rows.pop(0)

        students = []
        for row in rows:
            students.append(Student(name=row[nameIndex], rollNo=row[rollNoIndex], videoUrl=f'../Datasets/{row[videoUrlIndex]}'))

        return students

def extractFace(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return []
    
    (x, y, w, h) = faces[0]
    cropped_face = np.array(frame[y: y+h, x: x+w])

    return cropped_face

def extractAllFaceFromFrame(frame: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return ([], faces)
    
    cropped_faces = []

    for (x, y, w, h) in faces:
        cropped_faces.append(np.array(frame[y: y+h, x: x+w]))

    return (cropped_faces, faces)

def saveExtractedFaceFromFrames(student: Student, path: str):
    cap = cv2.VideoCapture(student.videoUrl)

    # frameInterval = 1000 # Extract a frame every second

    frameCount = 0
    savedFaceCount = 0

    while savedFaceCount < MAX_FACE_PER_PERSON:
        extract, frame = cap.read()

        if not extract:
            break

        frameCount += 1

        face = extractFace(frame)

        if len(face) == 0:
            print(f'Warning: failed to extract face from frame {frameCount} of {student.name}')
            continue

        facePath = os.path.join(path, f'{student.name}_{student.rollNo}_{frameCount}.jpg')
        cv2.imwrite(facePath, face)
        savedFaceCount += 1
    
    cap.release()

def saveFaces(students: list[Student]):
    dirPath = 'faces'
    # if not os.path.exists(dirPath):
    #     os.mkdir(dirPath)
    
    for student in students:
        path = os.path.join(dirPath, student.name)
        os.makedirs(path, exist_ok=True)
        saveExtractedFaceFromFrames(student = student, path = path)

def generateDB() -> list[Student]:
    students = loadDB()
    saveFaces(students=students)

    return students

def generateStudnetImageEncodings(students: list[Student]) -> CustomImageEncoding:
    imageEncodings = []

    dirPath = 'faces'

    for student in tqdm(students):
        path = os.path.join(dirPath, f'{student.name}')
        for filename in os.listdir(path):
            if not filename.endswith('.jpg'):
                continue
            
            imagePath = os.path.join(path, filename)
            face = face_recognition.load_image_file(imagePath)
            encoding = face_recognition.face_encodings(face)
            if len(encoding) == 0:
                continue
            encoding = encoding[0]
            imageEncodings.append(CustomImageEncoding(student=student, encoding=encoding))
    
    return imageEncodings

def markAttendenceFromFrame(frame: np.ndarray, studentsEncoding: StudentEncoding) -> np.ndarray:
    faces, facesCoords = extractAllFaceFromFrame(frame=frame)
    markedFrame = frame.copy()
    color = (0, 255, 0)
    
    for face, (x, y, w, h) in zip(faces, facesCoords):
        unknowEncoding = face_recognition.face_encodings(face)
        result = face_recognition.compare_faces(studentsEncoding.encodings, unknowEncoding)
        distances = face_recognition.face_distance(studentsEncoding.encodings, unknowEncoding)

        matches: list[Student] = []

        for i in range(len(result)):
            if result[i]:
                matches.append(studentsEncoding.students[i])
        
        # count = len(matches)
        matches = list(set(matches))

        if len(matches) != 1:
            # if > 1 -> matched with more than one face, ignore this match
            # show appropriate warning
            continue
        
        if min(distances) > 0.8:
            # best match should have a distance < 0.5, else the match is rejected
            continue
        
        student = matches[0]
        ATTENDENCE[student.rollNo] = True
        cv2.rectangle(markedFrame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(markedFrame, f'{student.name}_{student.rollNo}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return markedFrame

def markAttendence(videofile: str, studentsEncoding: StudentEncoding):
    cap = cv2.VideoCapture(videofile)
    prevFrameTime = 0
    curFrameTime = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            try:
                processedFrame = markAttendenceFromFrame(frame=frame, studentsEncoding=studentsEncoding)
                curFrameTime = time.time()
                fps = int(1 / (curFrameTime - prevFrameTime))
                prevFrameTime = curFrameTime

                cv2.putText(processedFrame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
                cv2.imshow('Output', processedFrame)
            except Exception as e:
                print(e)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def exportAttendenceAsExcel(students: list[Student]):
    for student in students:
        if not (student.rollNo in ATTENDENCE):
            ATTENDENCE[student.rollNo] = False
        
    data = {'Roll No': list(ATTENDENCE.keys()), 'PRESENT': list(ATTENDENCE.values())}
    df = pd.DataFrame(data=data)
    df.to_excel(f'{datetime.date.today().strftime("%d-%m-%Y")}.xlsx')

def cleanUp():
    shutil.rmtree('faces')


def main():
    students = generateDB()
    studnetEncodings = StudentEncoding(customEncoding = generateStudnetImageEncodings(students=students))
    markAttendence(videofile='test.mp4', studentsEncoding=studnetEncodings)
    exportAttendenceAsExcel(students=students)
    cleanUp()

main()