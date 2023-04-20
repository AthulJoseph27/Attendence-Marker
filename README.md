# Attendence Marker

## Algorithm

- Uses Haar Cascade to detect faces from frame.
- Using face_recognition library, compares the cropped faces to faces saved in the database and gets the distance between each faces in the database. If it matches with more than one student's face, the prediction is ignored. Else the best match of that students face is taken (database contains more than one image of a single student) and if the distance parameter is less than 0.5, then the face will be marked as recognized and the student's attendence will be marked.

## How to run this

- Run `pip install -m requirement.txt` to install the dependencies
- Place a csv file named `fvc.csv` with rows : Name, Roll Number and Video Link. Under the Video Link column, paste the path to the respective student's video.
- Name the video to be tested as `test.mp4`
- Run `python main.py`