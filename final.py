import face_recognition
import os
import cv2
import csv
import pyrebase
import imutils
from datetime import datetime

firebaseConfig = {
    "apiKey": "AIzaSyCarNXKXtR4ZhdlTO2n0i26PC_ef1yUWJ4",
    "authDomain": "my-application99-b7f7c.firebaseapp.com",
    "databaseURL": "https://my-application99-b7f7c.firebaseio.com",
    "projectId": "my-application99-b7f7c",
    "storageBucket": "my-application99-b7f7c.appspot.com",
    "messagingSenderId": "690816593286",
    "appId": "1:690816593286:web:0fef6a69a27cb144301007"
  }

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

field_names = ['Name', 'Enter Time', 'Exit Time', 'Elapsed Time', 'Attendance State']

def Attnedance(name,timee):
    if name not in names:
        names.append(name)
        name = name.replace(" ", "_")

        now = datetime.now()
        dtstring = now.strftime('%Y-%m-%d %H:%M:%S')

        time_delta = (now - now)
        total_seconds = time_delta.total_seconds()
        minutes = float(total_seconds / 60)
        timee = timee/60000
        print(minutes)
        print(timee)


        if minutes > timee:
            state = True
        else:
            state = False
        globals()[name] = {'Name': str(name).replace("_", " "), 'Enter Time': dtstring, 'Exit Time': dtstring, 'Elapsed Time': minutes,
            'Attendance State': state}
        students.append(globals()[name])

    elif name in names:
        name = name.replace(" ", "_")
        fmt = '%Y-%m-%d %H:%M:%S'
        d1 = datetime.strptime(globals()[name]['Enter Time'], fmt)

        now = datetime.now()
        dtstring = now.strftime('%Y-%m-%d %H:%M:%S')

        time_delta = (now - d1)
        total_seconds = time_delta.total_seconds()
        minutes = float(total_seconds / 60)
        timee = timee / 60000
        print(minutes)
        print(timee)

        if minutes > timee:
            state = True
        else:
            state = False
        for item in students:
            if item['Name'] == str(name).replace("_", " "):
                my_item = item
                my_item['Exit Time'] = dtstring
                my_item['Elapsed Time'] = minutes
                my_item['Attendance State'] = state
                break
        students.remove(globals()[name])
        students.append(my_item)

KNOWN_FACES_DIR = 'known_faces'

TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False

def markattendance():
    with open('Names.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            if len(row) > 1:
                name = row[0]
                state = row[4]
                state1 = str_to_bool(state)
                test = dict(db.child("now").get().val())
                db.child("Student").child(name).child("Courses").child(test["course"]).child("attendance").update(
                    {test["lec_name"]: state1})

print('Loading known faces...')
known_faces = []
known_names = []

# We oranize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)

print(known_names)
print('Processing unknown faces...')
# Now let's loop over a folder of faces we want to label

while True:
    students = []
    names = []
    test = dict(db.child("now").get().val())
    video = cv2.VideoCapture(0)
    print("waiting lec start")
    while (test["course"] != ""):
        print("lec started")
        # Load image
        ret, frame = video.read()


        # This time we first grab face locations - we'll need them to draw boxes
        locations = face_recognition.face_locations(frame, model=MODEL)

        # Now since we know loctions, we can pass them to face_encodings as second argument
        # Without that it will search for faces once again slowing down whole process
        encodings = face_recognition.face_encodings(frame, locations)

        # We passed our image through face_locations and face_encodings, so we can modify it
        # First we need to convert it from RGB to BGR as we are going to work with cv2

        # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
        print(f', found {len(encodings)} face(s)')

        for face_encoding, face_location in zip(encodings, locations):

            # We use compare_faces (but might use face_distance as well)
            # Returns array of True/False values in order of passed known_faces
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

            # Since order is being preserved, we check if any face was found then grab index
            # then label (name) of first matching known face withing a tolerance
            match = None
            if True in results:  # If at least one is true, get a name of first of found labels
                match = known_names[results.index(True)]
                print(f' - {match} from {results}')

                # Each location contains positions in order: top, right, bottom, left

                if match != '':
                    timee = test["duration"]
                    Attnedance(match,timee)
                    with open('Names.csv', 'w') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=field_names)
                        writer.writerows(students)
                    markattendance()
        test = dict(db.child("now").get().val())

        if test["course"] == "":
            print("lec end")
            video.release()
            break
