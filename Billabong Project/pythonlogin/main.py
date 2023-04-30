from flask import Flask, render_template, request, redirect, url_for, session, Response
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import cv2
import numpy as np
import os
import pytesseract
import mysql.connector
from werkzeug.utils import secure_filename

app = Flask(__name__)

camera = cv2.VideoCapture(0)

upload_folder = os.path.join('pythonlogin','static', 'uploads')

app.config['UPLOAD'] = upload_folder

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Draw green square around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Encode and yield the frame as an image
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'your secret key'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '12345'
app.config['MYSQL_DB'] = 'pythonlogin'

# Intialize MySQL
mysql = MySQL(app)

# the following will be our login page, which will use both GET and POST requests
@app.route('/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('index2'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('index.html', msg=msg)

# this will be the logout page
@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))

@app.route('/precap')
def index2():
    return render_template('index2.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    filename = request.form['filename']
    success, frame = camera.read()
    if success:
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Save only the first detected face
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_crop = frame[y:y+h, x:x+w]
            cv2.imwrite('pythonlogin/static/HaarProfile/{}.jpg'.format(filename), face_crop)
            # Load the saved image and encode it as a JPEG
            image_path = 'pythonlogin/static/HaarProfile/{}.jpg'.format(filename)
            img = cv2.imread(image_path)
            ret, buffer = cv2.imencode('.jpg', img)
            image = buffer.tobytes()
            # Pass the image data to the HTML template and render it
            return redirect(url_for('home'))
        else:
            return 'No faces detected'
    else:
        return 'Error capturing image'


# this will be the registration page, we need to use both GET and POST requests
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" POST requests exist (user submitted form)
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Read the image into a numpy array
        img = cv2.imread(img_path)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        # Find contour and sort by contour area
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # Find bounding box and extract ROI
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            ROI = img[y:y+h, x:x+w]
            break

        # resize the image
        rsize = cv2.resize(ROI, (800, 480),
                       interpolation = cv2.INTER_LINEAR)
        
        # Path to Tesseract executable
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

        img = rsize
        img2 = rsize
        # Convert image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        id_pic = cv2.getRectSubPix(img,(286,36),(480,70))

        fname_pic = cv2.getRectSubPix(img,(151,32),(434,160))
        # Perform OCR using Tesseract
        id_num = pytesseract.image_to_string(id_pic).replace(" ","")
        fname = pytesseract.image_to_string(fname_pic).replace(" ","")

        profile_pic = cv2.getRectSubPix(img2,(175,195),(683,331))
        filename = 'pythonlogin/static/data/{}.jpg'.format(fname[:-1])
        cv2.imwrite(filename, profile_pic)

        # Create variables for easy access
        username = fname[:-1]
        password = id_num[:-1]
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s)', (username, password,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'

        return render_template('register.html', img=img2)
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)

@app.route('/home')
def home():
    # Check if user is logged in
    if 'loggedin' in session:
        # User is logged in, show them the home page
        username = session['username']
        # Path to the two images to be compared
        img_path_1 = f"static/data/{username}.jpg"
        img_path_2 = f"static/HaarProfile/{username}.jpg"
        
        # Check if both images exist
        if os.path.exists(img_path_1) and os.path.exists(img_path_2):
            # Load the two images
            img_1 = cv2.imread(img_path_1)
            img_2 = cv2.imread(img_path_2)

            # Preprocess the images
            img_1 = cv2.equalizeHist(cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY))
            img_2 = cv2.equalizeHist(cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY))

            # Load face detector and face recognizer models
            face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Set the threshold value to 80
            face_recognizer.setThreshold(80.0)

            # Detect faces in the two images
            faces_1 = face_detector.detectMultiScale(img_1, scaleFactor=1.1, minNeighbors=3)
            faces_2 = face_detector.detectMultiScale(img_2, scaleFactor=1.1, minNeighbors=3)

            # Debug: print number of faces detected in each image
            print(faces_1, faces_2)
            
            # If only one face is detected in each image, compare them
            if len(faces_1) == 1 and len(faces_2) == 1:
                # Extract the face from each image and resize to 160x160
                x1, y1, w1, h1 = faces_1[0]
                x2, y2, w2, h2 = faces_2[0]
                face_1 = cv2.resize(img_1[y1:y1+h1, x1:x1+w1], (160, 160))
                face_2 = cv2.resize(img_2[y2:y2+h2, x2:x2+w2], (160, 160))

                # Train the face recognizer with the first image
                face_recognizer.train([face_1], np.array([1]))

                # Predict the label of the second image
                label, confidence = face_recognizer.predict(face_2)

                # Debug: print predicted label and confidence value
                print(label, confidence)

                # If the predicted label is 1 (same person)
                if label == 1:
                    score = confidence
                else:
                    score = 0

                # Render the home page with the score
                return render_template(
                    'home.html', 
                    username=username,
                    score=score
                )
        
        # If both images do not exist or comparison failed, render the home page without a score
        return render_template('home.html', username=username, score=None)
    
    # User is not logged in, redirect to login page
    return redirect(url_for('login'))

@app.route('/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)