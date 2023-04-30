# import the necessary modules
import requests
import json
from flask import Flask, request

# create a Flask app
app = Flask(__name__)

# define a route for face comparison
@app.route('/face_comparison', methods=['POST'])
def face_comparison():
    # get the API key and secret from the request parameters
    api_key = request.form['api_key']
    api_secret = request.form['api_secret']
    
    # define the API URL
    api_url = 'https://api-us.faceplusplus.com/facepp/v3/compare'
    
    # create a dictionary with the API key and secret
    params = {'api_key': api_key, 'api_secret': api_secret}
    
    # get the uploaded files from the request object
    file1 = request.files['C:/Users/Rujisaran/Desktop/Billabong Project/pythonlogin/static/HaarProfile/Dechathon.jpg']
    file2 = request.files['C:/Users/Rujisaran/Desktop/Billabong Project/pythonlogin/static/HaarProfile/test.jpg']
    
    # create a dictionary with the files
    files = {'image_file1': (file1.filename, file1.read()), 'image_file2': (file2.filename, file2.read())}
    
    # send the POST request to the API
    r = requests.post(api_url, params=params, files=files)
    
    # get the response from the API
    response = json.loads(r.text)
    
    # get the confidence and thresholds from the response
    confidence = response['confidence']
    thresholds = response['thresholds']
    
    # return the confidence and thresholds as a string
    return f'Confidence: {confidence}, Thresholds: {thresholds}'

# run the app
if __name__ == '__main__':
    app.run(debug=True)
