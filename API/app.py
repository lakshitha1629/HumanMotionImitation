from flask import Flask, request, redirect, send_file
from flask_restful import Resource, Api
from flask_cors import CORS
import pandas as pd
import os
import prediction

ALLOWED_EXTENSIONS = ['csv','png', 'jpg', 'jpeg']
UPLOAD_FOLDER = './upload'

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api = Api(app)

def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class Test(Resource):
    def get(self):
        return 'Welcome to, Human Motion Imitation API!'
    
    def post(self):
        try:
            value = request.get_json()
            if(value):
                return {'Post Values': value}, 201
            
            return {"error":"Invalid format."}
            
        except Exception as error:
            return {'error': error}

class GetOutput(Resource):
    def get(self):
        return {"error":"Invalid Method."}


    def post(self):
        if(os.path.exists("model.pkl")):
            os.remove("model.pkl")

        projectType = request.files.get("projectType")
        frontImg = request.files.get("frontImg")
        backImg = request.files.get("backImg")
        backgroundImg = request.files.get("backgroundImg")
        videoLink = request.files.get("videoLink")
        videoFile = request.files.get("videoFile")
        startTime = request.files.get("startTime")
        endTime = request.files.get("endTime")
        poseFactor = request.files.get("poseFactor")
        camFactor = request.files.get("camFactor")
            
        file_to_upload = request.files.get("file")
        predictValue = request.form.get("predictValue")

        if frontImg.filename  == '' or backImg.filename  == '' or backgroundImg.filename  == '' or videoFile.filename  == '':
            print('No selected file')
            return redirect(request.url)

        if not allowed_file(frontImg.filename):
            return {"error":"Invalid frontImg file format."}

        if not allowed_file(backImg.filename):
            return {"error":"Invalid backImg file format."}

        if not allowed_file(backgroundImg.filename):
            return {"error":"Invalid backgroundImg file format."}

        if not allowed_file(videoFile.filename):
            return {"error":"Invalid videoFile file format."}

        try:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file_to_upload.filename)
            file_to_upload.save(path)
            predictCol = str(predictValue)
            df = pd.read_csv(path)
            modelVal, x_test, y_test = prediction.fitModel(df, predictCol)
            model, score = prediction.createModel(modelVal, x_test, y_test)
            testScore = "{0:.2f} %".format(100 * score)
            return { 'modelScore': testScore }
            # return send_file('model.pkl', mimetype = 'pkl', attachment_filename= 'model.pkl', as_attachment = True) 

        except Exception as error:
            return {'error': error}
        

api.add_resource(Test,'/')
api.add_resource(GetOutput,'/GetOutput')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

