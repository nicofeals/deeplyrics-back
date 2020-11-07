import json
import os
from flask import Flask,jsonify,request
from flask_cors import CORS
from deeplyrics import deeplyrics

app = Flask(__name__)
CORS(app)
@app.route("/lyrics/",methods=['POST'])
def return_price():
#   input = request.args.get('input')
    res = request.get_json()
    print(res)
    input = res['input']
    print(input)
    dl = deeplyrics()
    lyrics = dl.predict(input)
    response = {
        'model':'rnn',
        'lyrics': lyrics,
    }
    return jsonify(response)

@app.route("/",methods=['GET'])
def default():
    return "<h1> Welcome to the lyric generator!<h1>"

if __name__ == "__main__":
    app.run()