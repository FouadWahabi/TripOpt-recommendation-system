import json
from lib import main
from flask import Flask, render_template, request
from bson import json_util

app = Flask(__name__)


@app.route('/activities/recommended', methods=['GET'])
def recommended():
    userID = request.args.get('userId', '')
    if not userID:
        response = app.response_class(
            response=json.dumps({"message": "Please enter valid userId"}),
            status=400,
            mimetype='application/json'
        )
    else:
        print(userID)
        resp = main.recommended(userID)
        if resp:
            response = app.response_class(
                response=json.dumps(resp, default=json_util.default),
                status=200,
                mimetype='application/json'
            )
        else:
            response = app.response_class(
                response=json.dumps({"message": "Please enter valid userId"}),
                status=400,
                mimetype='application/json'
            )
    return response


@app.route('/train', methods=['GET'])
def train_engine():
    main.train_engine()
    response = app.response_class(
        response=json.dumps({"message": "Successfully trained"}),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    app.run(port=5000)
