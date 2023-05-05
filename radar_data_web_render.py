from flask import Flask, render_template
import pymongo


app = Flask(__name__)

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
db_connect = myclient["radar_db"]  # database name
db_collection = db_connect["radar_data"]  # collection name


@app.route("/")
def index():
    data = {"Prediction": "occupied_room"}

    reply_got = [i for i in db_collection.find(data)][-1]
    reply = {'Prediction': reply_got['Prediction'], 'Time': reply_got['Time']}

    return render_template("index.html", results=reply)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
