from flask import Flask, url_for
from flask_socketio import *
import pandas as pd

# from Models.models import db
from Controllers.video_stream_controllers import video_stream_controller


app = Flask(__name__)

# Create database resource
app.config.from_object("config")

socketio = SocketIO(app, logger=True)

# db.init_app(app)
# with app.app_context():
#   db.create_all()

# Register blueprint routes
app.register_blueprint(video_stream_controller, url_prefix="/")


# Receive a message from the front end HTML
@socketio.on("request_attendance_list")
def request_attendance_list():
    try:
        df = pd.read_csv(f"Attendance.csv", dtype=str, encoding="utf-8")
        emit(
            "return_attendance_list",
            {"data": df.to_json(orient="records")},
        )
    except Exception as e:
        emit("return_attendance_list", {"data": []})


if __name__ == "__main__":
    socketio.run(app, port=3000, debug=True)
    # app.run(port=3000, debug=True)
