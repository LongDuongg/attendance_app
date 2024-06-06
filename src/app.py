from flask import Flask
from flask_socketio import *

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
@socketio.on("send_message")
def message_received(data):
    print(data["text"])
    emit("message_from_server", {"text": "Message received!"})


if __name__ == "__main__":
    socketio.run(app, port=3000, debug=True)
    # app.run(port=3000, debug=True)
