from flask import Flask
from tensorflow import keras

# from Models.models import db
from Controllers.video_stream_controllers import video_stream_Controller


def create_app():
    app = Flask(__name__)

    # Create database resource
    app.config.from_object("config")

    # db.init_app(app)
    # with app.app_context():
    #   db.create_all()

    # Register blueprint routes
    app.register_blueprint(video_stream_Controller, url_prefix="/")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(port=3000, debug=True)
