import os

SECRET_KEY = os.urandom(32)

# Grabs the folder where the script runs.
basedir = os.path.abspath(os.path.dirname(__file__))

# Enable debug mode.
DEBUG = True

# Connect to the database
SQLALCHEMY_DATABASE_URI = (
    "mysql+mysqlconnector://root:123456@localhost:3306/recog_system"
)

# Turn off the Flask-SQLAlchemy event system and warning
SQLALCHEMY_TRACK_MODIFICATIONS = False

TEMPLATES_AUTO_RELOAD = True
