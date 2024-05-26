from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Class(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey("student.id"), nullable=False)

    def __init__(self, name):
        self.name = name


class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(50), unique=True, nullable=False)
    classs = db.relationship("Class", backref="student", lazy=True)
    phone = db.Column(db.String(100), unique=True, nullable=False)
    address = db.Column(db.String(100), unique=True, nullable=False)
    birthday = db.Column(db.DateTime(), unique=True, nullable=False)
    faculty = db.relationship("Faculty", backref="student", lazy=True)

    def __init__(self, id, fullname, classs, phone, address, birthday):
        self.id = id
        self.fullname = fullname
        self.classs = classs
        self.phone = phone
        self.address = address
        self.birthday = birthday


class Faculty(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey("student.id"), nullable=False)

    def __init__(self, name):
        self.name = name
