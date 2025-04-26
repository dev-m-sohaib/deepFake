import secrets

class Config:
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:@localhost/fyp_database'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = secrets.token_hex(32) 