from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import re
import security  # assuming your security module remains for password functions
from functools import wraps
import os
import pickle
import numpy as np
np.core._ = None
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from indicnlp.tokenize import indic_tokenize
import logging

app = Flask(__name__, static_folder="build", static_url_path="/")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///twitter.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "some_secret_key"  # Needed for session management
logging.getLogger('werkzeug').setLevel(logging.WARNING)
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')

db = SQLAlchemy(app)
CORS(app, supports_credentials=True)

# --------------------
# Database Models
# --------------------

def transliterate_roman_to_telugu(text):
    """Convert ITRANS Romanized Telugu to Telugu script"""
    if re.search(r'[\u0C00-\u0C7F]', text):  # If Telugu chars already present
        return text
    try:
        return transliterate(text, sanscript.ITRANS, sanscript.TELUGU)
    except Exception as e:
        print(f"Transliteration error: {e}")
        return text  # Fallback to original

def telugu_preprocessor(text):
        text = re.sub(r'[^\u0C00-\u0C7F]', ' ', str(text))  # Remove non-Telugu characters
        tokens = indic_tokenize.trivial_tokenize(text)  # Tokenize using Indic NLP
        return ' '.join(tokens)

class ModelBank:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.tf_models = []
        self.sklearn_models = []
        self.vectorizer = None
        self.tokenizer = None
        self.max_len = 100
        
        # Verify model directory exists
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory '{self.model_dir}' not found")
            
        self.load_assets()

    

    def load_assets(self):
        # Custom unpickler for version compatibility
        class ForcedUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Fix numpy core references
                if module.startswith("numpy._core"):
                    module = module.replace("numpy._core", "numpy.core")
                return super().find_class(module, name)

        # Load preprocessing files
        vectorizer_path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
        tokenizer_path = os.path.join(self.model_dir, "dl_tokenizer.pkl")
        
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = ForcedUnpickler(f).load()
                
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, "rb") as f:
                tokenizer_data = ForcedUnpickler(f).load()
                self.tokenizer = tokenizer_data["tokenizer"]
                self.max_len = tokenizer_data.get("max_len", 100)

        # Load models
        for file in os.listdir(self.model_dir):
            file_path = os.path.join(self.model_dir, file)
            
            if file.endswith(".h5"):
                try:
                    model = load_model(file_path)
                    model_name = os.path.splitext(file)[0]  # Get filename without extension
                    model._name = model_name
                    self.tf_models.append(model)
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
                    
            elif file.endswith(".pkl") and "model" in file:
                try:
                    with open(file_path, "rb") as f:
                        model = ForcedUnpickler(f).load()
                        self.sklearn_models.append(model)
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")

    def preprocess_for_dl(self, text):
        sequences = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=self.max_len, dtype='int32')
        return padded

    # In ModelBank class
    def predict(self, text):
        offensive_votes = 0
        non_offensive_votes = 0
        total_votes = 0
        
        print("\n--- Model Voting Results ---")
        
        # Traditional Models
        if self.vectorizer:
            tfidf_text = self.vectorizer.transform([text])
            for model in self.sklearn_models:
                try:
                    pred = model.predict(tfidf_text)[0]
                    model_name = type(model).__name__
                    vote = "Offensive" if pred == "hate" else "Non-Offensive"
                    offensive_votes += 1 if pred == "hate" else 0
                    non_offensive_votes += 0 if pred == "hate" else 1
                    print(f"| {model_name:25} | {vote:15} |")
                    total_votes += 1
                except Exception as e:
                    print(f"| {model_name:25} | Error: {str(e)[:10]} |")

        # Deep Learning Models
        if self.tokenizer:
            dl_seq = self.preprocess_for_dl(text)
            for model in self.tf_models:
                try:
                    pred = (model.predict(dl_seq, verbose=0) > 0.5).astype(int)[0][0]
                    model_name = model.name
                    vote = "Offensive" if pred == 0 else "Non-Offensive"
                    if pred == 0:
                        offensive_votes += 1
                    else:
                        non_offensive_votes += 1
                    print(f"| {model_name:25} | {vote:15} |")
                    total_votes += 1
                except Exception as e:
                    print(f"| {model_name:25} | Error: {str(e)[:10]} |")

        # Print summary table
        print(f"\nSummary:")
        print(f"Offensive Votes: {offensive_votes}")
        print(f"Non-Offensive Votes: {non_offensive_votes}")
        print(f"Total Models Participated: {total_votes}")
        print("-" * 40 + "\n")
        
        return int(offensive_votes)

# Initialize with single model directory
model_bank = ModelBank(model_dir="models")

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(24))
    email = db.Column(db.String(64))
    pwd = db.Column(db.String(64))

    def __init__(self, username, email, pwd):
        self.username = username
        self.email = email
        self.pwd = pwd

class Tweet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column(db.Integer, db.ForeignKey("user.id"))
    user = db.relationship('User', foreign_keys=uid)
    title = db.Column(db.String(256))
    content = db.Column(db.String(2048))
    is_offensive = db.Column(db.Boolean, default=False)  # Add this line
# --------------------
# Helper Functions
# --------------------
def getUsers():
    users = User.query.all()
    return [{"id": i.id, "username": i.username, "email": i.email, "password": i.pwd} for i in users]

def getUser(uid):
    user = db.session.get(User, uid)
    return {"id": user.id, "username": user.username, "email": user.email, "password": user.pwd}

def addUser(username, email, pwd):
    try:
        user = User(username, email, pwd)
        db.session.add(user)
        db.session.commit()
        return True
    except Exception as e:
        print(e)
        return False

def removeUser(uid):
    try:
        user = User.query.get(uid)
        db.session.delete(user)
        db.session.commit()
        return True
    except Exception as e:
        print(e)
        return False

def getTweets():
    tweets = Tweet.query.all()
    return [{"id": i.id, "title": i.title, "content": i.content, "user": getUser(i.uid), "is_offensive": i.is_offensive} for i in tweets]

def addTweet(title, content, uid, is_offensive=False):  # Modified
    try:
        user = db.session.get(User, uid)
        if not user:
            return False
        twt = Tweet(title=title, content=content, user=user, is_offensive=is_offensive)  # Add is_offensive
        db.session.add(twt)
        db.session.commit()
        return True
    except Exception as e:
        print(e)
        return False

'''
def addTweet(title, content, uid):
    try:
        user = User.query.get(uid)
        if not user:
            return False
        twt = Tweet(title=title, content=content, user=user)
        db.session.add(twt)
        db.session.commit()
        return True
    except Exception as e:
        print(e)
        return False
'''

def delTweet(tid):
    try:
        tweet = db.session.get(Tweet, tid)
        db.session.delete(tweet)
        db.session.commit()
        return True
    except Exception as e:
        print(e)
        return False

# --------------------
# Custom Decorator for Session Check
# --------------------
def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Authentication required"}), 401
        return func(*args, **kwargs)
    return wrapper

# --------------------
# Routes
# --------------------
@app.route("/<a>")
def react_routes(a):
    return app.send_static_file("index.html")

@app.route("/")
def react_index():
    return app.send_static_file("index.html")

@app.route("/api/login", methods=["POST"])
def login():
    try:
        email = request.json["email"]
        password = request.json["pwd"]
        if email and password:
            users = getUsers()
            user = [u for u in users if security.dec(u["email"]) == email and security.checkpwd(password, u["password"])]
            if len(user) == 1:
                session["user_id"] = str(user[0]["id"])  # store as string
                return jsonify({"success": True})
            else:
                return jsonify({"error": "Invalid credentials"})
        else:
            return jsonify({"error": "Invalid form"})
    except Exception as e:
        print(e)
        return jsonify({"error": "Invalid form"})

@app.route("/api/register", methods=["POST"])
def register():
    try:
        email = request.json["email"].lower()
        password = security.encpwd(request.json["pwd"])
        username = request.json["username"]
        if not (email and password and username):
            return jsonify({"error": "Invalid form"})
        users = getUsers()
        if any(security.dec(u["email"]) == email for u in users):
            return jsonify({"error": "User already exists"})
        if not re.match(r"[\w._]{5,}@\w{3,}\.\w{2,4}", email):
            return jsonify({"error": "Invalid email"})
        addUser(username, security.enc(email), password)
        return jsonify({"success": True})
    except Exception as e:
        print(e)
        return jsonify({"error": "Invalid form"})

@app.route("/api/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route("/api/tweets")
def get_tweets():
    return jsonify(getTweets())

@app.route("/api/addtweet", methods=["POST"])
@login_required
def add_tweet():
    try:
        title = request.json.get("title", "").strip()
        raw_content = request.json.get("content", "").strip()
        
        clean_content = re.sub('<[^<]+?>', '', raw_content).strip()
        transliterated_content = transliterate_roman_to_telugu(clean_content)
        if not (title and clean_content):
            return jsonify({"error": "Invalid form"}), 422
        
        offensive_votes = model_bank.predict(transliterated_content)
        total_models = len(model_bank.tf_models) + len(model_bank.sklearn_models)
        is_offensive = offensive_votes >= 5  # Determine if offensive
        
        uid = session.get("user_id")
        if addTweet(title, raw_content, uid, is_offensive):  # Pass is_offensive
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Could not add tweet"}), 400
    except Exception as e:
        print(e)
        return jsonify({"error": "Invalid form"}), 422

'''
@app.route("/api/addtweet", methods=["POST"])
@login_required
def add_tweet():
    try:
        title = request.json.get("title", "").strip()
        raw_content = request.json.get("content", "").strip()
        
        # Remove HTML tags and whitespace
        clean_content = re.sub('<[^<]+?>', '', raw_content).strip()
        transliterated_content = transliterate_roman_to_telugu(clean_content)
        if not (title and clean_content):
            return jsonify({"error": "Invalid form"}), 422
        
        offensive_votes = model_bank.predict(transliterated_content)
        total_models = len(model_bank.tf_models) + len(model_bank.sklearn_models)
        
        if offensive_votes >= 5:
            return jsonify({
                "error": "This content appears to be offensive and cannot be posted",
                "offensive_votes": offensive_votes,
                "total_models": total_models
            }), 400
        
        uid = session.get("user_id")
        if addTweet(title, raw_content, uid):
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Could not add tweet"}), 400
    except Exception as e:
        print(e)
        return jsonify({"error": "Invalid form"}), 422
'''
@app.route("/api/deletetweet/<tid>", methods=["DELETE"])
@login_required
def delete_tweet(tid):
    try:
        if delTweet(tid):
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Could not delete tweet"}), 400
    except Exception as e:
        print(e)
        return jsonify({"error": "Invalid form"}), 422

@app.route("/api/getcurrentuser")
@login_required
def get_current_user():
    uid = session.get("user_id")
    return jsonify(getUser(uid))

@app.route("/api/changepassword", methods=["POST"])
@login_required
def change_password():
    try:
        uid = session.get("user_id")
        user = db.session.get(User, uid)
        if not (request.json["password"] and request.json["npassword"]):
            return jsonify({"error": "Invalid form"}), 422
        if not security.checkpwd(request.json["password"], user.pwd):
            return jsonify({"error": "Wrong password"}), 422
        user.pwd = request.json["npassword"]
        db.session.add(user)
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        print(e)
        return jsonify({"error": "Invalid form"}), 422

@app.route("/api/deleteaccount", methods=["DELETE"])
@login_required
def delete_account():
    try:
        uid = session.get("user_id")
        user = db.session.get(User, uid)
        tweets = Tweet.query.filter(Tweet.uid == uid).all()
        for tweet in tweets:
            delTweet(tweet.id)
        removeUser(user.id)
        session.clear()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 422

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

"""""
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import re
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, create_refresh_token, get_jwt
import security

app = Flask(__name__, static_folder="build", static_url_path="/")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///twitter.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
app.config["JWT_SECRET_KEY"] = "iamsupposedtobeasecret"
app.config["JWT_BLOCKLIST_ENABLED"] = True
app.config["JWT_BLOCKLIST_TOKEN_CHECKS"] = ["access", "refresh"]
jwt = JWTManager(app)
CORS(app)


# DB
class User(db.Model):
    id = db.Column(db.Integer,
                   primary_key=True)
    username = db.Column(db.String(24))
    email = db.Column(db.String(64))
    pwd = db.Column(db.String(64))

    # Constructor
    def __init__(self, username, email, pwd):
        self.username = username
        self.email = email
        self.pwd = pwd


def getUsers():
    users = User.query.all()
    return [{"id": i.id, "username": i.username, "email": i.email, "password": i.pwd} for i in users]


def getUser(uid):
    users = User.query.all()
    user = list(filter(lambda x: x.id == uid, users))[0]
    return {"id": user.id, "username": user.username, "email": user.email, "password": user.pwd}


def addUser(username, email, pwd):
    try:
        user = User(username, email, pwd)
        db.session.add(user)
        db.session.commit()
        return True
    except Exception as e:
        print(e)
        return False


def removeUser(uid):
    try:
        user = User.query.get(uid)
        db.session.delete(user)
        db.session.commit()
        return True
    except Exception as e:
        print(e)
        return False


class Tweet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column(db.Integer, db.ForeignKey("user.id"))
    user = db.relationship('User', foreign_keys=uid)
    title = db.Column(db.String(256))
    content = db.Column(db.String(2048))


def getTweets():
    tweets = Tweet.query.all()
    return [{"id": i.id, "title": i.title, "content": i.content, "user": getUser(i.uid)} for i in tweets]


def getUserTweets(uid):
    tweets = Tweet.query.all()
    return [{"id": item.id, "userid": item.user_id, "title": item.title, "content": item.content} for item in
            filter(lambda i: i.user_id == uid, tweets)]


def addTweet(title, content, uid):
    try:
        user = list(filter(lambda i: i.id == uid, User.query.all()))[0]
        twt = Tweet(title=title, content=content, user=user)
        db.session.add(twt)
        db.session.commit()
        return True
    except Exception as e:
        print(e)
        return False


def delTweet(tid):
    try:
        tweet = Tweet.query.get(tid)
        db.session.delete(tweet)
        db.session.commit()
        return True
    except Exception as e:
        print(e)
        return False


class InvalidToken(db.Model):
    __tablename__ = "invalid_tokens"
    id = db.Column(db.Integer, primary_key=True)
    jti = db.Column(db.String)

    def save(self):
        db.session.add(self)
        db.session.commit()

    @classmethod
    def is_invalid(cls, jti):
        q = cls.query.filter_by(jti=jti).first()
        return bool(q)


@jwt.token_in_blocklist_loader
def check_if_blocklisted_token(decrypted):
    jti = decrypted["jti"]
    return InvalidToken.is_invalid(jti)


# ROUTES
@app.route("/<a>")
def react_routes(a):
    return app.send_static_file("index.html")


@app.route("/")
def react_index():
    return app.send_static_file("index.html")


@app.route("/api/login", methods=["POST"])
def login():
    try:
        email = request.json["email"]
        password = request.json["pwd"]
        if email and password:
            user = list(filter(lambda x: security.dec(x["email"]) == email and security.checkpwd(password, x["password"]), getUsers()))
            # Check if user exists
            if len(user) == 1:
                token = create_access_token(identity=str(user[0]["id"]),fresh = True)
                refresh_token = create_refresh_token(identity=str(user[0]["id"]))
                return jsonify({"token": token, "refreshToken": refresh_token})
            else:
                return jsonify({"error": "Invalid credentials"})
        else:
            return jsonify({"error": "Invalid form"})
    except Exception as e:
        print(e)
        return jsonify({"error": "Invalid form"})


@app.route("/api/register", methods=["POST"])
def register():
    try:
        email = request.json["email"]
        email = email.lower()
        password = security.encpwd(request.json["pwd"])
        username = request.json["username"]
        print(email, password, request.json["pwd"], username)
        if not (email and password and username):
            return jsonify({"error": "Invalid form"})
        # Check to see if user already exists
        users = getUsers()
        if len(list(filter(lambda x: security.dec(x["email"] == email), users))) == 1:
            return jsonify({"error": "Invalid form"})
        # Email validation check
        if not re.match(r"[\w._]{5,}@\w{3,}\.\w{2,4}", email):
            return jsonify({"error": "Invalid email"})
        addUser(username, security.enc(email), password)
        return jsonify({"success": True})
    except Exception as e:
        print(e)
        return jsonify({"error": "Invalid form"})


@app.route("/api/checkiftokenexpire", methods=["POST"])
@jwt_required()
def check_if_token_expire():
    return jsonify({"success": True})


@app.route("/api/refreshtoken", methods=["POST"])
@jwt_required(refresh=True)
def refresh():
    identity = get_jwt_identity()
    token = create_access_token(identity=str(identity))
    return jsonify({"token": token})


@app.route("/api/logout/access", methods=["POST"])
@jwt_required()
def access_logout():
    jti = get_jwt()["jti"]
    try:
        invalid_token = InvalidToken(jti=jti)
        invalid_token.save()
        return jsonify({"success": True})
    except Exception as e:
        print(e)
        return {"error": e.message}


@app.route("/api/logout/refresh", methods=["POST"])
@jwt_required()
def refresh_logout():
    jti = get_jwt()["jti"]
    try:
        invalid_token = InvalidToken(jti=jti)
        invalid_token.save()
        return jsonify({"success": True})
    except Exception as e:
        print(e)
        return {"error": e.message}


@app.route("/api/tweets")
def get_tweets():
    return jsonify(getTweets())


@app.route("/api/addtweet", methods=["POST"])
@jwt_required()
def add_tweet():
    try:
        title = request.json["title"]
        content = request.json["content"]
        if not (title and content):
            return jsonify({"error": "Invalid form"})
        uid = int(get_jwt_identity())
        addTweet(title, content, uid)
        return jsonify({"success": "true"})
    except Exception as e:
        print(e)
        return jsonify({"error": "Invalid form"})


@app.route("/api/deletetweet/<tid>", methods=["DELETE"])
@jwt_required()
def delete_tweet(tid):
    try:
        delTweet(tid)
        return jsonify({"success": "true"})
    except:
        return jsonify({"error": "Invalid form"})


@app.route("/api/getcurrentuser")
@jwt_required()
def get_current_user():
    uid = int(get_jwt_identity())
    return jsonify(getUser(uid))


@app.route("/api/changepassword", methods=["POST"])
@jwt_required()
def change_password():
    try:
        user = User.query.get(int(get_jwt_identity()))
        if not (request.json["password"] and request.json["npassword"]):
            return jsonify({"error": "Invalid form"})
        if not security.checkpwd(request.json["password"], user.pwd):
            return jsonify({"error": "Wrong password"})
        user.pwd = request.json["npassword"]
        db.session.add(user)
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        print(e)
        return jsonify({"error": "Invalid form"})


@app.route("/api/deleteaccount", methods=["DELETE"])
@jwt_required()
def delete_account():
    try:
        user = User.query.get(int(get_jwt_identity()))
        tweets = Tweet.query.all()
        for tweet in tweets:
            if tweet.user.username == user.username:
                delTweet(tweet.id)
        removeUser(user.id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
"""