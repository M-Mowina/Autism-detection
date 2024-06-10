from flask import Flask, render_template, request, redirect, session, flash
from flask_session import Session
import tensorflow as tf
from PIL import Image
import numpy as np
import sqlite3
import bcrypt

# Database connection details
DATABASE_FILE = 'autism.db'

# Load the image classification model
model = tf.keras.models.load_model('autism_v1.h5')

app = Flask(__name__)

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Hash the password before storing
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Connect to the database
        conn = sqlite3.connect(DATABASE_FILE)
        cur = conn.cursor()

        # Check for existing username or email
        try:
            cur.execute("SELECT * FROM Users WHERE username = ? OR email = ?", (username, email))
            existing_user = cur.fetchone()
            if existing_user:
                # Handle existing username or email error (e.g., flash message)
                error = "Username or email already exists."
                flash(error, 'danger')
                print(error)
                return render_template('register.html', error=error)
                
        except sqlite3.Error as e:
            # Handle database errors gracefully (e.g., logging, flash message)
            error = str(e)
            flash(error, 'danger')
            return render_template('register.html', error=f"Database error: {error}")

        # Insert new user into database
        try:
            cur.execute("INSERT INTO Users (username, password, email) VALUES (?, ?, ?)", (username, hashed_password, email))
            conn.commit()
            return redirect('/login')  # Redirect to login page after successful registration
        except sqlite3.Error as e:
            # Handle database errors gracefully (e.g., logging, flash message)
            error = str(e)
            flash(error, 'danger')
            return render_template('register.html', error=f"Database error: {error}")
        finally:
            cur.close()
            conn.close()

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Connect to the database
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cur = conn.cursor()

            # Check for user existence
            cur.execute("SELECT * FROM Users WHERE username = ?", (username,))
            user = cur.fetchone()
            print('password: ',password)
            print('users pass:',user[3])
            if not user:
                # Handle invalid username
                error = "Invalid username or password."
                flash(error, 'danger')
                return render_template('login.html', error=error)

            # Verify password using bcrypt (no encoding needed for hashed_password)
            if not bcrypt.checkpw(password.encode('utf-8'), user[3]):  # Assuming password column index is 2
                # Handle invalid password
                error = "Invalid username or password."
                flash(error, 'danger')
                return render_template('login.html', error=error)

            # Login successful - Store user ID and name in session
            session['user_id'] = user[0]
            session['user_name'] = user[1]

            print('Session:', session)
            print('User:', user)

            return redirect("/")
        except sqlite3.Error as e:
            # Handle database error gracefully
            error = str(e)
            flash(f"Database error: {error}", 'danger')
            return render_template('login.html', error=error)
        finally:
            # Close the database connection (if opened)
            if conn:
                conn.close()

    return render_template('login.html')


@app.route("/logout")
def logout():
    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route('/image_detection', methods=['GET', 'POST'])
def image_detection():
    if request.method == 'POST':
        image_file = request.files["image"]
        img = Image.open(image_file)  # Ensure proper conversion based on model input

        # Preprocess image based on your model requirements (e.g., resizing, normalization)
        #preprocessed_img = convert(img)

        # Predict lapel using your model
        result = make_prediction(img)
        #predicted_class = np.argmax(label)
        #result = predicted_class

        print(result)
        return render_template('image_detection.html', result=result)
    else:
        return render_template('image_detection.html')


@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if request.method == 'POST':
        # Handle questionnaire submission and analysis here
        # Process user responses and return results
        pass
    return render_template('questionnaire.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


def convert(img):
    # Resize image to match model input size
    img = img.resize((224, 224))
    # Add a new axis for batch dimension
    return np.array(img)[np.newaxis]


def make_prediction(img):
    #img = Image.open(img_path)
    converted_img = convert(img)
    prediction = model.predict(converted_img)
    if prediction[0][0] > prediction[0][1]:
        return 'Autism'
    else:
        return 'Non-Autism'
    

if __name__ == '__main__':
    app.run()
