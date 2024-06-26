from flask import Flask, render_template, request, redirect, session, flash
from flask_session import Session
import tensorflow as tf
from PIL import Image
import numpy as np
import sqlite3
import bcrypt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import io
import base64

print(tf.__version__)
# Database connection details
DATABASE_FILE = 'autism.db'

# Load the image classification model
model = tf.keras.models.load_model('autism_v1.h5')
# Load the questionnaire model
questionnaire_mode = tf.keras.models.load_model('questionnair_v1.h5')
# Scale the input features
scaler = joblib.load('scaler.joblib')

# Questionnair columns
columns = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
       'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'ethnicity_Asian',
       'ethnicity_Black', 'ethnicity_Hispanic', 'ethnicity_Latino',
       'ethnicity_Middle Eastern', 'ethnicity_Others', 'ethnicity_Pasifika',
       'ethnicity_South Asian', 'ethnicity_Turkish',
       'ethnicity_White-European', 'relation_Health care professional',
       'relation_Parent', 'relation_Relative', 'relation_Self', 'gender_f',
       'gender_m', 'jundice_no', 'jundice_yes', 'used_app_before_no',
       'used_app_before_yes', "age_desc_'4-11 years'"]

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
                #print(error)
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
            #print('password: ',password)
            #print('users pass:',user[3])
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
        img = Image.open(image_file)

        # Predict lapel using your model
        result = make_prediction(img)
        print(result)

        # Convert image to base64 for efficient transfer
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format=img.format)
        encoded_image = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')
                
        return render_template('result.html', result=result, image=encoded_image)
    else:
        return render_template('image_detection.html')


@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
  if request.method == 'POST':
    inputs = {
      'A1_Score': [request.form['A1_Score']],
      'A2_Score': [request.form['A2_Score']],
      'A3_Score': [request.form['A3_Score']],
      'A4_Score': [request.form['A4_Score']],
      'A5_Score': [request.form['A5_Score']],
      'A6_Score': [request.form['A6_Score']],
      'A7_Score': [request.form['A7_Score']],
      'A8_Score': [request.form['A8_Score']],
      'A9_Score': [request.form['A9_Score']],
      'A10_Score': [request.form['A10_Score']],
      'gender': [request.form['gender']],
      'ethnicity': [request.form['ethnicity']],
      'jundice': [request.form['jundice']],
      'used_app_before': [request.form['used_app_before']],  # Assuming hidden input
      'age_desc': ["'4-11 years'"],  # Assuming hidden input
      'relation': [request.form['relation']]
    }
    x_new = pd.DataFrame(inputs)
    print(x_new)
    print()

    # Process user responses and return results
    result = predict(x_new)
    print('Result: ',result)
    return render_template('result.html', result=result)
  else:
    return render_template('questionnaire.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method == 'GET':
        if not session.get('user_id'):
            return redirect('/login')
        
        return render_template("contact.html")
    else:
        user_id = session.get('user_id')
        subject = request.form['subject']
        message = request.form['message']
        #print(message)
        # Connect to the database
        conn = sqlite3.connect(DATABASE_FILE)
        cur = conn.cursor()
        # Insert data into the ContactUs table
        try:
            user_id = session['user_id']
            # Use a parameterized query to prevent SQL injection attacks
            cursor = conn.cursor()
            cursor.execute("""INSERT INTO Feedback (subject, message, user_id)
                        VALUES (?, ?, ?)""", (subject, message, user_id))
            conn.commit()
            respond = 'Your message has been sent successfully!'
            flash(respond, 'success')
            print(respond)
            return render_template('contact.html', error=respond)
        except sqlite3.Error as e:
            # Handle database errors gracefully (e.g., logging, flash message)
            error = str(e)
            flash(f"Database error: {error}", 'danger')
            print(error)
            return render_template('contact.html', error=error)
        finally:
            cur.close()
            conn.close()



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
        return "Autism"
    else:
        return "Non-Autism" 
  
def preprocess_input(input_data, scaler = scaler, columns = columns):
    """
    Preprocesses the input data by applying one-hot encoding and scaling.
    
    Parameters:
    - input_data: pd.DataFrame, new input data
    - scaler: StandardScaler, fitted scaler from the training data
    - columns: list, columns of the training data after one-hot encoding
    
    Returns:
    - pd.DataFrame, preprocessed input data
    """
    # One-hot encode the input data
    input_data = pd.get_dummies(input_data, columns=['ethnicity','relation','gender','jundice','used_app_before','age_desc'])
    
    # Ensure input_data has the same columns as the training data
    missing_cols = set(columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[columns]
    
    # Scale the input data
    input_data = pd.DataFrame(scaler.transform(input_data), columns=columns, index=input_data.index)
    print(input_data)
    return input_data

def predict(input_data, model = questionnaire_mode, scaler = scaler, columns = columns):
    # Preprocess the input data
    input_data_preprocessed = preprocess_input(input_data, scaler, columns)
    
    # Make predictions
    predictions = model.predict(input_data_preprocessed)
    
    if predictions[0][0] > 0.5:
        return "Autism"
    else:
        return "Non-Autism"
    

if __name__ == '__main__':
    app.run()
