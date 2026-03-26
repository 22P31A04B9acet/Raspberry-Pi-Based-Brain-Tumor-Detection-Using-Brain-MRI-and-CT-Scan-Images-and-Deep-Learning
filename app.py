import os
import cv2
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
DATABASE = 'users.db'

# ---------------------- Load YOLO Model ONCE ---------------------- #
MODEL_PATH = "runs/detect/train3/weights/best.pt"
model = YOLO(MODEL_PATH)

# ---------------------- Database Setup ---------------------- #
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    if not os.path.exists(DATABASE):
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

init_db()

# ---------------------- Federated Learning (Simulated Placeholder) ---------------------- #
def federated_average(dummy_models):
    # Placeholder for your project concept (does not affect prediction)
    print(f"Simulating FedAvg with {len(dummy_models)} clients.")
    return model


# ---------------------- Routes ---------------------- #
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')

        if not username or not email or not password or not confirm:
            flash('Please fill out all fields.', 'error')
            return redirect(url_for('signup'))
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password)

        conn = get_db()
        try:
            conn.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, hashed_password)
            )
            conn.commit()
            flash('Signup successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.', 'error')
            return redirect(url_for('signup'))
        finally:
            conn.close()

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        conn = get_db()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user'] = user['username']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('index'))


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        flash('No file part in the request', 'error')
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    # Save uploaded image
    upload_folder = os.path.join(app.root_path, 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Simulated federated step (for project concept only)
    aggregated_model = federated_average([model, model, model])

    # Predict
    results = aggregated_model(file_path)
    img = cv2.imread(file_path)

    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())

        # Correct class mapping
        if cls_id == 1:
            label = f"Brain Tumor {confidence:.2f}"
            color = (0, 0, 255)  # Red
        else:
            label = f"No Brain Tumor {confidence:.2f}"
            color = (0, 255, 0)  # Green

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Save result image
    result_folder = os.path.join(app.root_path, 'static', 'results')
    os.makedirs(result_folder, exist_ok=True)
    result_filename = "result_" + file.filename
    result_path = os.path.join(result_folder, result_filename)
    cv2.imwrite(result_path, img)

    return render_template(
        'predict.html',
        result_image=url_for('static', filename='results/' + result_filename)
    )


if __name__ == '__main__':
    app.run(debug=True)
