from flask import Flask, request, jsonify, render_template, redirect, url_for, session, Response
import subprocess
import os
import numpy as np
import mysql.connector
from database.db import get_db_connection
from database.save_user import save_user_data
from ml_logic.predict import get_user_details, predict_manually, compute_atherosclerosis_risk
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch

# Import from updated scan.py
from opencv_scan.scan import generate_frames, get_live_data
from opencv_scan.scan import get_rppg_signal

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this

# ---- ROUTES ----

@app.route('/')
def index():
    if 'email' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))

@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        email = data.get('email')
        nickname = data.get('nickname')
        password = data.get('password')

        if not email or not password:
            return jsonify({"message": "Email and password are required"}), 400

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    email VARCHAR(255) UNIQUE,
                    username VARCHAR(255),
                    nickname VARCHAR(255),
                    password VARCHAR(255)
                )
            """)
            conn.commit()
            # Upsert-like behavior: try insert, if exists update username/nickname/password
            try:
                cursor.execute(
                    "INSERT INTO users (email, username, nickname, password) VALUES (%s, %s, %s, %s)",
                    (email, username, nickname, password)
                )
                conn.commit()
            except mysql.connector.errors.IntegrityError:
                cursor.execute(
                    "UPDATE users SET username=%s, nickname=%s, password=%s WHERE email=%s",
                    (username, nickname, password, email)
                )
                conn.commit()
            return jsonify({"message": "User registered successfully"}), 200
        except Exception as e:
            return jsonify({"message": str(e)}), 400
        finally:
            try:
                cursor.close()
                conn.close()
            except Exception:
                pass

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({"success": False, "message": "Email and password are required"}), 400

        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT email, password FROM users WHERE email=%s", (email,))
            user = cursor.fetchone()
            if not user:
                return jsonify({"success": False, "message": "Email not found"}), 404
            if user['password'] == password:
                session['email'] = email
                return jsonify({"success": True, "redirect": url_for('index')})
            else:
                return jsonify({"success": False, "message": "Incorrect password"}), 401
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500
        finally:
            try:
                cursor.close()
                conn.close()
            except Exception:
                pass

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('signup'))

@app.route('/home')
def home():
    if 'email' in session:
        return render_template('home.html')
    else:
        return redirect(url_for('login'))

# --- MODIFIED SECTION FOR VIDEO AND LIVE DATA ---

@app.route('/start', methods=['POST'])
def start_script():
    try:
        return render_template('scan.html')
    except Exception as e:
        return f"Error starting scan: {str(e)}", 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_data')
def live_data():
    data = get_live_data()
    return jsonify(data)

# --- END OF MODIFIED SECTION ---

@app.route('/results', methods=['GET', 'POST'])
def show_results():
    if request.method == 'POST':
        avg_bpm = request.form.get('avg_bpm')
        avg_hrv = request.form.get('avg_hrv')
        avg_stress = request.form.get('avg_stress')
        avg_spo2 = request.form.get('avg_spo2')

        # 👉 Store into session!
        session['avg_bpm'] = avg_bpm
        session['avg_hrv'] = avg_hrv
        session['avg_stress'] = avg_stress
        session['avg_spo2'] = avg_spo2
    else:
        avg_bpm = avg_hrv = avg_stress = avg_spo2 = None

    additional_graph_exists = os.path.exists("static/final_graph.png")

    bpm_data = []
    if os.path.exists("bpm_values.txt"):
        with open("bpm_values.txt", "r") as file:
            for line in file:
                bpm = float(line.strip())
                bpm_data.append(bpm)

    average_bpm = np.mean(bpm_data[5:]) if len(bpm_data) > 5 else 0

    return render_template('results.html',
                           additional_graph_exists=additional_graph_exists,
                           average_bpm=average_bpm,
                           avg_bpm=avg_bpm,
                           avg_hrv=avg_hrv,
                           avg_stress=avg_stress,
                           avg_spo2=avg_spo2)


@app.route('/details_form')
def details_form():
    return render_template('detail.html')

@app.route('/save_details', methods=['POST'])
def save_details():
    try:
        form = request.form
        age = form['age']
        sex = form.get('sex') == '1'
        diabetes = 'diabetes' in form
        famhistory = 'famhistory' in form
        smoking = 'smoking' in form
        obesity = 'obesity' in form
        alcohol = 'alcohol' in form
        exercise = form['exercise']
        diet = form['diet']
        heartproblem = 'heartproblem' in form
        bmi = form['bmi']
        physicalactivity = form['physicalactivity']
        sleep = form['sleep']
        bp1 = form['bp1']
        bp2 = form['bp2']
        email = session.get('email')

        save_user_data(email, age, sex, diabetes, famhistory, smoking, obesity, alcohol,
                       exercise, diet, heartproblem, bmi, physicalactivity, sleep, bp1, bp2)

        return render_template('index.html')

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return "An error occurred while processing your request.", 500

# [rest of ml prediction, download_report remains same, not changed]

@app.route('/ml', methods=['GET', 'POST'])
def ml():
    avg_bpm = session.get('avg_bpm')
    avg_hrv = session.get('avg_hrv')
    avg_stress = session.get('avg_stress')
    avg_spo2 = session.get('avg_spo2')

    email = session.get('email')
    user_details = None
    if email:
        user_details = get_user_details(email)

    return render_template('ml.html',
                           user_details=user_details,
                           avg_bpm=avg_bpm,
                           avg_hrv=avg_hrv,
                           avg_stress=avg_stress,
                           avg_spo2=avg_spo2)



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = session.get('email')
        if not email:
            return render_template('ml.html', error="Session expired. Please log in again.")

        user_details = get_user_details(email)
        if not user_details:
            return render_template('ml.html', error="User details not found in the database.")

        age = int(user_details['age'])
        sex = int(user_details['sex'])
        diabetes = int(user_details['diabetes'])
        family_history = int(user_details['famhistory'])
        smoking = int(user_details['smoking'])
        obesity = int(user_details['obesity'])
        alcohol_consumption = int(user_details['alcohol'])
        exercise_hours_per_week = float(user_details['exercise_hours'])
        diet = int(user_details['diet'])
        previous_heart_problems = int(user_details['heart_problem'])
        bmi = float(user_details['bmi'])
        physical_activity_days_per_week = int(user_details['physical_activity'])
        sleep_hours_per_day = float(user_details['sleep_hours'])
        bp1 = int(user_details['blood_pressure_systolic'])
        bp2 = int(user_details['blood_pressure_diastolic'])

        try:
            heart_rate = int(float(request.form['heart_rate']))
            stress_level = int(float(request.form['stress_level']))
            hrv = int(float(request.form['hrv']))
            spo2 = int(float(request.form['spo2']))
        except (ValueError, TypeError):
            return render_template('ml.html', error="Invalid input data.")


        predicted_heart_attack_risk, predicted_percentage = predict_manually(
            age, sex, heart_rate, diabetes, family_history, smoking, obesity,
            alcohol_consumption, exercise_hours_per_week, diet, previous_heart_problems,
            0, stress_level, bmi, physical_activity_days_per_week, sleep_hours_per_day, bp1, bp2
        )

        messages = []
        if hrv < 45:
            messages.append("Your HRV is too low")
            predicted_percentage += 25.8
        if hrv > 200:
            messages.append("Your HRV is too high")
            predicted_percentage += 5.65
        if spo2 < 90:
            messages.append("Your SpO2 level is below normal.")
            predicted_percentage += 4.5

        if age > 50:
            messages.append("Age greater than 50.")
        if heart_rate < 60 or heart_rate > 100:
            messages.append("Your heart rate is not good.")
        if diabetes:
            messages.append("You have diabetes.")
        if family_history:
            messages.append("You have a family history of heart problems.")
        if smoking:
            messages.append("You are a smoker.")
        if obesity:
            messages.append("You are obese.")
        if alcohol_consumption:
            messages.append("You consume alcohol.")
        if previous_heart_problems:
            messages.append("You have had previous heart problems.")
        if stress_level > 6:
            messages.append("Stress Level is High, Take rest or Hangout.")
        if bp1 > 140 or bp2 > 80:
            messages.append("Your blood pressure is high.")

        if predicted_heart_attack_risk == 0 and predicted_percentage < 50:
            category = "Heart attack risk 0, heart attack risk percentage less than 50. You are safe. Please take care of yourself."
        elif predicted_heart_attack_risk == 0 and predicted_percentage >= 50:
            category = "Heart attack risk 0, heart attack risk percentage greater than 50. Please consult the doctor by sharing the report."
        elif predicted_heart_attack_risk == 1 and predicted_percentage < 50:
            category = "Heart attack risk 1, heart attack risk percentage less than 50. Something unpredictable. Please consult a doctor."
        else:
            category = "Heart attack risk 1, heart attack risk percentage greater than 50. Don't be afraid. Just contact the nearest hospital. That's all."

        # Compute atherosclerosis risk from captured rPPG
        rppg = get_rppg_signal()
        ath = compute_atherosclerosis_risk(rppg)

        return render_template('result.html', heart_attack_risk=predicted_heart_attack_risk,
                               percentage=predicted_percentage, messages=messages, category=category,
                               athero=ath)
    else:
        return render_template('ml.html')

from datetime import datetime

from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from io import BytesIO
from flask import Response, session, request
import os

@app.route('/download_report', methods=['POST'])
def download_report():
    if request.method == 'POST':
        email = session.get('email')
        if not email:
            return "Session expired. Please log in again."

        user_details = get_user_details(email)
        if not user_details:
            return "User details not found in the database."

        report_content = request.form['report_content']

        # Fetch avg values from session
        avg_bpm = session.get('avg_bpm', 'N/A')
        avg_hrv = session.get('avg_hrv', 'N/A')
        avg_stress = session.get('avg_stress', 'N/A')
        avg_spo2 = session.get('avg_spo2', 'N/A')

        # Mapping values
        sex_map = {0: 'Female', 1: 'Male'}
        yes_no_map = {0: 'No', 1: 'Yes'}
        diet_map = {0: 'Poor Diet', 1: 'Average Diet', 2: 'Good Diet'}

        formatted_details = {
            'Vital': 'Value',
            'Age': user_details['age'],
            'Sex': sex_map.get(user_details['sex'], 'Unknown'),
            'Diabetes': yes_no_map.get(user_details['diabetes'], 'Unknown'),
            'Family History': yes_no_map.get(user_details['famhistory'], 'Unknown'),
            'Smoking': yes_no_map.get(user_details['smoking'], 'Unknown'),
            'Obesity': yes_no_map.get(user_details['obesity'], 'Unknown'),
            'Alcohol Consumption': yes_no_map.get(user_details['alcohol'], 'Unknown'),
            'Exercise Hours/Week': user_details['exercise_hours'],
            'Diet': diet_map.get(user_details['diet'], 'Unknown'),
            'Previous Heart Problems': yes_no_map.get(user_details['heart_problem'], 'Unknown'),
            'BMI': user_details['bmi'],
            'Physical Activity Days/Week': user_details['physical_activity'],
            'Sleep Hours/Day': user_details['sleep_hours'],
            'Blood Pressure (Sys/Dia)': f"{user_details['blood_pressure_systolic']}/{user_details['blood_pressure_diastolic']}",
        }

        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=20,
            alignment=1,  # Center
            textColor=colors.darkblue
        )
        normal_style = ParagraphStyle(
            'NormalStyle',
            parent=styles['BodyText'],
            fontSize=12,
            spaceAfter=10,
        )

        elements = []

        # Top title
        elements.append(Paragraph("Cardio Care Online Report", title_style))
        elements.append(Spacer(1, 10))

        # Date and time
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        elements.append(Paragraph(f"<b>Generated on:</b> {current_time}", normal_style))
        elements.append(Spacer(1, 20))

        # User details table
        user_data = [[Paragraph(f"<b>{key}</b>", normal_style), Paragraph(str(value), normal_style)] for key, value in formatted_details.items()]
        user_table = Table(user_data, colWidths=[220, 300])
        user_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(user_table)
        elements.append(Spacer(1, 20))

        # Vitals neatly
        vitals_data = [
            ["Vital", "Value"],
            ["Average Heart Rate (BPM)", avg_bpm],
            ["Average HRV (ms)", avg_hrv],
            ["Average Stress Index", avg_stress],
            ["Average SpO2 (%)", avg_spo2],
        ]
        vitals_table = Table(vitals_data, colWidths=[220, 300])
        vitals_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.aliceblue),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(vitals_table)

        # 👉 After vitals, go to new page
        elements.append(PageBreak())

        # Full second page - Report Summary
        elements.append(Paragraph("<b>Report Summary:</b>", title_style))
        elements.append(Spacer(1, 20))
        elements.append(Paragraph(report_content, normal_style))

        # 👉 After report summary, go to next page
        elements.append(PageBreak())

        # Third page - Graph
        graph_path = 'static/final_graph.png'
        if os.path.exists(graph_path):
            img = Image(graph_path, width=500, height=600)
            elements.append(img)

        doc.build(elements)

        pdf_buffer.seek(0)
        response = Response(pdf_buffer, mimetype='application/pdf')
        response.headers.set("Content-Disposition", "attachment", filename="cardio_care_report.pdf")

        return response
    else:
        return "Method not allowed."



# -------------- Integrated Chatbot (upload + chat) -----------------
import fitz  # PyMuPDF
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

def ask_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful health report assistant. Summarize simply."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    resp = requests.post(GROQ_API_URL, headers=headers, json=data)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"Groq error: {resp.text}"

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    extracted_text = extract_text_from_pdf(file)
    session['report_text'] = extracted_text
    ai_response = ask_groq(f"Summarize and explain this health report:\n{extracted_text}")
    return jsonify({"ai_response": ai_response})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    report_text = session.get('report_text')
    if not report_text:
        return jsonify({"error": "No report uploaded yet."}), 400
    full_prompt = (
        f"Given this health report:\n{report_text}\n\n"
        f"Answer under 50 words: {user_message}"
    )
    ai_response = ask_groq(full_prompt)
    return jsonify({"ai_response": ai_response})

@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

# -------------- Integrated Nearby Doctors -----------------
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@app.route('/nearby')
def nearby_index():
    return render_template('nearby_index.html', google_api_key=GOOGLE_API_KEY)

@app.route('/nearby-doctors', methods=['POST'])
def get_nearby_doctors():
    data = request.get_json()
    user_lat = data.get("lat")
    user_lng = data.get("lng")
    places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    places_params = {
        "location": f"{user_lat},{user_lng}",
        "radius": 3000,
        "type": "doctor|hospital",
        "key": GOOGLE_API_KEY
    }
    places_response = requests.get(places_url, params=places_params)
    places_data = places_response.json()
    results = places_data.get("results", [])
    if not results:
        return jsonify([])
    destinations = [f"{p['geometry']['location']['lat']},{p['geometry']['location']['lng']}" for p in results]
    matrix_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    matrix_params = {
        "origins": f"{user_lat},{user_lng}",
        "destinations": "|".join(destinations),
        "key": GOOGLE_API_KEY
    }
    matrix_response = requests.get(matrix_url, params=matrix_params)
    distance_data = matrix_response.json()
    for i, place in enumerate(results):
        try:
            place["distance_text"] = distance_data["rows"][0]["elements"][i]["distance"]["text"]
            place["distance_value"] = distance_data["rows"][0]["elements"][i]["distance"]["value"]
        except KeyError:
            place["distance_text"] = "Unknown"
            place["distance_value"] = float('inf')
    sorted_places = sorted(results, key=lambda x: x["distance_value"])
    doctors = []
    for place in sorted_places:
        loc = place["geometry"]["location"]
        place_details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        place_details_params = {
            "place_id": place["place_id"],
            "fields": "name,formatted_phone_number,international_phone_number",
            "key": GOOGLE_API_KEY
        }
        details_response = requests.get(place_details_url, params=place_details_params)
        details_data = details_response.json()
        phone_number = None
        if details_data.get("result"):
            phone_number = details_data["result"].get("international_phone_number")
        doctors.append({
            "name": place["name"],
            "lat": loc["lat"],
            "lng": loc["lng"],
            "address": place.get("vicinity", ""),
            "distance": place.get("distance_text", "Unknown"),
            "rating": place.get("rating", "N/A"),
            "phone": phone_number
        })
    return jsonify(doctors)

if __name__ == '__main__':
    app.run(debug=True)
