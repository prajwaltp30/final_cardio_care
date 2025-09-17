import mysql.connector
from .db import get_db_connection
from flask import jsonify

def save_user_data(email, age, sex, diabetes, famhistory, smoking, obesity, alcohol, exercise_hours, diet,
                   heart_problem, bmi, physical_activity, sleep_hours, blood_pressure_systolic,
                   blood_pressure_diastolic):
    conn = None  # Initialize conn outside the try block

    try:
        # conn = mysql.connector.connect(
        #     host="cardio-care-nagavenisgowdru-bd3e.k.aivencloud.com",
        #     port=21607,
        #     user= "avnadmin",
        #     password="AVNS_kU3Vg3PUMtzarpZ_vdl",
        #     database="cardio_care"
        # )
        conn = get_db_connection()

        cursor = conn.cursor()

        # Ensure table and email unique index exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_details (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) UNIQUE,
                age INT,
                sex TINYINT,
                diabetes TINYINT,
                famhistory TINYINT,
                smoking TINYINT,
                obesity TINYINT,
                alcohol TINYINT,
                exercise_hours FLOAT,
                diet TINYINT,
                heart_problem TINYINT,
                bmi FLOAT,
                physical_activity INT,
                sleep_hours FLOAT,
                blood_pressure_systolic INT,
                blood_pressure_diastolic INT
            )
        """)
        conn.commit()

        # Upsert on email
        query_insert = (
            "INSERT INTO user_details (email, age, sex, diabetes, famhistory, smoking, obesity, alcohol, exercise_hours, diet, heart_problem, bmi, physical_activity, sleep_hours, blood_pressure_systolic, blood_pressure_diastolic) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        values = (email, age, sex, diabetes, famhistory, smoking, obesity, alcohol, exercise_hours, diet,
                  heart_problem, bmi, physical_activity, sleep_hours, blood_pressure_systolic,
                  blood_pressure_diastolic)
        try:
            cursor.execute(query_insert, values)
            conn.commit()
        except mysql.connector.errors.IntegrityError:
            query_update = (
                "UPDATE user_details SET age=%s, sex=%s, diabetes=%s, famhistory=%s, smoking=%s, obesity=%s, alcohol=%s, exercise_hours=%s, diet=%s, heart_problem=%s, bmi=%s, physical_activity=%s, sleep_hours=%s, blood_pressure_systolic=%s, blood_pressure_diastolic=%s WHERE email=%s"
            )
            update_values = (age, sex, diabetes, famhistory, smoking, obesity, alcohol, exercise_hours, diet,
                             heart_problem, bmi, physical_activity, sleep_hours, blood_pressure_systolic,
                             blood_pressure_diastolic, email)
            cursor.execute(query_update, update_values)
            conn.commit()
        

    except Exception as e:
        # If an exception occurs, return the error message
        return jsonify({'message': str(e)}), 500

    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
