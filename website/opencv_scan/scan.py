import cv2
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Important: Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import logging

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Camera settings
realWidth = 640
realHeight = 480
videoWidth = 320
videoHeight = 240
videoChannels = 3
videoFrameRate = 15

# Processing parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 128
bufferIndex = 0

bpmCalculationFrequency = 15
bpmBufferIndex = 0
bpmBufferSize = 10

# Data buffers
bpm_data = []
time_data = []
rr_intervals = []
hrv_data = []
stress_data = []
spo2_data = []
rppg_signal = []

# Global variables for latest values (for live graph)
latest_values = {
    "bpm": 0,
    "hrv": 0,
    "stress": 0,
    "spo2": 0,
    "avg_bpm": 0,
    "avg_hrv": 0,
    "avg_stress": 0,
    "avg_spo2": 0,
    "scan_completed": False,
    "progress": 0,
    "elapsed_seconds": 0
}

# Flags
graph_started = False
start_time = time.time()
scan_start_time = None
MAX_SCAN_SECONDS = 45
WARMUP_FRAMES = 5
DETECT_EVERY_N = 5
frame_counter = 0
last_face = None

# Structured logger for intermediate steps
logger = logging.getLogger("scan")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Initialize buffers
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))

def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

firstGauss = buildGauss(firstFrame, levels + 1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))
frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)
bpmBuffer = np.zeros((bpmBufferSize))

# 💥 New: Function to reset everything before every new scan
def reset_scan_data():
    global bpm_data, time_data, rr_intervals, hrv_data, stress_data, spo2_data, rppg_signal
    global latest_values, bufferIndex, bpmBufferIndex, graph_started, start_time, scan_start_time
    global frame_counter, last_face

    bpm_data.clear()
    time_data.clear()
    rr_intervals.clear()
    hrv_data.clear()
    stress_data.clear()
    spo2_data.clear()
    rppg_signal.clear()

    latest_values.update({
        "bpm": 0,
        "hrv": 0,
        "stress": 0,
        "spo2": 0,
        "avg_bpm": 0,
        "avg_hrv": 0,
        "avg_stress": 0,
        "avg_spo2": 0,
        "scan_completed": False
    })

    bufferIndex = 0
    bpmBufferIndex = 0
    graph_started = False
    start_time = time.time()
    scan_start_time = time.time()
    frame_counter = 0
    last_face = None

def generate_frames():
    global bufferIndex, bpmBufferIndex, graph_started, start_time, scan_start_time, frame_counter, last_face

    reset_scan_data()  # Reset everything before starting new scan

    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, realWidth)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, realHeight)
    webcam.set(cv2.CAP_PROP_FPS, videoFrameRate)
    webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Prefer MJPG for faster USB cams
    try:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        webcam.set(cv2.CAP_PROP_FOURCC, fourcc)
    except Exception:
        pass

    # Warmup frames to stabilize exposure
    for _ in range(WARMUP_FRAMES):
        ret, _ = webcam.read()
        if not ret:
            break
    logger.info("Camera warmup complete. Starting scan loop.")

    loop_count = 0  # Add a counter to track the number of loops

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        frame_counter += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        if frame_counter % DETECT_EVERY_N == 0 or last_face is None:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                last_face = faces[0]
                logger.debug(f"Face detected at {last_face} (frame {frame_counter}).")
        if last_face is None:
            continue

        x, y, w, h = last_face
        # Define a centered ROI around the face, clamped to frame bounds
        y1 = max(y + int(0.15*h), 0)
        y2 = min(y + int(0.45*h), realHeight)
        x1 = max(x + int(0.2*w), 0)
        x2 = min(x + int(0.8*w), realWidth)
        if y2 - y1 < videoHeight or x2 - x1 < videoWidth:
            # fallback to central crop if face box too small
            detectionFrame = frame[videoHeight // 2:realHeight - videoHeight // 2, videoWidth // 2:realWidth - videoWidth // 2, :]
        else:
            detectionFrame = frame[y1:y2, x1:x2, :]
            detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))

        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
        # Capture simple rPPG proxy: mean green channel in ROI
        roi_green = detectionFrame[:, :, 1].astype(np.float32)
        rppg_signal.append(float(np.mean(roi_green)))
        if bufferIndex % bpmCalculationFrequency == 0:
            logger.debug(f"rPPG sample count: {len(rppg_signal)}; mean green: {rppg_signal[-1]:.2f}")
        fourierTransform = np.fft.fft(videoGauss, axis=0)
        fourierTransform[mask == False] = 0

        if bufferIndex % bpmCalculationFrequency == 0:
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

            bpm_data.append(bpm)
            logger.debug(f"Computed BPM: {bpm:.2f}")

            if bpm > 0:
                rr_interval = 60 * 1000 / bpm
                rr_intervals.append(rr_interval)
                avg_hrv = np.mean(rr_intervals[-10:]) if len(rr_intervals) >= 10 else np.mean(rr_intervals)
                hrv_data.append(avg_hrv)
                logger.debug(f"HRV (ms est): {avg_hrv:.2f}")

                stress_score = (avg_hrv - 0.75) * 50 + (bpm - 75) * 0.1
                stress_data.append(stress_score)

                spo2 = 98 - (0.05 * (bpm - 75)) - (0.02 * (800 - avg_hrv))
                spo2 = max(min(spo2, 100), 90)
                spo2_data.append(spo2)
                logger.debug(f"Stress: {stress_score:.2f}, SpO2: {spo2:.2f}")

                latest_values["bpm"] = round(bpm, 2)
                latest_values["hrv"] = round(avg_hrv, 2)
                latest_values["stress"] = round(stress_score, 2)
                latest_values["spo2"] = round(spo2, 2)

        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha
        filteredFrame = reconstructFrame(filtered, bufferIndex, levels)

        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        frame[videoHeight // 2:realHeight - videoHeight // 2, videoWidth // 2:realWidth - videoWidth // 2, :] = outputFrame

        bufferIndex = (bufferIndex + 1) % bufferSize

        current_time = time.time() - start_time
        time_data.append(current_time)

        if not graph_started and current_time >= 5:
            graph_started = True
            start_time = time.time()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Update progress for UI
        elapsed_scan = time.time() - scan_start_time
        latest_values["elapsed_seconds"] = int(elapsed_scan)
        latest_values["progress"] = int(min(100, (elapsed_scan / MAX_SCAN_SECONDS) * 100))

        # Stop after max scan duration or sufficient data
        loop_count += 1
        if elapsed_scan >= MAX_SCAN_SECONDS:
            break
        if graph_started and elapsed_scan >= 20 and len(bpm_data) > 10:
            break

    webcam.release()

    latest_values["avg_bpm"] = round(np.mean(bpm_data), 2) if bpm_data else 0
    latest_values["avg_hrv"] = round(np.mean(hrv_data), 2)/10 if hrv_data else 0
    latest_values["avg_stress"] = round(np.mean(stress_data), 2)/10000 if stress_data else 0
    latest_values["avg_spo2"] = round(np.mean(spo2_data), 2) if spo2_data else 0
    latest_values["scan_completed"] = True
    latest_values["progress"] = 100

    save_static_final_graph()
    logger.info("Scan completed. Averages computed and graph saved.")


def save_static_final_graph():
    try:
        plt.figure(figsize=(10, 9))

        time_points = list(range(len(bpm_data)))
        time_hrv = list(range(len(hrv_data)))
        time_stress = list(range(len(stress_data)))
        time_spo2 = list(range(len(spo2_data)))

        plt.subplot(221)
        plt.plot(time_points, bpm_data, marker='o', linestyle='-', color='blue')
        plt.title("Heart Rate (BPM)")
        plt.xlabel("Time")
        plt.ylabel("BPM")
        plt.grid(True)

        plt.subplot(222)

        # Ignore first 3 data points
        time_hrv_trimmed = time_hrv[3:]
        hrv_data_trimmed = hrv_data[3:]

        plt.plot(time_hrv_trimmed, hrv_data_trimmed, marker='o', linestyle='-', color='purple')
        plt.title("HRV (ms)")
        plt.xlabel("Time")
        plt.ylabel("HRV (ms)")
        plt.grid(True)


        plt.subplot(223)
        plt.plot(time_stress, stress_data, marker='o', linestyle='-', color='red')
        plt.title("Stress Level")
        plt.xlabel("Time")
        plt.ylabel("Stress")
        plt.grid(True)

        plt.subplot(224)
        plt.plot(time_spo2, spo2_data, marker='o', linestyle='-', color='green')
        plt.title("SpO₂ (%)")
        plt.xlabel("Time")
        plt.ylabel("SpO₂ %")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('static/final_graph.png')
        plt.close()
    except Exception as e:
        print(f"Error saving static graph: {e}")

# API for real-time data
def get_live_data():
    return latest_values

# Provide last N samples of rPPG signal for downstream analysis
def get_rppg_signal(max_samples: int = 600):
    if len(rppg_signal) <= max_samples:
        return rppg_signal
    return rppg_signal[-max_samples:]
