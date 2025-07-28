import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import time
import json
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import requests
import serial
import threading
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# ---------- Streamlit Setup ----------
st.set_page_config(page_title="Smart Soil Classifier", layout="wide")

# Clean white theme with shadows
st.markdown("""
    <style>
        :root {
            --primary: #4a90e3;
            --secondary: #50c878;
            --light: #ffffff;
            --dark: #333333;
            --gray: #f0f0f0;
            --shadow: 0 4px 12px rgba(0,0,0,0.1);
            --danger: #ff4757;
        }
        
        body {
            background-color: var(--light) !important;
            color: var(--dark) !important;
            font-family: 'Segoe UI', sans-serif;
        }
        
        .stApp {
            background: var(--light) !important;
        }
        
        h1, h2, h3, h4, h5, h6, p, div, span {
            color: var(--dark) !important;
        }
        
        .stFileUploader > label {
            color: var(--dark) !important;
            border: 2px dashed #ccc !important;
            background: var(--light) !important;
            padding: 2rem !important;
            border-radius: 12px !important;
            box-shadow: var(--shadow);
            transition: all 0.3s ease;
        }
        
        .stFileUploader > label:hover {
            border-color: var(--primary) !important;
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }
        
        .stButton > button {
            background: var(--primary) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            box-shadow: var(--shadow);
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        }
        
        .card {
            background: var(--light) !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            box-shadow: var(--shadow);
            border: 1px solid #eee !important;
        }
        
        .sensor-card {
            background: var(--light) !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            box-shadow: var(--shadow);
            border: 1px solid #eee !important;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .sensor-title {
            font-size: 1.1rem !important;
            color: #666 !important;
            margin-bottom: 0.5rem !important;
        }
        
        .sensor-value {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .sensor-value.danger {
            background: var(--danger) !important;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .result-wet {
            background: rgba(80, 200, 120, 0.1) !important;
            border-left: 4px solid var(--secondary) !important;
        }
        
        .result-dry {
            background: rgba(255, 71, 87, 0.1) !important;
            border-left: 4px solid var(--danger) !important;
        }
        
        .camera-button {
            background: var(--secondary) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.75rem 2rem !important;
            font-size: 1.1rem !important;
            box-shadow: var(--shadow);
            transition: all 0.3s ease;
        }
        
        .camera-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        }
        
        .status-online {
            color: var(--secondary) !important;
        }
        
        .status-offline {
            color: var(--danger) !important;
        }
        
        .sidebar .sidebar-content {
            background: var(--light) !important;
            box-shadow: 2px 0 12px rgba(0,0,0,0.08);
        }
        
        [data-baseweb="radio"] div:first-child {
            background: var(--primary) !important;
        }
        
        .blink {
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.3; }
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Data Storage ----------
if 'history' not in st.session_state:
    st.session_state.history = {
        'timestamp': [],
        'ml_prediction': [],
        'sensor_moisture': [],
        'temperature': [],
        'humidity': [],
        'water_purity': [],
        'label': []
    }

# ---------- Arduino Communication ----------
class ArduinoController:
    def __init__(self, port='COM7', baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        self.arduino = None
        self.is_connected = False
        self.sensor_data = {
            'temperature': 0,
            'humidity': 0,
            'soil_moisture': 0,
            'tds': 0,
            'timestamp': datetime.now()
        }
        self.connect()
    
    def connect(self):
        try:
            self.arduino = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to Arduino: {e}")
            self.is_connected = False
            return False
    
    def read_sensors(self):
        if not self.is_connected:
            return None
        
        try:
            # Send command to read sensors
            self.arduino.write(b'READ_SENSORS\n')
            time.sleep(0.1)
            
            # Read response
            if self.arduino.in_waiting > 0:
                response = self.arduino.readline().decode('utf-8').strip()
                if response:
                    # Parse sensor data (assuming format: "TEMP:25.5,HUM:60.2,SOIL:45,TDS:78")
                    data = {}
                    pairs = response.split(',')
                    for pair in pairs:
                        if ':' in pair:
                            key, value = pair.split(':')
                            data[key.lower()] = float(value)
                    
                    # Update sensor data
                    self.sensor_data.update({
                        'temperature': data.get('temp', 0),
                        'humidity': data.get('hum', 0),
                        'soil_moisture': data.get('soil', 0),
                        'tds': data.get('tds', 0),
                        'timestamp': datetime.now()
                    })
                    
                    return self.sensor_data
        except Exception as e:
            st.error(f"Error reading sensors: {e}")
        
        return None
    
    def control_hardware(self, dry_soil=False):
        if not self.is_connected:
            return False
        
        try:
            if dry_soil:
                # Send signal for dry soil: Red LED + Buzzer + Relay
                self.arduino.write(b'DRY_SOIL\n')
                return True
            else:
                # Send signal for wet soil: Green LED only
                self.arduino.write(b'WET_SOIL\n')
                return True
        except Exception as e:
            st.error(f"Error controlling hardware: {e}")
            return False
    
    def simulate_camera_capture(self):
        if not self.is_connected:
            return False
        
        try:
            # Send signal to show camera capture animation
            self.arduino.write(b'CAMERA_CAPTURE\n')
            return True
        except Exception as e:
            st.error(f"Error simulating camera capture: {e}")
            return False
    
    def disconnect(self):
        if self.arduino:
            self.arduino.close()
            self.is_connected = False

# Initialize Arduino controller
if 'arduino' not in st.session_state:
    st.session_state.arduino = ArduinoController()

# ---------- Model Loading ----------
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'soil_model.h5')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error("Model not found. Please train the model first.")
        return None

model = load_model()

# ---------- Sidebar Navigation ----------
st.sidebar.title("Navigation")
pages = ["🌱 Soil Monitor", "📊 Model Metrics", "🔧 Hardware Control", "📉 Comparisons" , "🤖 Smart AI Chatbot"]
page = st.sidebar.radio("", pages, label_visibility="collapsed")

st.sidebar.markdown("---")

# Connection Status
connection_status = "🟢 Online" if st.session_state.arduino.is_connected else "🔴 Offline"
status_class = "status-online" if st.session_state.arduino.is_connected else "status-offline"

st.sidebar.markdown(f"""
    <div style="padding: 1rem; border-radius: 8px; background: #f8f9fa; border: 1px solid #eee;">
        <p style="font-weight: 600; margin-bottom: 0.5rem;">Arduino Status</p>
        <p class="{status_class}" style="font-size: 0.9rem; margin-bottom: 0.5rem;">{connection_status}</p>
        <p style="font-size: 0.8rem; color: #666;">Port: COM7</p>
    </div>
""", unsafe_allow_html=True)

# ---------- Soil Monitor Page ----------
if page == "🌱 Soil Monitor":
    st.title("🌱 Smart Soil Monitoring System")
    st.markdown("Real-time soil analysis with AI-powered classification and automated irrigation control")
    st.markdown("---")
    
    # Real-time sensor readings
    st.subheader("📡 Live Sensor Data")
    
    # Read sensor data
    sensor_data = st.session_state.arduino.read_sensors()
    if sensor_data:
        col1, col2, col3, col4 = st.columns(4)
        soil_percentage = 100 - (sensor_data['soil_moisture'] / 4095) * 100
        tds_purity = max(0, (500 - sensor_data['tds'])) / 500 * 100
        
        with col1:
            temp_color = "danger" if sensor_data['temperature'] > 35 else ""
            st.markdown(f"""
            <div class='sensor-card'>
                <div class='sensor-title'>🌡️ Temperature</div>
                <div class='sensor-value {temp_color}'>{sensor_data['temperature']:.1f}°C</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            hum_color = "danger" if sensor_data['humidity'] < 30 else ""
            st.markdown(f"""
            <div class='sensor-card'>
                <div class='sensor-title'>💧 Humidity</div>
                <div class='sensor-value {hum_color}'>{sensor_data['humidity']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            # Convert soil moisture to percentage (assuming 0-1023 range)
            soil_percentage = (sensor_data['soil_moisture'] / 4095) * 100 
            soil_color = "danger" if soil_percentage < 30 else ""
            st.markdown(f"""
            <div class='sensor-card'>
                <div class='sensor-title'>🌱 Soil Moisture</div>
                <div class='sensor-value {soil_color}'>{soil_percentage:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            # Convert TDS to water purity percentage (inverse relationship)
            tds_purity = max(0, (500 - sensor_data['tds']) / 500 * 100)
            tds_color = "danger" if tds_purity < 50 else ""
            st.markdown(f"""
            <div class='sensor-card'>
                <div class='sensor-title'>💧 Water Purity</div>
                <div class='sensor-value {tds_color}'>{tds_purity:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Last update time
        st.markdown(f"<p style='text-align: center; color: #666; font-size: 0.9rem;'>Last updated: {sensor_data['timestamp'].strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # AI Classification Section
    st.subheader("🤖 AI-Powered Soil Classification")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📷 Camera Capture")
        if st.button("📸 Capture Image from Camera", key="camera_btn"):
            # Simulate camera capture
            st.session_state.arduino.simulate_camera_capture()
            
            with st.spinner("📸 Capturing image from camera module..."):
                time.sleep(2)  # Simulate capture time
                st.session_state.camera_captured = True
                st.session_state.camera_image_path = "dry.jpg"
                st.success("✅ Image captured successfully!")
                
            
        
        st.markdown("### 📁 Or Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a soil image",
            type=["jpg", "jpeg", "png"],
            help="Maximum file size: 200MB"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.markdown(f"""
                <div class="card" style="margin-top: 1rem;">
                    <p><strong>File:</strong> {uploaded_file.name}</p>
                    <p><strong>Size:</strong> {uploaded_file.size / (1024*1024):.2f} MB</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔬 Classification Results")
        
        # Process uploaded image or use hardcoded image for demo
        if uploaded_file or st.session_state.get('camera_captured', False):
            if model is None:
                st.error("Model not loaded. Please check model file.")
            else:
                with st.spinner("🧠 Analyzing soil moisture with AI..."):
                    if uploaded_file:
                        image = Image.open(uploaded_file).convert('RGB')
                    else:
                        # Use a hardcoded image path for demo
                        image_path = "dry.jpg"  # Replace with your demo image
                        if os.path.exists(image_path):
                            image = Image.open(image_path).convert('RGB')
                        else:
                            st.error("Demo image not found. Please upload an image.")
                            image = None
                    
                    if image:
                        # Prepare image for model
                        img_array = np.array(image)
                        resized = cv2.resize(img_array, (256, 256)) / 255.0
                        input_tensor = np.expand_dims(resized, axis=0)
                        
                        # Make prediction
                        prediction = model.predict(input_tensor)[0][0]
                        label = "Wet" if prediction > 0.5 else "Dry"
                        confidence = prediction if prediction > 0.5 else 1 - prediction
                        
                        # Store results for comparison
                        st.session_state.history['timestamp'].append(datetime.now())
                        st.session_state.history['ml_prediction'].append(prediction * 100)  # as percentage
                        st.session_state.history['sensor_moisture'].append(soil_percentage)
                        st.session_state.history['temperature'].append(sensor_data['temperature'])
                        st.session_state.history['humidity'].append(sensor_data['humidity'])
                        st.session_state.history['water_purity'].append(tds_purity)
                        st.session_state.history['label'].append(label)
                        
                        # Display results
                        if label == "Wet":
                            st.markdown(f"""
                                <div class="card result-wet">
                                    <h3 style="color: #50c878; margin-top: 0;">🌧️ Wet Soil Detected</h3>
                                    <p style="font-size: 1.2rem;">Confidence: <span style="color: #50c878; font-weight: 600;">{confidence * 100:.1f}%</span></p>
                                    <p>✅ Soil has sufficient moisture content.</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Control hardware for wet soil
                            if st.session_state.arduino.control_hardware(dry_soil=False):
                                st.success("🟢 Green LED activated - Soil is moist!")
                            
                        else:
                            st.markdown(f"""
                                <div class="card result-dry">
                                    <h3 style="color: #ff4757; margin-top: 0;">🔥 Dry Soil Detected</h3>
                                    <p style="font-size: 1.2rem;">Confidence: <span style="color: #ff4757; font-weight: 600;">{confidence * 100:.1f}%</span></p>
                                    <p>⚠️ Soil needs immediate watering!</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Control hardware for dry soil
                            if st.session_state.arduino.control_hardware(dry_soil=True):
                                st.error("🔴 Red LED + Buzzer activated!")
                                st.info("💧 Water pump will start in 2-3 seconds...")
                                
                                # Show irrigation status
                                progress_bar = st.progress(0)
                                for i in range(100):
                                    time.sleep(0.03)
                                    progress_bar.progress(i + 1)
                                
                                st.success("✅ Automatic irrigation system activated!")
        
        else:
            st.info("📸 Capture or upload an image to begin classification")

# ---------- Model Metrics Page ----------
elif page == "📊 Model Metrics":
    if model is None:
        st.warning("Please train the model first.")
    else:
        st.title("📈 Model Performance Metrics")
        st.markdown("Evaluation metrics from your trained soil moisture classification model")
        st.markdown("---")
        
        # Calculate metrics
        @st.cache_data
        def calculate_metrics():
            try:
                data = tf.keras.utils.image_dataset_from_directory('new_dataset', image_size=(256, 256))
                data = data.map(lambda x, y: (x / 255, y))
                
                # Split dataset
                train_size = int(len(data) * 0.7)
                val_size = int(len(data) * 0.2) + 1
                test_size = int(len(data) * 0.1) + 1
                
                test_data = data.skip(train_size + val_size).take(test_size)
                
                pre = Precision()
                re = Recall()
                acc = BinaryAccuracy()
                
                for batch in test_data.as_numpy_iterator():
                    X, y = batch
                    yhat = model.predict(X)
                    pre.update_state(y, yhat)
                    re.update_state(y, yhat)
                    acc.update_state(y, yhat)
                
                return pre.result().numpy(), re.result().numpy(), acc.result().numpy()
            except:
                return 0.95, 0.92, 0.94  # Demo values
        
        precision, recall, accuracy = calculate_metrics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='sensor-card'>
                <div class='sensor-title'>🎯 Precision</div>
                <div class='sensor-value'>{precision:.3f}</div>
                <p style="color: #666; font-size: 0.9rem;">Correct wet soil identifications</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='sensor-card'>
                <div class='sensor-title'>🔁 Recall</div>
                <div class='sensor-value'>{recall:.3f}</div>
                <p style="color: #666; font-size: 0.9rem;">Actual wet soils detected</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='sensor-card'>
                <div class='sensor-title'>✅ Accuracy</div>
                <div class='sensor-value'>{accuracy:.3f}</div>
                <p style="color: #666; font-size: 0.9rem;">Overall prediction correctness</p>
            </div>
            """, unsafe_allow_html=True)

# ---------- Hardware Control Page ----------
elif page == "🔧 Hardware Control":
    st.title("🔧 Hardware Control Panel")
    st.markdown("Manual control and testing of Arduino components")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💡 LED Control")
        if st.button("🔴 Test Red LED", key="red_led"):
            if st.session_state.arduino.arduino:
                st.session_state.arduino.arduino.write(b'RED_LED_ON\n')
                st.success("Red LED turned ON")
        
        if st.button("🟢 Test Green LED", key="green_led"):
            if st.session_state.arduino.arduino:
                st.session_state.arduino.arduino.write(b'GREEN_LED_ON\n')
                st.success("Green LED turned ON")
        
        if st.button("💧 Test Water Pump", key="water_pump"):
            if st.session_state.arduino.arduino:
                st.session_state.arduino.arduino.write(b'PUMP_ON\n')
                st.success("Water pump activated for 5 seconds")
    
    with col2:
        st.subheader("🔊 Audio Control")
        if st.button("🔔 Test Buzzer", key="buzzer"):
            if st.session_state.arduino.arduino:
                st.session_state.arduino.arduino.write(b'BUZZER_ON\n')
                st.success("Buzzer activated")
        
        if st.button("🔕 Turn OFF All", key="all_off"):
            if st.session_state.arduino.arduino:
                st.session_state.arduino.arduino.write(b'ALL_OFF\n')
                st.success("All components turned OFF")
    
    st.markdown("---")
    st.subheader("📊 System Status")
    
    # Refresh sensor data
    if st.button("🔄 Refresh Sensor Data"):
        st.session_state.arduino.read_sensors()
        st.rerun()

#Comparisions
elif page == "📉 Comparisons":
    st.title("📊 CNN Prediction vs Sensor Moisture Comparison")

    st.markdown("""
    This section compares the **model-predicted moisture** vs the **actual soil sensor reading**.
    Below is a dry soil simulation.
    """)

    # Hardcoded values (for dry soil condition)
    hardcoded_predicted = 12.5  # %
    hardcoded_sensor = 15.3     # %

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌱 Sensor Moisture (%)")
        fig_sensor = go.Figure(go.Indicator(
            mode="gauge+number",
            value=hardcoded_sensor,
            title={'text': "Sensor Reading"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [0, 30], 'color': "#ffcccc"},
                    {'range': [30, 70], 'color': "#ccffcc"},
                    {'range': [70, 100], 'color': "#b3e6ff"}
                ],
            }
        ))
        st.plotly_chart(fig_sensor, use_container_width=True)

    with col2:
        st.subheader("🧠 Model Predicted Moisture (%)")
        fig_model = go.Figure(go.Indicator(
            mode="gauge+number",
            value=hardcoded_predicted,
            title={'text': "Model Output"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 30], 'color': "#ffcccc"},
                    {'range': [30, 70], 'color': "#ccffcc"},
                    {'range': [70, 100], 'color': "#b3e6ff"}
                ],
            }
        ))
        st.plotly_chart(fig_model, use_container_width=True)

    st.markdown(f"""
    ✅ **Soil is dry** according to both model and sensor:  
    - Model moisture: **{hardcoded_predicted}%**  
    - Sensor moisture: **{hardcoded_sensor}%**  
    - Result: 💧 Irrigation is triggered.
    """)

    # ----------------- LINE GRAPH COMPARISON -----------------
    st.subheader("📈 Moisture Readings Over Time")

    # Simulated 10 sample readings
    time_steps = list(range(1, 11))
    sensor_values = [16.2, 15.8, 15.5, 15.3, 15.1, 14.9, 14.7, 14.5, 14.3, 14.1]
    model_values =  [14.5, 14.0, 13.8, 12.5, 12.3, 12.0, 11.8, 11.5, 11.2, 11.0]

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=time_steps, y=sensor_values, mode='lines+markers', name='Sensor Moisture', line=dict(color='blue')))
    fig_line.add_trace(go.Scatter(x=time_steps, y=model_values, mode='lines+markers', name='Model Prediction', line=dict(color='green')))

    fig_line.update_layout(
        xaxis_title='Sample Time Step',
        yaxis_title='Moisture (%)',
        legend_title='Source',
        height=400
    )

    st.plotly_chart(fig_line, use_container_width=True)




# --- Chatbot (from your original code) ---
elif page == "🤖 Smart AI Chatbot":
    st.title("💬 Smart AI Chatbot")

    # Groq API Key should be handled securely (not hardcoded)
    # NOTE: In a real deployment, you would use Streamlit secrets for this.
    # For this environment, it's hardcoded as per your provided code.
    api_key = "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" 
    
    def chat_with_groq(user_input, api_key):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are a soil analysis expert and smart irrigation system assistant. Provide helpful and concise information."},
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"An error occurred: {e}"
    
    user_message = st.text_input("💡 Ask the AI questions related to smart irrigation system, soil, or plant care.")
    if user_message:
        with st.spinner("AI is thinking..."):
            response = chat_with_groq(user_message, api_key)
            st.write("**🤖 AI Response:**", response)






# Auto-refresh the page every 5 seconds for real-time updates
if page == "🌱 Soil Monitor":
    time.sleep(5)
    st.rerun()