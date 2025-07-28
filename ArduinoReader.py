import serial

def get_sensor_data():
    try:
        arduino = serial.Serial('COM5', 9600, timeout=2)  # Change COM port if needed
        arduino.flush()
        if arduino.in_waiting > 0:
            line = arduino.readline().decode('utf-8').strip()
            values = line.split(',')
            if len(values) == 4:
                temp = float(values[0])
                humidity = float(values[1])
                soil_raw = int(values[2])
                tds = int(values[3])
                soil_percent = (4095 - soil_raw) / 4095 * 100
                return {
                    'temp': temp,
                    'humidity': humidity,
                    'soil_raw': soil_raw,
                    'soil_percent': round(soil_percent, 2),
                    'tds': tds
                }
    except Exception as e:
        print("Error reading Arduino:", e)
    return None
