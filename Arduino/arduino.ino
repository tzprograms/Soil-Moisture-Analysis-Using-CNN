#include <DHT.h>

// Pin definitions
#define DHTPIN 44          // DHT11 data pin (digital)
#define DHTTYPE DHT11

#define SOIL_PIN A0       // Analog pin for soil moisture
#define TDS_PIN A2        // Changed from A1 to A2 for compatibility

#define RELAY_PIN 7       // Controls water pump (Active-LOW Relay)
#define BUZZER_PIN 10     // Buzzer
#define RED_LED 8         // Dry indicator
#define GREEN_LED 9       // Moist indicator

// Initialize DHT sensor
DHT dht(DHTPIN, DHTTYPE);

// Variables for sensor readings
float temperature = 0;
float humidity = 0;
int soilMoisture = 0;
int tdsValue = 0;

// Communication variables
String command = "";
unsigned long lastSensorRead = 0;
const unsigned long sensorInterval = 2000; // Read sensors every 2 seconds

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize DHT sensor
  dht.begin();
  
  // Initialize pins
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  
  // Set initial states (all OFF)
  digitalWrite(RELAY_PIN, HIGH);  // Active-LOW relay, so HIGH = OFF
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(RED_LED, LOW);
  digitalWrite(GREEN_LED, LOW);
  
  Serial.println("Arduino Soil Monitor System Ready");
  Serial.println("Commands: READ_SENSORS, DRY_SOIL, WET_SOIL, CAMERA_CAPTURE");
}

void loop() {
  // Read sensors periodically
  if (millis() - lastSensorRead >= sensorInterval) {
    readSensors();
    lastSensorRead = millis();
  }
  
  // Check for serial commands
  if (Serial.available() > 0) {
    command = Serial.readStringUntil('\n');
    command.trim();
    processCommand(command);
  }
  
  delay(100); // Small delay to prevent overwhelming the serial port
}

void readSensors() {
  // Read DHT11 sensor
  temperature = dht.readTemperature();
  humidity = dht.readHumidity();
  
  // Read soil moisture (0-1023, where 0 is very wet, 1023 is very dry)
  int raw_value = analogRead(SOIL_PIN);
  int soilMoisture = map(raw_value, 0, 4095, 0, 100); // Convert raw to percentage
soilMoisture = 100 - soilMoisture; // Invert the percentage
  
  // Read TDS sensor (0-1023, convert to approximate TDS value)
  int tdsAnalog = analogRead(TDS_PIN);
  tdsValue = map(tdsAnalog, 0, 1023, 0, 1000); // Convert to TDS range 0-1000 ppm
  
  // Check for sensor errors
  if (isnan(temperature) || isnan(humidity)) {
    temperature = 0;
    humidity = 0;
  }
}

void processCommand(String cmd) {
  if (cmd == "READ_SENSORS") {
    sendSensorData();
  }
  else if (cmd == "DRY_SOIL" || cmd == "DRY_SOIL_ONCE") {  // Modified this line
    handleDrySoilOnce();  // Changed from handleDrySoil()
  }
  else if (cmd == "WET_SOIL") {
    handleWetSoil();
  }
  else if (cmd == "CAMERA_CAPTURE") {
    simulateCameraCapture();
  }
  else if (cmd == "RED_LED_ON") {
    testRedLED();
  }
  else if (cmd == "GREEN_LED_ON") {
    testGreenLED();
  }
  else if (cmd == "BUZZER_ON") {
    testBuzzer();
  }
  else if (cmd == "PUMP_ON") {
    testWaterPump();
  }
  else if (cmd == "ALL_OFF") {
    turnOffAll();
  }
  else {
    Serial.println("Unknown command: " + cmd);
  }
}

void sendSensorData() {
  // Send sensor data in format: TEMP:25.5,HUM:60.2,SOIL:450,TDS:78
  Serial.print("TEMP:");
  Serial.print(temperature, 1);
  Serial.print(",HUM:");
  Serial.print(humidity, 1);
  Serial.print(",SOIL:");
  Serial.print(soilMoisture);
  Serial.print(",TDS:");
  Serial.println(tdsValue);
}

void handleDrySoil() {
  Serial.println("Dry soil detected - Activating irrigation system");
  
  // Turn on red LED
  digitalWrite(RED_LED, HIGH);
  digitalWrite(GREEN_LED, LOW);
  
  // Activate buzzer for 2-3 seconds (only once)
  for (int i = 0; i < 6; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(250);
    digitalWrite(BUZZER_PIN, LOW);
    delay(250);
  }
  
  // Wait 2 seconds then activate water pump
  delay(2000);
  
  // Turn on water pump (Active-LOW relay)
  digitalWrite(RELAY_PIN, LOW);
  Serial.println("Water pump activated");
  
  // Keep pump running for 10 seconds (adjust as needed)
  delay(5000);
  
  // Turn off water pump and red LED
  digitalWrite(RELAY_PIN, HIGH);
  digitalWrite(RED_LED, LOW);
  Serial.println("Water pump deactivated");
}

void handleDrySoilOnce() {
  handleDrySoil();  // Just call the same function
}

void handleWetSoil() {
  Serial.println("Wet soil detected - Soil is healthy");
  
  // Turn on green LED only
  digitalWrite(GREEN_LED, HIGH);
  digitalWrite(RED_LED, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(RELAY_PIN, HIGH); // Make sure pump is off
  
  // Keep green LED on for 5 seconds
  delay(5000);
  digitalWrite(GREEN_LED, LOW);
}

void simulateCameraCapture() {
  Serial.println("Simulating camera capture");
  
  // Blink both LEDs to simulate camera activity
  for (int i = 0; i < 3; i++) {
    digitalWrite(RED_LED, HIGH);
    digitalWrite(GREEN_LED, HIGH);
    delay(200);
    digitalWrite(RED_LED, LOW);
    digitalWrite(GREEN_LED, LOW);
    delay(200);
  }
  Serial.println("Camera capture simulation complete");
}

void testRedLED() {
  Serial.println("Testing Red LED");
  digitalWrite(RED_LED, HIGH);
  delay(2000);
  digitalWrite(RED_LED, LOW);
  Serial.println("Red LED test complete");
}

void testGreenLED() {
  Serial.println("Testing Green LED");
  digitalWrite(GREEN_LED, HIGH);
  delay(2000);
  digitalWrite(GREEN_LED, LOW);
  Serial.println("Green LED test complete");
}

void testBuzzer() {
  Serial.println("Testing Buzzer");
  for (int i = 0; i < 3; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(500);
    digitalWrite(BUZZER_PIN, LOW);
    delay(500);
  }
  Serial.println("Buzzer test complete");
}

void testWaterPump() {
  Serial.println("Testing Water Pump");
  digitalWrite(RELAY_PIN, LOW);  // Turn ON (Active-LOW)
  delay(5000);  // Run for 5 seconds
  digitalWrite(RELAY_PIN, HIGH); // Turn OFF
  Serial.println("Water pump test complete");
}

void turnOffAll() {
  Serial.println("Turning off all components");
  digitalWrite(RED_LED, LOW);
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(RELAY_PIN, HIGH);  // Turn OFF relay (Active-LOW)
  Serial.println("All components turned off");
}