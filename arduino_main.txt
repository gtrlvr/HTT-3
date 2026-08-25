#include <Servo.h>

Servo myServo;

// -------------------------
// PINS
// -------------------------
const int SERVO_PIN = 9;
const int MOS_PIN = 8;

// -------------------------
// SERVO SETTINGS
// -------------------------
const int REST_POSITION = 40;
const int FIRE_POSITION = 105;

const int FIRE_TIME = 130;  // milliseconds


void setup() {

  // -------------------------
  // SERIAL
  // -------------------------
  Serial.begin(9600);


  // -------------------------
  // MOSFET
  // -------------------------
  pinMode(MOS_PIN, OUTPUT);

  // MOSFET starts ON
  digitalWrite(MOS_PIN, HIGH);


  // -------------------------
  // SERVO
  // -------------------------
  myServo.attach(SERVO_PIN);

  myServo.write(REST_POSITION);

  delay(500);


  // -------------------------
  // STARTUP MESSAGE
  // -------------------------
  Serial.println("System ready.");
  Serial.println("F = activate servo");
  Serial.println("Q = MOSFET OFF");
  Serial.println("E = MOSFET ON");
}


void loop() {

  if (Serial.available() > 0) {

    char command = Serial.read();


    // -------------------------
    // SERVO
    // -------------------------
    if (command == 'F' || command == 'f') {

      Serial.println("Servo activated.");

      myServo.write(FIRE_POSITION);

      delay(FIRE_TIME);

      myServo.write(REST_POSITION);

      Serial.println("Servo returned to 40 degrees.");
    }


    // -------------------------
    // MOSFET OFF
    // -------------------------
    else if (command == 'Q' || command == 'q') {

      digitalWrite(MOS_PIN, LOW);

      Serial.println("MOSFET OFF.");
    }


    // -------------------------
    // MOSFET ON
    // -------------------------
    else if (command == 'E' || command == 'e') {

      digitalWrite(MOS_PIN, HIGH);

      Serial.println("MOSFET ON.");
    }
  }
}