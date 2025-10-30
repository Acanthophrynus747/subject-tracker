// https://forum.arduino.cc/t/pc-arduino-comms-using-python-updated/574496

#include <Arduino.h>

void setup(){
    Serial.begin(115200);
    Serial.println("starting");
}

void loop(){
    if (Serial.available()) {
        String msg = Serial.readString();
        Serial.print("<");
        Serial.print(msg);
        Serial.print(">");
    }
}