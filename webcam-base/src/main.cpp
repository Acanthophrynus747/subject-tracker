//https://www.instructables.com/Face-Tracking-Device-Python-Arduino/
#include <Arduino.h>
#include <ESP32Servo.h>

int x = 90;
int y = 90;

const int servoX_pin = 9;
const int servoY_pin = 10; //prob change later

Servo servoX;
Servo servoY;

char input = 0x00; //check if this works

void setup(){
    Serial.begin(115200);

    servoX.attach(servoX_pin);
    servoY.attach(servoY_pin);

    servoX.write(x);
    servoY.write(y);
    
    delay(1000);
}

void loop(){
    if(Serial.available()){ //checks if any data is in the serial buffer
        input = Serial.read();

        //this could probably be a switch
        if(input == 'U'){
            servoY.write(y+1);    //adjusts the servo angle according to the input
            y += 1;               //updates the value of the angle
        }
        else if(input == 'D'){ 
            servoY.write(y-1);
            y -= 1;
        }
        else{
            servoY.write(y);
        } 
        if(input == 'L'){
            servoX.write(x-1);
            x -= 1;
        } 
        else if(input == 'R'){
            servoX.write(x+1);
            x += 1;
        }
        else{
            servoX.write(x);
        }
        input = 0x00;           //clears the variable
    }
}