//https://www.instructables.com/Face-Tracking-Device-Python-Arduino/
#include <Arduino.h>
#include <ESP32Servo.h>
#include <WiFi.h>
#include "wifi_password.h"

int x = 90;
int y = 90;

const int servoX_pin = 22;
const int servoY_pin = 23; //prob change later

unsigned long current_time = 0;
unsigned long previous_time = 0;

Servo servoX;
Servo servoY;

char input = 0x00; //check if this works

TaskHandle_t WifiTask;

WiFiServer server(80);
String header;

String wifi_message = "";

void WiFiCode(void * parameter);
void html(String);

void setup(){
    Serial.begin(115200);

    xTaskCreatePinnedToCore(
        WiFiCode,
        "WiFiTask",
        10000,
        NULL,
        0,
        NULL,
        0
    );

    servoX.attach(servoX_pin);
    servoY.attach(servoY_pin);

    servoX.write(x);
    servoY.write(y);

    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    
    delay(2000);

    Serial.println(WiFi.localIP());
    Serial.println(WiFi.status() == WL_CONNECTED);
}

void loop(){
    if(Serial.available()){ //checks if any data is in the serial buffer
        input = Serial.read();

        //this could probably be a switch
        if(input == 'U'){
            y += 1;  //updates the value of the angle
            wifi_message = "heard UP";
        }
        else if(input == 'D'){ 
            y -= 1;
            wifi_message = "heard DOWN";
        }
        if(input == 'L'){
            x -= 1;
            wifi_message = "heard LEFT";        
        } 
        else if(input == 'R'){
            x += 1;
            wifi_message = "heard RIGHT";
        }
        
        servoX.write(x);
        servoY.write(y);

        input = 0x00;           //clears the variable
    }
    else{
        wifi_message = "no data received";
    }
}

void WiFiCode(void * parameter){
    for(;;){
        // code for wifi task core 0

        // check and reconnect wifi connection
        unsigned long current_time = millis();
        if ((WiFi.status() != WL_CONNECTED) && (current_time - previous_time >= 30000)){
            //reconnect to wifi
            WiFi.disconnect();
            WiFi.reconnect();
            previous_time = current_time;
        }

        WiFiClient client = server.available();

        if (client){
            String current_line;
            while (client.connected()){
                if (client.available()){
                    char c = client.read();
                    header += c;
                    if (c == '\n'){
                        if (current_line.length() == 0){
                            // HTTP headers always start with a response code (e.g. HTTP/1.1 200 OK) https://randomnerdtutorials.com/esp32-web-server-arduino-ide/
                            // and a content-type so the client knows what's coming, then a blank line:
                            client.println("HTTP/1.1 200 OK");
                            client.println("Content-type:text/html");
                            client.println("Connection: close");
                            client.println();
                            
                            // Display the HTML web page
                            client.println("<!DOCTYPE html><html>");
                            client.println("<head><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">");
                            client.println("<link rel=\"icon\" href=\"data:,\">");
                            // CSS to style the on/off buttons 
                            // Feel free to change the background-color and font-size attributes to fit your preferences
                            client.println("<style>html { font-family: Helvetica; display: inline-block; margin: 0px auto; text-align: center;}");
                            client.println("text-decoration: none; font-size: 30px; margin: 2px; cursor: pointer;}");

                                
                            // Web Page Heading
                            client.println("<body><h1>ESP32 Web Server</h1>");
                            
                            // Display current state, and ON/OFF buttons for GPIO 26  
                            client.println("<p>Message: " + wifi_message + "</p>");
                            // If the output26State is off, it displays the ON button       

                            // The HTTP response ends with another blank line
                            client.println();
                            // Break out of the while loop
                            break;
                        }
                        else{
                            current_line = "";
                        }
                    }
                    else if (c != '\r'){
                        current_line += c;
                    }
                }
            }
            header = "";
            client.stop();
        }
    }
}

void html(String msg){

}
