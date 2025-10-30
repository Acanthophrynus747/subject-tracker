import serial
import time
from threading import Thread

msg1 = "abcdef"
msg2 = "efghij"

usb = serial.Serial("COM5", 115200)

def serialRead():
    try:
        while True:
            data = usb.readline().decode()
            print(f"heard {data}")
    except KeyboardInterrupt:
        exit()

def serialWrite():
    try:
        while True:
            usb.write(msg1.encode())
            time.sleep(1)
            usb.write(msg2.encode())
            time.sleep(1)
    except KeyboardInterrupt:
        exit()

if __name__ == "__main__":
    Thread(target = serialRead).start()
    Thread(target = serialWrite).start()
