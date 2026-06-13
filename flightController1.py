""" Flight Controller Script for Raspberry Pi.
Maps keyboard keys to control 4 motors in real-time via the terminal.
Press 'Q' to quit and safely stop all motors.
"""

import sys
import tty
import termios
import time
import RPi.GPIO as GPIO
from dataclasses import dataclass

# --- 1. KEYBOARD CONFIGURATION ---
def get_key():
    """Reads a single keypress from the terminal immediately without hitting Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch.lower()

# --- 2. MOTOR HARDWARE SETUP ---
@dataclass
class MotorPins:
    input1: int
    input2: int
    enable: int

PWM_FREQUENCY = 1000

class Motor:
    def __init__(self, pins: MotorPins) -> None:
        self._pins = pins
        GPIO.setup(self._pins.input1, GPIO.OUT)
        GPIO.setup(self._pins.input2, GPIO.OUT)
        GPIO.setup(self._pins.enable, GPIO.OUT)
        
        self._pwm = GPIO.PWM(self._pins.enable, PWM_FREQUENCY)
        self._pwm.start(0)
        self.stop()

    def setSpeed(self, speed: float) -> None:
        speed = max(-1.0, min(1.0, speed))
        dutyCycle = abs(speed) * 100.0

        if speed > 0:
            GPIO.output(self._pins.input1, GPIO.HIGH)
            GPIO.output(self._pins.input2, GPIO.LOW)
            self._pwm.ChangeDutyCycle(dutyCycle)
        elif speed < 0:
            GPIO.output(self._pins.input1, GPIO.LOW)
            GPIO.output(self._pins.input2, GPIO.HIGH)
            self._pwm.ChangeDutyCycle(dutyCycle)
        else:
            self.stop()

    def stop(self) -> None:
        GPIO.output(self._pins.input1, GPIO.LOW)
        GPIO.output(self._pins.input2, GPIO.LOW)
        self._pwm.ChangeDutyCycle(0)

    def close(self) -> None:
        self.stop()
        self._pwm.stop()

# Wire mapping from your PCB configurations
MOTORS_PINS = [
    MotorPins(input1=21, input2=13, enable=13), # Motor 1 (Left 1)
    MotorPins(input1=20, input2=6,  enable=12), # Motor 2 (Left 2)
    MotorPins(input1=16, input2=5,  enable=22), # Motor 3 (Right 1)
    MotorPins(input1=19, input2=26, enable=25), # Motor 4 (Right 2)
]

# --- 3. MAIN CONTROLLER LOOP ---
def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    print("Initializing Flight Controller...")
    
    # Instantiate all 4 motors dynamically into a list
    motors = [Motor(pins) for pins in MOTORS_PINS]
    
    print("\n==========================================")
    print("  FLIGHT CONTROLLER ACTIVE")
    print("  Controls:")
    print("    W -> Motors 1 & 2 Spin CW (0.3 Speed)")
    print("    S -> Motors 1 & 2 Spin CCW (-0.3 Speed)")
    print("    A -> Motor 1 CW, Motor 2 CCW")
    print("    D -> Motor 1 CCW, Motor 2 CW")
    print("    E -> Motors 1 & 2 STOP")
    print("------------------------------------------")
    print("    U -> Motors 3 & 4 Spin CW (0.5 Speed)")
    print("    I -> Motors 3 & 4 STOP")
    print("------------------------------------------")
    print("    Q -> EMERGENCY STOP & QUIT")
    print("==========================================")

    try:
        while True:
            key = get_key()
            
            if key == 'w':
                print("-> Command: W (M1 & M2 CW)")
                motors[0].setSpeed(0.3)
                motors[1].setSpeed(0.3)
                
            elif key == 's':
                print("-> Command: S (M1 & M2 CCW)")
                motors[0].setSpeed(-0.3)
                motors[1].setSpeed(-0.3)
                
            elif key == 'a':
                print("-> Command: A (M1 CW, M2 CCW)")
                motors[0].setSpeed(0.5)   # M1 Clockwise
                motors[1].setSpeed(-0.5)  # M2 Counter-Clockwise
                
            elif key == 'd':
                print("-> Command: D (M1 CCW, M2 CW)")
                motors[0].setSpeed(-0.5)  # M1 Counter-Clockwise
                motors[1].setSpeed(0.5)   # M2 Clockwise
                
            elif key == 'e':
                print("-> Command: E (M1 & M2 STOP)")
                motors[0].stop()
                motors[1].stop()

            elif key == 'u':
                print("-> Command: U (M3 & M4 CW)")
                motors[2].setSpeed(0.5)
                motors[3].setSpeed(0.5)

            elif key == 'i':
                print("-> Command: I (M3 & M4 STOP)")
                motors[2].stop()
                motors[3].stop()
                
            elif key == 'q':
                print("\nExiting flight controller...")
                break # Breaks out of the loop to run cleanup
                
    except KeyboardInterrupt:
        print("\nForced termination.")
        
    finally:
        # Crucial safety mechanism: kill all active PWM signals and power channels
        print("Shutting down flight software and killing motor power...")
        for motor in motors:
            motor.close()
        GPIO.cleanup()
        print("System Safe. Goodbye.")

if __name__ == "__main__":
    main()
