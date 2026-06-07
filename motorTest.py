""" This is a motorTest script that will make all motors connected
to the SMCS Red PCB spin 3 seconds CW, and 3 seconds CCW at a controlled speed
"""

import RPi.GPIO as GPIO # These are General Purpose Input/Output ports on the RPI
import time # Necessary to control duration of spin
from dataclasses import dataclass # Auto-generates __init__ methods based on var names

# To control direction there are two inputs, 1 and 2 (H-Bridge)
# To control whether the motor spins or not, there is an enable input
@dataclass
class MotorPins:
    input1: int
    input2: int
    enable: int
  
PWM_FREQUENCY = 1000

class Motor:
    def __init__(self, pins: MotorPins) -> None:
        self._pins = pins # Assigns THIS motor its specific pin values
        GPIO.setup(self._pins.input1, GPIO.OUT) # Pin1 is an output pin
        GPIO.setup(self._pins.input2, GPIO.OUT) # Pin2 is an output pin
        GPIO.setup(self._pins.enable, GPIO.OUT) # The enable pin is an output pin
        
        # PWM is assigned to the enable pin to control the speed
        self._pwm = GPIO.PWM(self._pins.enable, PWM_FREQUENCY) 
        self._pwm.start(0) # Starts at 0, so no jitter when it starts
        self.stop() # Pre-emptive function that cuts off power safely
#PWM Summary!:
"""
In the old days, you would power stuff with a constant flow of 
current and voltage. However, that would be extremely power ineffeicient.
Instead you can rapidly toggle between two states. ON and OFF, where there is either no current flow, or no resistance
Speed is the duty cycle bc it determines the avg V received to power motors, so if 50 on / Overall -> 50% duty cycle

"""
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

# Creates motor pin configurations
MOTORS_PINS = [
    MotorPins(input1=21, input2=13, enable=13), # Left Motor 1
    MotorPins(input1=20, input2=6,  enable=21), # Left Motor 2
    MotorPins(input1=16, input2=5,  enable=22), # Right Motor 1
    MotorPins(input1=19, input2=26, enable=25), # Right Motor 2
]

def main():
    GPIO.setmode(GPIO.BCM) # How to interpret pin #'s
    GPIO.setwarnings(False) # Ignore terminal warnings

    print("Starting Motor Test...")
    try:
        for i, pins in enumerate(MOTORS_PINS, 1):
            print(f"\n--- Testing Motor {i} ---")
            print(f"Pins: IN1={pins.input1}, IN2={pins.input2}, EN={pins.enable}")

            motor = Motor(pins)

            print(f"  Motor {i}: Forward (3s)")
            motor.setSpeed(0.5)
            time.sleep(3) # Changed from 1 to 3 to match your description docstring!
            motor.setSpeed(0)

            print(f"  Motor {i}: Backward (3s)")
            motor.setSpeed(-0.5)
            time.sleep(3) # Changed from 1 to 3 to match your description docstring!
            motor.setSpeed(0)

            print(f"  Motor {i}: Stopping")
            motor.close()
            time.sleep(1) # Brief pause between motors

        print("\nAll motor tests complete.")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        GPIO.cleanup()
        print("GPIO cleaned up.")

if __name__ == "__main__":
    main()
