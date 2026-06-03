"""The goal of this script is to run one motor at a time via ssh on a raspberry pi.
This code will be copy pasted into a terminal script called motorTest.py for quick testing.

- Rotate 3 Seconds one direction
- Rotate 3 Seconds the other direction
"""

import RPi.GPIO as GPIO #Input Output ports of RPI 0
import time             #Time based commands
import dataclasses import dataclass #It auto builds the init() method you need, cleaning up code

@dataclass(frozen=True)
class MotorPins:
  in1: int
  in2: int
  enable: int

class Motor:
  def __init__(self, pins: MotorPins) -> None:
    GPIO.setup(self._pins.in1, GPIO.OUT)
    GPIO.setup(self._pins.in2, GPIO.OUT)
    GPIO.setup(self._pins.enable, GPIO.OUT)
    self._pwm = GPIO.PWM(self._pins.enable, PWM_FREQUENCY)
    self._pwm.start(0)
    self.stop()

  def set_speed(self, speed: float) -> None:
    speed = max(-1.0, min (1.0, speed))
    dutyCycle = abs(speed) * 100.0

    if speed > 0:
              GPIO.output(self._pins.in1, GPIO.HIGH)
              GPIO.output(self._pins.in2, GPIO.LOW)
              self._pwm.ChangeDutyCycle(duty_cycle)
          elif speed < 0:
              GPIO.output(self._pins.in1, GPIO.LOW)
              GPIO.output(self._pins.in2, GPIO.HIGH)
              self._pwm.ChangeDutyCycle(duty_cycle)
          else:
              self.stop()

    def stop(self) -> None:
        GPIO.output(self._pins.in1, GPIO.LOW)
        GPIO.output(self._pins.in2, GPIO.LOW)
        self._pwm.ChangeDutyCycle(0)

    def close(self) -> None:
        self.stop()
        self._pwm.stop()
MOTOR_PINS = [
  MotorPins(in1 = 21, in2 = 13, enable=13)
]

def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    print("Starting Motor Test...")
    print(f"\n--- Testing Motor {1} ---")
            print(f"Pins: IN1={pins.in1}, IN2={pins.in2}, EN={pins.enable}")

            motor = Motor(pins)

            print(f"  Motor {1}: Forward (3s)")
            motor.set_speed(0.5)
            time.sleep(1)
            motor.set_speed(0)

            print(f"  Motor {1}: Backward (3s)")
            motor.set_speed(-0.5)
            time.sleep(1)
            motor.set_speed(0)

            print(f"  Motor {1}: Stopping")
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
