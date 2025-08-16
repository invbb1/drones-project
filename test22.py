
from djitellopy import Tello
import time

tello = Tello()
tello.connect()

print(f"Battery: {tello.get_battery()}%")


tello.takeoff()
time.sleep(3)

tello.move_up(50)
time.sleep(2)
tello.rotate_clockwise(90)
time.sleep(2)


tello.land()