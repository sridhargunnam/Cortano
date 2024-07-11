# import evdev

# devices = [evdev.InputDevice(fn) for fn in evdev.list_devices()]
# for device in devices:
#     print(f"Device name: {device.name}, Path: {device.fn}")

import evdev
from evdev import InputDevice, categorize, ecodes

# Find the controller device
devices = [evdev.InputDevice(fn) for fn in evdev.list_devices()]
controller = None
for device in devices:
    if 'Wireless Controller' in device.name:  # Adjust this string if necessary to match your controller's name
        controller = device
        break
    else:
        print(device.name)

if controller is None:
    print("No Sony controller found.")
    exit()

print(f"Using device: {controller.name} ({controller.fn})")

# Listen for events
for event in controller.read_loop():
    if event.type == ecodes.EV_KEY:
        key_event = categorize(event)
        if key_event.keystate == key_event.key_down:
            print(f"Button {key_event.keycode} pressed.")
        elif key_event.keystate == key_event.key_up:
            print(f"Button {key_event.keycode} released.")
    elif event.type == ecodes.EV_ABS:
        abs_event = categorize(event)
        if abs_event.event.value > 160 or abs_event.event.value < 100:
            print(f"Control {abs_event.event.code} moved to position {abs_event.event.value}.")

#  Accordingly change the code in vex_serial.py to use wireless controller mapping's instead of keyboard. Write new functions to perform the robo'ts rotation, and drive if needed and use them for wireless controller based control.  
#  left side joy  control has event codes 0,1   sideways(0), up-down(1)
#  right side joy control has event codes 3,4   sideways(3), up-down(4) 
#  In the sideways control leftmost sides position's event value is 0, rightmost sides position's event value is 255. So the entire range is [0-255]. If nothing is pressed a neutral value is 128 is read as event value. 
#  In the up-down control bottom most(down) position's event value is 0, top most(up) positions event value is 255. So the entire range is [0-255]. If nothing is pressed a neutral value is 128 is read as event value. 
#  i.e value of 128 implies nothing is being manupulated. 
#  If the sideways(event code 0) is great than 128 rotate clockwise. If it is less than 128 rotate counterclockwise. The value of motors rotation in control.rotateRobot or the new rotate function should be controlled by the differnce in event value for the sideways event.
#  Similarly use control.drive function should get inputs from event code 1 which is used to control the drive. If the values are greater than 128 move in the forward in direction with 255 - value, as the motor drive. If the value is less than 128 it implies the robot needs to move backwards with the value equal to the event value - 128.  
#  event code 3 is used for claw open and close. event values greater then 128 is for open, values less then 128 implies close.
#  event code 4 is used for arm moveing up or down. Values greate than 128 imples moving up. less then 128 implies moving down. 
#  If button keycode values of BTN_TR2, or BTN_TL2 are either pressed or released then stop the robot drive for all of the motors. 
#  BTN_TR2 pressed/released - stop drive
#  BTN_TL2 pressed/released - stop drive