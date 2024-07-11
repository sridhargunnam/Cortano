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
        print(f"Control {abs_event.event.code} moved to position {abs_event.event.value}.")

# import evdev

# devices = [evdev.InputDevice(fn) for fn in evdev.list_devices()]
# for device in devices:
#     print(f"Device name: {device.name}, Path: {device.fn}")
