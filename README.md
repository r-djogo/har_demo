# HAR Demo

Code for demo accepted at the 2023 IEEE International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC 2023)

The demo makes use of two ESP-32 Wi-Fi enabled microcontrollers, one for transmitting and one for receiving Wi-Fi signals within different environments. By sampling the channel state information from the collected Wi-Fi signals, we show that a machine learning model can be trained on this data to perform real-time classification between the basic classes: "no gesture", "left-right", and "push-pull". This is possible because the wireless signals reflect from the individual performing the gestures, and by sampling them over time we can monitor the changes occuring in the environment.

The advantages of Wi-Fi-based sensing are:
- Wi-Fi is already abundant indoors
- Passive sensing (no wearables needed)
- Better user privacy than cameras
- Non-line-of-sight sensing ability
- Works with wide range of hardware (ESP-32, Arduino, Raspberry Pi, Routers)

Our hardware setup: ESP-32 microcontroller, cabable of extracting 52 active subcarriers at 2.4GHz range (20 MHz bandwidth), at a sampling rate of 100Hz.

Videos "demo_home.webm" and "demo_lab.webm" show the functional demo application in use.

Code based partially on: https://github.com/StevenMHernandez/ESP32-CSI-Tool
