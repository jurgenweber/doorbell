# My DoorBell

script to stream my video camera and see if faces turn up in front of it
it will then send a payload to moquitto mqtt.

Links:
https://www.hackster.io/robin-cole/pi-camera-doorbell-with-notifications-408d3d
https://github.com/ageitgey/face_recognition/blob/master/examples/blur_faces_on_webcam.py
https://github.com/arunponnusamy/cvlib/tree/master/examples
https://face-recognition.readthedocs.io/en/latest/face_recognition.html#module-face_recognition.api

# configuration
There is a configuration file needed to tell the script where your camera and mqtt are.

```yaml
camera:
  host: 192.168.1.1
  user: user
  pass: pass
mqtt:
  host: 192.168.1.2
  user: user
  pass: pass
```

How to install in systemd:
```
cp doorbell.service to /etc/systemd/system
systemctl enable doorbell.service
service doorbell start
service doorbell status
```
I have many incarnations of this script and I have used a few python face_reconigtion library's but continue to struggle with accuracy.
I am using cvlib now, but it seems to pick up a lot of cars and miss a lot of faces. With the ageitgey/face_recognition project I seemed
to pick up a lot of shadows.
