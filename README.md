#
# script to stream video camera and see if faces turn up in front of it
# then notify moquitto mqtt to do 'stuff' as a doorbell
#

Links:
https://www.hackster.io/robin-cole/pi-camera-doorbell-with-notifications-408d3d
https://github.com/ageitgey/face_recognition/blob/master/examples/blur_faces_on_webcam.py
https://face-recognition.readthedocs.io/en/latest/face_recognition.html#module-face_recognition.api

How to install in systemd:
```
cp doorbell.service to /etc/systemd/system
systemctl enable doorbell.service
service doorbell start
service doorbell status
```
I have many incarnations of this script and I have used a few python face_reconigtion library's but continue to struggle with accuracy.
