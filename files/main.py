#!/usr/bin/env python

# https://github.com/ageitgey/face_recognition

# just taking note on how I installed stuff, using python37
# pkg install python37 py37-pip py37-numpy py37-scipy py37-opencv py37-tensorflow
# ln -s /usr/local/bin/python3.7-config /usr/local/bin/python-config
# ln -s /usr/local/bin/python3.7 /usr/local/bin/python
# pip install cvlib

import numpy
# cv2 vs opencv? opencv handles the import cv2
import cv2
import cvlib as cv
# import face_recognition

import logging

import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish

import multiprocessing as mp
import os
import pathlib
import fcntl

from PIL import Image, ImageDraw

import signal
import sys
import time
import yaml

class DoorBell():

    ### MQTT
    mqttQos = 0
    mqttRetained = False
    pidfile = '/var/run/doorbell.pid'
    lockfile = '/var/run/doorbell.lock'
    topic ='dev/test'

    def __init__(self):
        global logger
        logger = self.setupLogger()
        logger.info("init")

        global cfg
        with open(pathlib.Path(__file__).parent / "config.yaml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    def on_connect(self, client, userdata, flags, rc):
        logger.info("Connected with result code "+str(rc))
        client.subscribe(self.topic)

    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        payload = str(msg.payload.decode('ascii'))  # decode the binary string
        logger.info(msg.topic + " " + payload)
        process_trigger(payload)

    def process_trigger(payload):
        if payload == 'ON':
            logger.info('ON triggered')
            post_image()

    def connect_to_mqtt(self):
        client = mqtt.Client()
        db = self
        client.on_connect = db.on_connect    # call these on connect and on message
        client.on_message = db.on_message
        client.username_pw_set(username=cfg["mqtt"]["user"],password=cfg["mqtt"]["pass"])  # need this
        client.connect(cfg["mqtt"]["host"])
        #client.loop_forever()    #  don't get past this
        client.loop_start()    #  run in background and free up main thread

        return client

    def setupLogger(self):
        loglevel =  os.environ.get("LOGLEVEL", "INFO")
        # https://docs.python.org/2/howto/logging-cookbook.html
        logging.basicConfig(level=loglevel,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d-%m %H:%M:%S',
                    filename='/var/log/doorbell.log',
                    filemode='w')
        logger = logging.getLogger("doorbell")

        return logger

    def lockPidFile(self):
        db = self
        file_handle = open(db.lockfile, 'w+')
        logger.info("test to see if this is already running")

        try:
            fcntl.lockf(file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            # while the thing exits when it is already running, it does not seem to run the logger
            logger.error("process file lock could not be gained, process already running? Exiting")
            sys.exit(1)

        return file_handle

    def writePidFile(self):
        db = self
        pid = str(os.getpid())
        f = open(db.pidfile, 'w')
        f.write(pid)
        f.close() # this will close the lock

    def get_start(self):
        db = self

        # get a lock on the pidfile
        lock = db.lockPidFile()
        db.writePidFile()

        logger.info("starting")

        # set to Sydney TZ
        os.environ['TZ'] = 'Australia/Sydney'

        # sub stream, https://www.use-ip.co.uk/forum/threads/hikvision-rtsp-stream-urls.890/
        camera_url = 'rtsp://' + cfg["camera"]["user"] + ':' + cfg["camera"]["pass"] + '@' + cfg["camera"]["host"] + '/Streaming/Channels/1'
        video_capture = cv2.VideoCapture(camera_url)

        # Initialize some variables
        face_locations = []
        now = int(time.time())
        post = False
        mqtt_name = 'camera/porch'

        # shared variable
        # used for ensuring there is X amount of seconds between sending positive matches to HASS
        global last_face_detection
        last_face_detection = mp.Value('i', now)
        # used for ensuring there is X amount of seconds for updating the MQTT camera in HASS
        global porch_camera_image
        porch_camera_image = mp.Value('i', now)

        queue = mp.Queue()

        # handle signal shit as per the below:
        # https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

        pool = mp.Pool(4, db.worker, (porch_camera_image,post,queue,mqtt_name,last_face_detection))

        signal.signal(signal.SIGTERM, original_sigint_handler)

        try:

            while True:
                # seems to drop CPU by 15% (0.1 second sleep)
                # time.sleep(0.5) # seemed to run at .2 load
                # time.sleep(0.25)
                time.sleep(0.05)
                # new setup seems to take 0 CPU, removing, #.9 to 1.0 load

                # Grab a single frame of video
                ret, frame = video_capture.read()
                if not ret:
                    logger.error("failed to get a frame from the camera, try again")
                    video_capture.open(camera_url)
                    continue

                try:
                    queue.put(frame)
                except:
                    logger.error("queue full")

                # trying a pause as it seems to lag
                # 0.05 is 50 miliseconds, there is 1000ms in a second
                # time.sleep(0.1) #removed, I change the sampling and there are a lot more resources now

        except KeyboardInterrupt:

            logger.warning("Caught KeyboardInterrupt, terminating workers")

            # tell hass/mqtt
            client = db.connect_to_mqtt()
            payload = 'offline'
            client.publish('doorbell/status', payload, db.mqttQos, db.mqttRetained)

            pool.terminate()
            lock.close()
        else:

            logger.info("Normal termination")

            # tell hass/mqtt
            client = db.connect_to_mqtt()
            payload = 'offline'
            client.publish('doorbell/status', payload, db.mqttQos, db.mqttRetained)

            pool.close()
            lock.close()

        # Release handle to the webcam
        video_capture.release()
        lock.close()

    def worker(self,porch_camera_image,post,queue,mqtt_name,last_face_detection):

        db = self
        client = db.connect_to_mqtt()

        logger.info("processing")

        while True:

            # no longer needed but keeping for a reminder
            # process_name = mp.current_process().name

            the_time = str(time.strftime("%Y%m%d%H%M%S"))

            try:
                # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue.get
                frame = queue.get(True,60)
                logger.debug("I have a frame from the queue")

            except queue.empty:
                logger.error("my queue is empty, continue")
                continue

            except:
                logger.error("failed to get frame from the queue, continue")
                continue

            # this is then used as the final file that is written to disk and processed
            # I disabled the crop since the frame size changed when I went to the primary stream on the cameras and up'd their resolution
            # small_frame = frame[80:360,0:640], 
            small_frame = frame

            logger.debug("Find all the faces and face encodings in the current frame of video")
            faces, confidences = cv.detect_face(small_frame)
            face_count = len(faces)

            # are there any faces in the array?
            if face_count > 0:
                logger.info("I found {} face(s) in this photograph.".format(face_count))

                logger.info(faces)
                logger.info(confidences)

                # using a global variable to avoid every thread in the pool posting at the same time
                most_recent_face_detection = int(time.time())
                diff_face_detection = most_recent_face_detection - last_face_detection.value

                # this is some funky stuff for the shared variable
                with last_face_detection.get_lock():
                    last_face_detection.value = int(time.time())

                # this used to be once every 15 seconds, massively reducing to get more numbers
                # and see what confidences looks like, this variable is a different in seconds since the last post
                if diff_face_detection > 1:
                    post = True

                    # https://github.com/arunponnusamy/cvlib/blob/master/examples/face_detection.py
                    for face,conf in zip(faces,confidences):

                        confpercentage = (conf * 100)

                        if confpercentage > 60:

                            client.publish('doorbell/porch', 'ding,dong')
                            filename = '/opt/face_recognition/pictures/face/' + the_time + '.png'
                            mqtt_name = 'camera/porch_face'

                            (startX,startY) = face[0],face[1]
                            (endX,endY) = face[2],face[3]

                            # draw rectangle over face
                            cv2.rectangle(small_frame, (startX,startY), (endX,endY), (0,255,0), 2)

                            # https://github.com/arunponnusamy/cvlib/blob/master/examples/face_detection_webcam.py#L44
                            # write confidence percentage on top of face rectangle
                            text = "{:.2f}%".format(confpercentage)
                            Y = startY - 10 if startY - 10 > 10 else startY + 10
                            cv2.putText(small_frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                    post_frame = small_frame

            else:
                # using a global variable to avoid every thread in the pool posting at the same time
                later = int(time.time())
                diff = later - porch_camera_image.value
                filename = '/opt/face_recognition/pictures/image/' + the_time + '.png'
                if diff > 10:
                    logger.debug("it has been longer than 10 seconds since the last post to HASS")
                    mqtt_name = 'camera/porch'
                    post = True
                    post_frame = frame

                    # this is some funky stuff for the shared variable
                    with porch_camera_image.get_lock():
                        porch_camera_image.value = int(time.time())

            if post == True:
                # this writes to file and works
                cv2.imwrite(filename,post_frame)

                # prepare the frame/image for posting to HASS mqtt
                with open(filename, "rb") as imageFile:
                    myFile = imageFile.read()
                    data = bytearray(myFile)

                try:
                    # post to mosquitto
                    client.publish(mqtt_name, data, db.mqttQos, db.mqttRetained)
                    logger.debug("pushed to mqtt")
                except:
                    logger.error("push failed")

                # update the sensor
                payload = str(face_count)
                client.publish('doorbell/state', payload, db.mqttQos, db.mqttRetained)

                if face_count > 0:
                    confidences_average = sum(confidences) * 100 / len(confidences)
                else:
                    confidences_average = 0

                payload = '{ "date": ' + the_time + ', "face": ' + str(face_count) + ', "confidences": ' + str(confidences_average) + '  }'
                client.publish('doorbell/attributes', payload, db.mqttQos, db.mqttRetained)

                # tell hass we are still online
                payload = 'online'
                client.publish('doorbell/status', payload, db.mqttQos, db.mqttRetained)

            # reset variables back to defaults
            # reset the trigger to post
            post = False
            mqtt_name = 'camera/porch'
            post_frame = frame

            # os.remove(filename)

if __name__ == '__main__':

    # https://stackoverflow.com/questions/1603109/how-to-make-a-python-script-run-like-a-service-or-daemon-in-linux
    fpid = os.fork()
    if fpid != 0:
        sys.exit(0)

    db = DoorBell()
    db.get_start()

