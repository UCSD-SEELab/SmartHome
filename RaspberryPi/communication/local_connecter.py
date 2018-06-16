#!/usr/bin/env python2.7
import time
import paho.mqtt.publish as publish
import numpy as np

if __name__ == '__main__':

    while True:
        f = open("../data/subject1.txt", "r")
        for line_ in f:
            line_ = line_.strip().split(' ')
            line_ = [np.float32(i) for i in line_]
            line = " ".join(str(x) for x in line_
            publish.single("house.hand", line, client_id="henrykuo2", hostname= "192.168.10.6", port=61613, auth={'username': "admin", 'password': "IBMProject$"})
            time.sleep(2)
        f.close()



