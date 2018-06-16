#!/usr/bin/env python2.7
import time
import paho.mqtt.client as mqtt
import numpy as np
import json
import tensorflow as tf

class Config(object):
    """ Read Config file

    """
    def __init__(self, config_file):
        self._config_file = config_file
        self.read_config()

    def read_config(self):

        try:       
            with open(self.config_file) as f:
                config = json.load(f)
                mqtt_config = config.get("mqtt", {})
                misc_config = config.get("misc", {})

                self._username = mqtt_config.get("username")
                self._password = mqtt_config.get("password")
                self._host = mqtt_config.get("host")
                self._port = mqtt_config.get("port")
                self._topic = mqtt_config.get("topic")
                self._topic_sensor = mqtt_config.get("topic_sensor")

        except IOError as error:
            print("Error opening config file '%s'" % self.config_file, error)

    @property
    def config_file(self):
        return self._config_file

    @property
    def username(self):
        return self._username

    @property
    def password(self):
        return self._password

    @property
    def host(self):
        return self._host
    @property
    def port(self):
        return self._port
      
    @property
    def topic(self):
        return self._topic

    @property
    def topic_sensor(self):
        return self._topic_sensor  
      
class Client(object):

    def __init__(self, Config, name):
        """
            Args:
               
            Returns:
        """
        self.Config = Config
        self.name = name
        self.filename = "../models/" + "frozen.pb"        
        self.dics = {0:"running",1:"jumping",2:"sitting",3:"walking",4:"lying"}

    def start(self):
        self.connect()
        self.client.on_message = self.on_message
        self.client.on_connect = self.on_connect
        self.client.loop_forever()

    def connect(self):
        self.client = mqtt.Client(self.name)
        self.host = self.Config.host
        self.port = self.Config.port
        try:
            print("Connecting..")
            self.client.username_pw_set(self.Config._username, self.Config._password)
            self.client.connect(self.host, self.port)
            print("Success!")
        except IOError as error:
            print(error)

    def on_message(self, client, userdata, message):

        line = str(message.payload.decode("utf-8"))
        print line

    def on_connect(self, client, userdata, flags, rc):
        self.client.subscribe(self.Config.topic_sensor)
       
if __name__ == '__main__':
    config = Config("./config.json")
    client1 = Client(config, "henrykuo1")
    client1.start()
    



