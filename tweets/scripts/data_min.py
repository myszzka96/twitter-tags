#! /usr/bin/env python

#intended for use if the robot is not avaiable. 
#Requires setting up a service key with Google Cloud

import rospy, hsrb_interface, rospkg, math, yaml, sys
import numpy as np
from std_msgs.msg import String
import re

import pyaudio
import speech_recognition as sr

from google.cloud import texttospeech

from rasa_nlu.model import Interpreter

from rospkg import RosPack

import pyowm
from pyowm import OWM
from pyowm.caches.lrucache import LRUCache

#to play the converted audio
import os #you might need to get an audio player for terminal 'apt-get install mpg123'

# Initialization -----------------------------------------------------------------------------------------

#print("Taking control of the robot's interface")

#robot = hsrb_interface.robot.Robot()

#tts = robot.get('default_tts')
#tts.language = tts.ENGLISH

#print("Took control")

#make sure to use below line to add the path to your env to get the API key
# export GOOGLE_APPLICATION_CREDENTIALS="path_to_\slang\Palpi-project.json"

m = sr.Microphone()
r = sr.Recognizer()

rp = RosPack()

interpreter = Interpreter.load(rp.get_path('slang') + '/output/RasaCovid/RasaCovid')

#Helper functions -----------------------------------------------------------------------------------------
# takes only the important data from the parsed object
def objectify(parsed):
    acc = {}
    acc['intent'] = parsed[u'intent'][u'name']
    for p in parsed[u'entities']:
        acc[p[u'entity']] = p[u'value']
    return acc 

def publish_weather(ws):
    # hsrb_mode triggers rosmaster node IP address = hsrb.local:11311 (have to be in RoboCanes-5G network)
    # sim_mode triggers rosmaster node IP address = your local IP address
    # Run roscore to init ros_master node, and in another terminal window run 'sim_mode' before this script.
    rospy.init_node('weather_publisher')
    pub = rospy.Publisher('weather', String, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    if not rospy.is_shutdown():
      weather_str = ws #% rospy.get_time()
      rospy.loginfo(weather_str)
      pub.publish(weather_str)
      rate.sleep() #it gives some time before speaking?


def speak_wait(s):
  nwords = len(s.split(" "))
  sperword = 0.4
  publish_weather(s)
  tts.say(s)
  rospy.sleep(nwords*sperword)

def speak_comp(s):
  nwords = len(s.split(" "))
  sperword = 0.4 
  # Instantiates a client
  client = texttospeech.TextToSpeechClient()
  # Set the text input to be synthesized
  synthesis_input = texttospeech.types.SynthesisInput(text=s)
  # Build the voice request, select the language code ("en-US") and the ssml with voice gender ("neutral")
  voice = texttospeech.types.VoiceSelectionParams(language_code='en-US', ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)
  # Select the type of audio file you want returned
  audio_config = texttospeech.types.AudioConfig(audio_encoding=texttospeech.enums.AudioEncoding.MP3)
  #Perform the text-to-speech request on the text input with the selected
  # voice parameters and audio file type
  publish_weather(s)
  response = client.synthesize_speech(synthesis_input, voice, audio_config)
  print('response.audio_content')
  print(type(response.audio_content))

  # The response's audio_content is binary.
  with open('output.mp3', 'wb') as out:
    # Write the response to the output file.
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')
  os.system('mpg123 output.mp3')
  os.remove('output.mp3') #removes the mp3 file


def recognize(prompt, timeout = 5):
  while timeout > 0:
    timeout -= 1
    speak_comp(prompt)
    try:
      with m as source: 
        audio = r.listen(source)
      print('source')
      print(type(source))

      w = r.recognize_google(audio).lower()
      recognized = True
      return w
    except (sr.UnknownValueError):
      speak_comp("Sorry I couldn't hear that.")
    except (sr.RequestError):
      speak_comp("Google speech recognition failure.")
      fail()

def offer_info(topic):
   if topic == 'tips' or topic == 'tip' or topic == 'advice' or topic == 'behaviour information':
      #TODO here we should search for latest tweet from CDC with tag tips
      s = "Ok, let me offer some advice from CDC."
      t = "More than 194,000 COVID19 cases identified yesterday. We must get the pandemic under control. Do your part to slow the spread and save lives. Wear a Mask. Cover mouth AND nose. Stay 6 feet from others. Wash your hands. Stay home if you can."
      speak_comp(s+t)
   elif topic == 'statistics':
      #TODO here we should search for latest tweet from CDC with tag stat
      s = "Here are the latest statistics and facts from CDC. "
      t = "Although the percentage of ER visits for childrens mental health was higher during COVID19, this percentage could've been affected by an overall decrease in ER visits. Monitoring, promoting coping and resilience, and expanding mental healthcare are key."
      speak_comp(s+t)
   elif topic == 'report' or topic == 'news':
      #TODO here we should search for latest tweet from CDC with tag report
      s = "The latest CDC COVIDView report shows all the indicators used to track COVID19 activity have been increasing nationally in the United States since the beginning of October."
      speak_comp(s)
   elif topic == 'vaccination' or topic == 'vaccine' or topic == 'vaccines' or topic == 'vaccinations':
      #TODO here we should search for latest tweet from CDC with tag vaccine
      s = "While there is not yet an authorized or approved vaccine to prevent COVID19 in the U.S., CDC is focused on vaccine planning and working closely with health departments and partners to prepare."
      speak_comp(s)

def check_mood(topic):
   s = "Currently, public opinion about covid is mostly neutral. On Twitter posts there are more negative posts on the topic than positive."
   speak_comp(s)


if __name__ == "__main__":

  tlk = "Adjusting for ambient noise, please be quiet"
  speak_comp(tlk)

  with m as source: r.adjust_for_ambient_noise(source)

  rospy.sleep(1)
  check = True
  count = 0
  while check:
    if count == 0:
      s = recognize("Hi, I am Palpi. Your personal assistant in times of covid19. I can help with covid tips, reports, statistics or tell you news about possible vaccine. What would you line to know?").lower()
    else:
      s = recognize("Would you like to know more about tips, reports, public opinion or statistics?").lower()
    print("heard: " + s)
    print(type(s))
    parsed = interpreter.parse(s)
    obj = objectify(parsed)
    
    print(obj)

    intent = obj['intent']
    print(intent)

    if intent == 'covid':
      if obj['topic'] == 'mood' or obj['topic'] == 'opinion':
        item = obj['topic']
        print("This is topic: " + item)
        check_mood(item)
        count=count+1
      else:
        item = obj['topic']
        print("This is topic: " + item)
        offer_info(item)
        count=count+1


    elif intent == 'negative':
      speak_comp('Ok, nice talking to you. Bye!')
      break
        
    else:
      speak_comp('Sorry, what did you ask?')
      count=count+1


