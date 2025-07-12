import pyttsx3
import time

# Initialize the engine
engine = pyttsx3.init()

# Set speaking rate (optional)
engine.setProperty('rate', 150)

# Set voice (optional)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # [0] for male, [1] for female (platform-dependent)

# Text to speak
#text = "ooe Anuj chup chap aapna kaam kr"

def Speak(data):
    engine.say(data)
    engine.runAndWait()
    time.sleep(1)



