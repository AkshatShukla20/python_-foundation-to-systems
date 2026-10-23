# This Project Is Named Jarvis and This is basic Chatbot project 
# modules used :- webbrowser, speech recoginition, pyttsx3, google


import speech_recognition as sr 
import webbrowser
import pyttsx3 

recognizer = sr.Recognizer()
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    speak("Initializing Jarvis....")

    while True:
        # Listen and wake up when Jarvis is called
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
            print("Recognizing....")

        try:
            command = recognizer.recognize_google(audio).lower()
            print(command)

            if "open youtube" in command:
                webbrowser.open("https://www.youtube.com")
                speak("Opening YouTube.")
            elif "open google" in command:
                webbrowser.open("https://www.google.com")
                speak("Opening Google.")
            elif "hello" in command:
                speak("Hello, sir. How can I help you?")
            elif "exit" in command or "stop" in command:
                speak("Goodbye, sir.")
                break
            else:
                speak(f"You said: {command}")

        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said.")
        except sr.RequestError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")
