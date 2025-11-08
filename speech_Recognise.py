
import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Use the default microphone as the audio source
with sr.Microphone() as source:
    print("🎤 Listening... Please speak something.")
    # Adjusts for background noise for better accuracy
    recognizer.adjust_for_ambient_noise(source, duration=1)
    
    # Capture audio
    audio_data = recognizer.listen(source)
    print("✅ Audio captured. Recognizing...")

    try:
        # Convert speech to text using Google’s recognizer
        text = recognizer.recognize_google(audio_data)
        print("🗣 You said:", text)

        # Store recognized text in a string variable
        detected_audio = text
        print("✅ Stored in variable ->", detected_audio)

    except sr.UnknownValueError:
        print("❌ Sorry, could not understand the audio.")
    except sr.RequestError:
        print("⚠️ Could not request results. Check your internet connection.")