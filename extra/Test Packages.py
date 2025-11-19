def test_packages():
    
    packages = {
        "NumPy": "numpy",
        "Pandas": "pandas",
        "Scikit-learn": "sklearn",
        "OpenCV": "cv2",
        "MediaPipe": "mediapipe",
        "PyAudio": "pyaudio",
        "SpeechRecognition": "speech_recognition",
        "TensorFlow": "tensorflow",
    }

    for name, module in packages.items():
        try:
            pkg = __import__(module)
            # Special handling for protobuf module
            if module == "google.protobuf":
                version = pkg.__version__
            else:
                version = pkg.__version__
            print(f" {name}: {version}")
        except ImportError:
            print(f" {name} not installed")
        except AttributeError:
            print(f" {name} installed but version not found")

# Call the function
test_packages()