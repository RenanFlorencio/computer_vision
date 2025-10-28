import requests

if __name__ == "__main__":
    with open("src/tts/audios/sound.wav", 'rb') as f:
        requests.post(f"http://100.118.7.80:5000/play_audio",
                      files={"audio": f})
