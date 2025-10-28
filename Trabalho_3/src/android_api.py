from flask import Flask, request
import os
import subprocess

app = Flask(__name__)
AUDIO_PATH = "/data/data/com.termux/files/home/temp.wav"


@app.route("/play_audio", methods=["POST"])
def play_audio():
    file = request.files['audio']
    file.save(AUDIO_PATH)

    # Play VLC without blocking request
    subprocess.Popen(
        ["vlc", "--play-and-exit", AUDIO_PATH],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return "OK", 200


app.run(host="0.0.0.0", port=5000)
