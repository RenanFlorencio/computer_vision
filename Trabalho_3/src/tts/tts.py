import wave
from piper import PiperVoice, SynthesisConfig
import os


def synthesize_speech(text, filename):

    syn_config = None  # default synthesis configuration
    # syn_config = SynthesisConfig(
    #     volume=0.5,  # half as loud
    #     length_scale=2.0,  # twice as slow
    #     noise_scale=1.0,  # more audio variation
    #     noise_w_scale=1.0,  # more speaking variation
    #     normalize_audio=False,  # use raw audio from voice
    # )

    here = os.path.dirname(__file__)
    model_path = os.path.join(here, "pt_BR-cadu-medium.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Voice model not found: {model_path}")

    voice = PiperVoice.load(model_path, use_cuda=True)

    with wave.open(os.path.join(here, f"{filename}"), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=syn_config)


if __name__ == "__main__":
    synthesize_speech("VAI CORINTIAAAAAAAAAAAMS.", "test.wav")
    print("Síntese concluída. Arquivo salvo como 'test.wav'.")
