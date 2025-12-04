import os
import threading

import simpleaudio as sa   # pip install simpleaudio
import pyttsx3


class AudioService:
    def __init__(self, beep_path: str = None):
        # Resolve default beep path relative to this file
        if beep_path is None:
            base_dir = os.path.dirname(__file__)
            beep_path = os.path.join(base_dir, "checkout_sound.wav")

        # Beep sound
        self.wave_obj = None
        if os.path.exists(beep_path):
            try:
                self.wave_obj = sa.WaveObject.from_wave_file(beep_path)
            except Exception as e:
                print(f"[AudioService] Could not load beep sound: {e}")
        else:
            print(f"[AudioService] Beep file not found at: {beep_path}")

        # Text-to-speech engine (offline)
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 180)  # speaking speed
        except Exception as e:
            print(f"[AudioService] Could not init TTS engine: {e}")
            self.engine = None

    def play_beep(self):
        """Play the checkout beep sound (non-blocking)."""
        if self.wave_obj is None:
            return

        def _play():
            try:
                self.wave_obj.play()
            except Exception as e:
                print(f"[AudioService] Beep play error: {e}")

        threading.Thread(target=_play, daemon=True).start()

    def speak(self, text: str):
        """Speak a short sentence using TTS (blocking but quick)."""
        if self.engine is None:
            return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"[AudioService] TTS error: {e}")