import librosa
import numpy as np
import soundfile as sf

def anonymize(input_audio_path):
    """
    Anonymizes an audio file by pitch shifting and adding noise.

    Parameters
    ----------
    input_audio_path : str
        path to the source audio file in ".wav" format.

    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type `np.float32`.
    sr : int
        The sample rate of the processed audio.
    """

    # Load the audio file
    audio, sr = librosa.load(input_audio_path, sr=None)

    # Pitch shifting
    audio = librosa.effects.pitch_shift(audio, sr, n_steps=4)  # shift by 4 semitones

    # Adding Gaussian noise
    noise = np.random.normal(0, 0.01, audio.shape)
    audio = audio + noise

    # Ensure the audio values are within the range [-1, 1]
    audio = np.clip(audio, -1, 1)

    # Convert to float32 for compatibility with soundfile.write()
    audio = audio.astype(np.float32)

    return audio, sr
