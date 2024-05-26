import io
import multiprocessing
import tempfile
import time
import warnings
import librosa
import os
import audioop
import wave
import math
import whisper
import pysrt
import six
import ffmpeg
import tkinter as tk
from tkinter import filedialog
import torch

warnings.filterwarnings('ignore')

SAVING_PATH = os.curdir
WHISPER_MODEL = 'base' # Or any whisper model, check 'Available models and languages' section on https://pypi.org/project/openai-whisper/ 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# WHISPER_MODEL_LIST = ['tiny','base','small','medium','large']

class WhisperRecogniser:
    def __init__(self):
        self.model = whisper.load_model(name=WHISPER_MODEL,device=DEVICE)
        self.device = DEVICE

    def __call__(self, audio_data):
        audio_data = whisper.pad_or_trim(audio_data)
        mel = whisper.log_mel_spectrogram(audio_data).to(self.device)

        # Detect language in audio
        _, probs = self.model.detect_language(mel)

        transcription = self.model.decode(mel, whisper.DecodingOptions(fp16=False))
        return transcription.text, max(probs, key=probs.get)
    
class Converter:
    def __init__(self, path):
        self.path = path

    def __call__(self, region):
        start_point, end_point = region

        # Adjust start and end times based on include_before and include_after
        start_point = max(0, start_point - 0.3)
        end_point += 0.3

        # Create ffmpeg input options
        in_opts = {
            'ss': start_point,
            't': end_point - start_point,
        }

        # Run ffmpeg command and capture output
        out, _ = (
            ffmpeg
            .input(self.path, **in_opts)
            .output('pipe:', format='wav')
            .run(capture_stdout=True, capture_stderr=True)
        )

        # Convert bytes to numpy array
        data, _ = io.BytesIO(out).read(), None
        return data


class Audio2Subtitle:

    def __init__(self, filename):
        self.filename = filename
        self.process_finished = False

    def format_srt_subtitles(self, subtitles): # For srt file
        srt_file = pysrt.SubRipFile()

        for index, subtitle_data in enumerate(subtitles, start=1):
            (start_time, end_time), text = subtitle_data
            subtitle_item = pysrt.SubRipItem()
            subtitle_item.index = index
            subtitle_item.text = six.text_type(text)
            subtitle_item.start.seconds = max(0, start_time)
            subtitle_item.end.seconds = end_time
            srt_file.append(subtitle_item)

        formatted_srt = '\n'.join(six.text_type(item) for item in srt_file)
        return formatted_srt
    
    def format_transcript(self, subtitles):
        srt_file = ''

        for _, subtitle_data in enumerate(subtitles, start=1):
            (start_time, end_time), text = subtitle_data
            srt_file += f"[{start_time:.2f}->{end_time:.2f}] {text}\n"

        return srt_file
    
    def transform_audio(self):
        temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        if not os.path.isfile(self.filename):
            print("Not found {}".format(self.filename))
            return None

        ffmpeg.input(self.filename).output(temp.name, ac=1, ar=16000).run(overwrite_output=True, quiet=True)

        return temp.name

    @staticmethod
    def calculate_percentile(arr, percentile_value):
        sorted_arr = sorted(arr)
        index = (len(sorted_arr) - 1) * percentile_value
        floor_index = math.floor(index)
        ceil_index = math.ceil(index)
        if floor_index == ceil_index:
            return sorted_arr[int(index)]
        lower_value = sorted_arr[int(floor_index)] * (ceil_index - index)
        upper_value = sorted_arr[int(ceil_index)] * (index - floor_index)
        
        return lower_value + upper_value

    def regions_dividing(self, filename):
        frame_width=4096
        min_size = 0.5
        max_size = 6
        reader = wave.open(filename)
        sample_width = reader.getsampwidth()
        rate = reader.getframerate()
        n_channels = reader.getnchannels()
        chunk_duration = float(frame_width) / rate

        n_chunks = int(math.ceil(reader.getnframes() * 1.0 / frame_width))
        energies = self.calculate_chunk_energies(reader, n_chunks, frame_width,
                                                 sample_width, n_channels)

        threshold = self.calculate_energy_threshold(energies, 0.2)
        elapsed_time = 0
        regions = []
        region_start = None

        for energy in energies:
            is_silence = energy <= threshold
            max_exceeded = region_start and elapsed_time - region_start >= max_size

            if (max_exceeded or is_silence) and region_start:
                if elapsed_time - region_start >= min_size:
                    regions.append((region_start, elapsed_time))
                    region_start = None

            elif (not region_start) and (not is_silence):
                region_start = elapsed_time
            elapsed_time += chunk_duration

        return regions

    def calculate_chunk_energies(self, reader, n_chunks, frame_width, sample_width, n_channels):
        energies = []
        for _ in range(n_chunks):
            chunk = reader.readframes(frame_width)
            energies.append(audioop.rms(chunk, sample_width * n_channels))
        return energies

    def calculate_energy_threshold(self, energies, percentile_value):
        sorted_energies = sorted(energies)
        index = int(len(sorted_energies) * percentile_value)
        return sorted_energies[index]

    def generate_subtitle(self):
        audio_filename = self.transform_audio()
        partitions = self.regions_dividing(audio_filename)
        pool = multiprocessing.Pool(10)
        converter = Converter(path=audio_filename)
        recognizer = WhisperRecogniser()
        total_transcribe_text = []
        print("Preparing to start generating subtitles.")
        start_time = time.time()
        if partitions:
            divided_partitions = []
            total_regions = len(partitions)  
            for i, divided_region in enumerate(pool.imap(converter, partitions)):
                data, _ = librosa.load(io.BytesIO(divided_region), sr=16000)
                divided_partitions.append(data)

            for i, data in enumerate(divided_partitions):
                transcribe_text, language = recognizer(data)
                # Print percentage
                progress_percentage = (i + 1) / total_regions * 100
                print(f"[{language}][{progress_percentage:.2f}%]:{transcribe_text} ")
                total_transcribe_text.append(transcribe_text)
                print()

        transcripts_with_time_stamp = ([(r, t) for r, t in zip(partitions, total_transcribe_text) if t])
        formatted_transcript = self.format_transcript(transcripts_with_time_stamp)
        fullname = os.path.splitext(self.filename)[0]
        name = os.path.basename(fullname)
        full_path = f"{SAVING_PATH}\\{name}.txt"
        with open(full_path, 'wb') as output_file:
            output_file.write(formatted_transcript.encode("utf-8"))
        os.remove(audio_filename)
        self.process_finished = True
        elapse = time.time() - start_time
        print(f"Successfully transcript at:{full_path}")
        print(f"Total time cost: {elapse}s")


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    # Prompt file selection dialog
    file_path = filedialog.askopenfilename()

    # Check if a file was selected
    if file_path:
        # Use the selected file path
        print("Selected file:", file_path)
    else:
        print("No file selected.")
        exit()
    
    example = Audio2Subtitle(file_path)
    example.generate_subtitle()
