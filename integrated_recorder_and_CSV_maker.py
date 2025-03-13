import tkinter as tk
import pyaudio
import wave
import time
import threading
import re
import csv
from collections import defaultdict
from datetime import timedelta, datetime
from openai import OpenAI
import queue
import logging
import os

# Initialize OpenAI client (ensure your API key is set up)
client = OpenAI()

# Set up logging
logging.basicConfig(filename="audio_recorder.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def get_dated_folder(base_folder):
    """Create and return the path to a dated subfolder (e.g., 'audio_recordings/2023-10-01')."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join(base_folder, date_str)
    os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist
    return folder_path

class AudioRecorder:
    def __init__(self, format=pyaudio.paInt16, channels=1, rate=44100, chunk=1024):
        """Initialize the audio recorder with specified parameters."""
        self.FORMAT = format
        self.CHANNELS = channels
        self.RATE = rate
        self.CHUNK = chunk
        self.frames = []
        self.is_recording = False
        self.start_time = None
        self.audio_stream = None
        self.audio = pyaudio.PyAudio()

    def start_recording(self):
        """Start recording audio in a background thread."""
        if not self.is_recording:
            self.is_recording = True
            self.frames = []
            self.start_time = time.time()
            self.audio_stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
            threading.Thread(target=self._capture_audio, daemon=True).start()

    def _capture_audio(self):
        """Capture audio data while recording is active."""
        while self.is_recording:
            data = self.audio_stream.read(self.CHUNK, exception_on_overflow=False)
            self.frames.append(data)

    def stop_recording(self):
        """Stop recording and save the audio file."""
        if self.is_recording:
            self.is_recording = False
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            output_filename = self.save_audio()
            num_frames = len(self.frames)
            duration = num_frames * self.CHUNK / self.RATE
            return str(timedelta(seconds=int(duration))), output_filename

    def save_audio(self):
        """Save the recorded audio to a WAV file in the dated audio folder."""
        base_folder = "audio_recordings"
        folder_path = get_dated_folder(base_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(folder_path, f"recorded_audio_{timestamp}.wav")
        try:
            with wave.open(output_filename, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(self.frames))
            logging.info(f"Audio saved to {output_filename}")
        except Exception as e:
            logging.error(f"Error saving audio: {e}")
            return None
        return output_filename

    def terminate(self):
        """Clean up audio resources."""
        self.audio.terminate()

def transcribe_audio(audio_file, model="whisper-1"):
    """Transcribe the audio file using OpenAI's Whisper model."""
    if not os.path.exists(audio_file):
        logging.error(f"Audio file {audio_file} not found.")
        return None
    try:
        with open(audio_file, "rb") as file:
            logging.info("Transcribing audio file...")
            transcription = client.audio.transcriptions.create(model=model, file=file)
            logging.info("Transcription complete.")
            return transcription.text
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        return None

def call_gpt4(input_text, model="gpt-4o", temperature=0.25):
    """Process the transcription with GPT-4 to generate a structured procedure."""
    try:
        instructions_text = '''\n\nTurn the informal audio transcript above into a structured engineering procedure, with Section Names, Steps and Step Notes. An example output is below (it MUST match this format to be parsed correctly):

        Section 1: Preparing Materials
        Step A1: Get bread, butter and a butter knife

        Section 2: Making toast
        Step 2: Put bread in the toaster slots
        Step 3: Press down toaster lever
        Step 3 Note: Toast should take approximately 1-2 minutes to cook. 
        Step 4: Once toasting completes, pull toast riser to elevate toast.
        Step 5: Use your hands to remove the toast.
        Step 5 Note: Toast may be hot. 

        Section 3: Buttering the toast
        Step 6: Scrape some butter from the tub.
        Step 7: Spread it evenly across the toast.
        Step 7 Note: Spread butter to the edges.
        '''
        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": input_text + instructions_text}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error calling GPT-4: {e}")
        return None

def parse_procedure(input_text):
    """Parse the GPT-4 response into a structured procedure."""
    try:
        procedure = defaultdict(lambda: {"steps": [], "notes": defaultdict(str)})
        current_section = ""
        section_pattern = re.compile(r"Section\s*(\d+):\s*(.+)", re.IGNORECASE)
        step_pattern = re.compile(r"Step\s*(\d+|[A-Za-z]\d*):\s*(.+)", re.IGNORECASE)
        note_pattern = re.compile(r"Step\s*(\d+|[A-Za-z]\d*)\s*Note\s*:\s*(.+)", re.IGNORECASE)

        for line in input_text.strip().split('\n'):
            line = line.strip()
            section_match = section_pattern.match(line)
            if section_match:
                section_number, section_name = section_match.groups()
                current_section = f"{section_name.strip()}"
                continue
            step_match = step_pattern.match(line)
            if step_match:
                step_number, step_description = step_match.groups()
                procedure[current_section]["steps"].append((step_number, step_description))
                continue
            note_match = note_pattern.match(line)
            if note_match:
                note_step_number, note_description = note_match.groups()
                procedure[current_section]["notes"][note_step_number] = note_description
        return procedure
    except Exception as e:
        logging.error(f"Error parsing the procedure: {e}")
        return None

def export_to_csv(procedure):
    """Export the parsed procedure to a CSV file in the dated CSV folder."""
    base_folder = "procedure_exports"
    folder_path = get_dated_folder(base_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(folder_path, f"procedure_steps_{timestamp}.csv")
    try:
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(["Step Name", "", "Notes"])
            for section, content in procedure.items():
                writer.writerow([section, "", ""])
                for step in content["steps"]:
                    step_num, step_description = step
                    note = content["notes"].get(step_num, "")
                    writer.writerow([f"Step {step_num}", step_description, note])
                writer.writerow(["", "", ""])
        logging.info(f"Procedure exported to {file_name}")
        return file_name
    except Exception as e:
        logging.error(f"Error exporting to CSV: {e}")
        return None

class AudioRecorderApp:
    def __init__(self, root, recorder):
        """Initialize the Tkinter GUI application."""
        self.root = root
        self.recorder = recorder
        self.message_queue = queue.Queue()
        self.text_box = tk.Text(root, height=10, width=50, state='disabled', wrap='word')
        self.text_box.pack(pady=10)
        self.button = tk.Button(root, text="Start Recording", command=self.toggle_recording, width=30, height=2)
        self.button.pack(pady=10)
        self.is_recording = False
        self.check_queue()

    def check_queue(self):
        """Periodically check the message queue for updates."""
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.display_message(message)
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)

    def display_message(self, message):
        """Display a message in the text box."""
        self.text_box.config(state='normal')
        self.text_box.insert(tk.END, message + '\n')
        self.text_box.see(tk.END)
        self.text_box.config(state='disabled')

    def toggle_recording(self):
        """Toggle between starting and stopping the recording."""
        if not self.is_recording:
            self.recorder.start_recording()
            self.is_recording = True
            self.button.config(text="Stop Recording")
            self.message_queue.put("Recording started...")
        else:
            duration, output_filename = self.recorder.stop_recording()
            self.is_recording = False
            self.button.config(text="Start Recording")
            if output_filename:
                self.message_queue.put(f"Recording finished. Duration: {duration}. Audio saved to {output_filename}")
                threading.Thread(target=self.process_audio, args=(output_filename,), daemon=True).start()
            else:
                self.message_queue.put("Error saving audio file.")

    def process_audio(self, output_filename):
        """Process the audio file in a background thread."""
        try:
            self.message_queue.put("Transcribing audio...")
            transcription = transcribe_audio(output_filename)
            if not transcription:
                self.message_queue.put("Error in transcription.")
                return
            self.message_queue.put("Transcription complete. Processing with GPT-4...")
            gpt4_response = call_gpt4(transcription)
            if not gpt4_response:
                self.message_queue.put("Error in GPT-4 response.")
                return
            self.message_queue.put("Procedure parsed. Exporting to CSV...")
            procedure = parse_procedure(gpt4_response)
            if not procedure:
                self.message_queue.put("Error parsing procedure.")
                return
            file_name = export_to_csv(procedure)
            if file_name:
                self.message_queue.put(f"Export complete! File saved as '{file_name}'")
            else:
                self.message_queue.put("Error exporting to CSV.")
        except Exception as e:
            self.message_queue.put(f"An error occurred: {e}")

    def on_close(self):
        """Handle window close event by cleaning up resources."""
        self.recorder.terminate()
        self.root.destroy()

def main():
    """Main function to run the application."""
    recorder = AudioRecorder()
    root = tk.Tk()
    root.title("Audio Recorder")
    app = AudioRecorderApp(root, recorder)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()