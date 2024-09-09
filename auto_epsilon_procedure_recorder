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

# Initialize OpenAI client
client = OpenAI()

# Audio Recorder Class
class AudioRecorder:
    def __init__(self, format=pyaudio.paInt16, channels=1, rate=44100, chunk=1024):
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
        if not self.is_recording:
            self.is_recording = True
            self.frames = []
            self.start_time = time.time()

            # Open a new audio stream
            self.audio_stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

            # Start capturing audio in a separate thread
            threading.Thread(target=self._capture_audio).start()

    def _capture_audio(self):
        while self.is_recording:
            data = self.audio_stream.read(self.CHUNK)
            self.frames.append(data)

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.audio_stream.stop_stream()
            self.audio_stream.close()

            # Save the recorded audio
            output_filename = self.save_audio()

            # Calculate and return the duration
            duration = time.time() - self.start_time
            return str(timedelta(seconds=int(duration))), output_filename

    def save_audio(self):
        # Generate a unique filename using the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"recorded_audio_{timestamp}.wav"

        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.frames))

        return output_filename

    def terminate(self):
        self.audio.terminate()

# Functions for transcription, GPT-4 processing, parsing, and exporting to CSV
def transcribe_audio(audio_file, model="whisper-1"):
    """Transcribes the audio file using the Whisper API."""
    try:
        with open(audio_file, "rb") as file:
            print("Transcribing audio file...")
            transcription = client.audio.transcriptions.create(model=model, file=file)
            print("Transcription complete.")
            return transcription.text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def call_gpt4(input_text, model="gpt-4", temperature=0.25):
    """Calls GPT-4 to structure the input text into an engineering procedure."""
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
        response_text = completion.choices[0].message.content
        return response_text
    except Exception as e:
        print(f"Error calling GPT-4: {e}")
        return None

def parse_procedure(input_text):
    """Parses the input text into sections, steps, and notes."""
    try:
        procedure = defaultdict(lambda: {"steps": [], "notes": defaultdict(str)})
        current_section = ""

        # Regular expressions for matching Sections, Steps, and Notes
        section_pattern = re.compile(r"Section\s+(\d+):\s+(.+)")
        step_pattern = re.compile(r"Step\s+(\d+):\s+(.+)")
        note_pattern = re.compile(r"Step\s+(\d+)\s+Note:\s+(.+)")

        for line in input_text.strip().split('\n'):
            line = line.strip()

            # Check for sections
            section_match = section_pattern.match(line)
            if section_match:
                section_number, section_name = section_match.groups()
                current_section = f"{section_name.strip()}"
                continue

            # Check for steps
            step_match = step_pattern.match(line)
            if step_match:
                step_number, step_description = step_match.groups()
                procedure[current_section]["steps"].append((int(step_number), step_description))
                continue

            # Check for notes
            note_match = note_pattern.match(line)
            if note_match:
                note_step_number, note_description = note_match.groups()
                procedure[current_section]["notes"][int(note_step_number)] = note_description

        return procedure
    except Exception as e:
        print(f"Error parsing the procedure: {e}")
        return None

def export_to_csv(procedure, file_name="procedure_steps_from_audio.csv"):
    """Exports the parsed procedure into a CSV file with the required structure."""
    try:
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            
            # Write the header
            writer.writerow(["Step Name", "", "Notes"])
            
            # Loop through each section and write steps and notes
            for section, content in procedure.items():
                writer.writerow([section, "", ""])
                for step in content["steps"]:
                    step_num, step_description = step
                    note = content["notes"].get(step_num, "")
                    writer.writerow([f"Step {step_num}", step_description, note])

                writer.writerow(["", "", ""])

        print(f"Procedure exported to {file_name}")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")

# GUI Code (Tkinter)
class AudioRecorderApp:
    def __init__(self, root, recorder):
        self.root = root
        self.recorder = recorder
        
        # Create a text widget for status messages
        self.text_box = tk.Text(root, height=10, width=50, state='disabled', wrap='word')
        self.text_box.pack(pady=10)
        
        # Create a button to start/stop recording
        self.button = tk.Button(root, text="Start Recording", command=self.toggle_recording, width=30, height=2)
        self.button.pack(pady=10)
        
        self.is_recording = False

    def toggle_recording(self):
        if not self.is_recording:
            self.recorder.start_recording()
            self.is_recording = True
            self.button.config(text="Stop Recording")
            self.display_message("Recording started...")
        else:
            # Stop recording and begin processing
            duration, output_filename = self.recorder.stop_recording()
            self.is_recording = False
            self.button.config(text="Start Recording")
            self.display_message(f"Recording finished. Duration: {duration}. Transcribing audio...")

            # Begin transcription, GPT-4 processing, and CSV export
            try:
                transcription = transcribe_audio(output_filename)
                if transcription:
                    self.display_message("Transcription complete. Parsing procedure...")
                    gpt4_response = call_gpt4(transcription)
                    if gpt4_response:
                        self.display_message("Procedure parsed. Exporting to CSV...")
                        procedure = parse_procedure(gpt4_response)
                        if procedure:
                            export_to_csv(procedure)
                            self.display_message("Export complete! File saved as 'procedure_steps_from_audio.csv'")
                        else:
                            self.display_message("Error parsing procedure.")
                    else:
                        self.display_message("Error in GPT-4 response.")
                else:
                    self.display_message("Error in transcription.")
            except Exception as e:
                self.display_message(f"An error occurred: {e}")

    def display_message(self, message):
        self.text_box.config(state='normal')  # Enable text widget for editing
        self.text_box.insert(tk.END, message + '\n')  # Insert the message
        self.text_box.see(tk.END)  # Scroll to the end of the text box
        self.text_box.config(state='disabled')  # Disable text widget again to prevent user edits

    def on_close(self):
        self.recorder.terminate()
        self.root.destroy()

# Main function for initializing the app
def main():
    recorder = AudioRecorder()
    
    root = tk.Tk()
    root.title("Audio Recorder")
    
    app = AudioRecorderApp(root, recorder)
    
    root.protocol("WM_DELETE_WINDOW", app.on_close)  # Ensure resources are cleaned up when closing the window
    
    root.mainloop()

if __name__ == "__main__":
    main()
