import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pyaudio
import numpy as np
import wave
import os
import glob
import time
import pickle
from sklearn.svm import SVC
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from tkinter import filedialog
import keyboard as kb
import mouse as ms
import statistics as st
import pyautogui as py
import threading


path = "../python/guidata"
selected_type = None
model_trained = False


move_count = 0


# Function to scroll the mouse wheel
def scroll(direction):
    if direction == "up":
        ms.wheel(1)
    elif direction == "down":
        ms.wheel(-1)


def before(value, a):
    # Find first part and return slice before it.
    pos_a = value.find(a)
    if pos_a == -1: return ""
    return value[0:pos_a]


def update_recorded_status(type_name, recorded=True):
    for idx, (t, p, r) in enumerate(selected_types_and_purposes):
        if t == type_name:
            selected_types_and_purposes[idx] = (t, p, recorded)
            break

    # Update listbox colors
    update_listbox_colors()

    # Check if all types are recorded and enable/disable the "Train Model" button accordingly
    all_recorded = all(recorded for _, _, recorded in selected_types_and_purposes)
    train_button.config(state=tk.NORMAL if all_recorded else tk.DISABLED)


def update_listbox_colors():
    type_listbox.delete(0, tk.END)
    purpose_listbox.delete(0, tk.END)

    for type_name, purpose, recorded in selected_types_and_purposes:
        type_listbox.insert(tk.END, f"{type_name}")
        purpose_listbox.insert(tk.END, f"{purpose}")

        # Set background color based on recording status
        bg_color = "lightgreen" if recorded else "white"
        type_listbox.itemconfig(tk.END, {'bg': bg_color})
        purpose_listbox.itemconfig(tk.END, {'bg': bg_color})


def record_training_data(type_name, directory):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = 6
    count = 0
    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames
    for i in range(0, int(fs / chunk * seconds)):
        input_data = np.frombuffer(stream.read(chunk), dtype=np.int16)
        # print(input_data)
        if np.mean(abs(input_data)) > 50:
            filename = path
            filename = filename + '/' + str(type_name) + '_' + str(count) + '.wav'
            count = count + 1
            wf = wave.open(filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(sample_format))
            wf.setframerate(fs)
            wf.writeframes(b''.join(input_data))
            wf.close()

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()


def validate_entries():
    type_name = type_entry.get()
    selected_purpose = purpose_combobox.get()

    if type_name and selected_purpose != "Select Function":
        add_button.config(state=tk.NORMAL)
    else:
        add_button.config(state=tk.DISABLED)


# Store selected types, purposes, and recording status
selected_types_and_purposes = []


def add_type_with_purpose(type_name, purpose):

    if type_name and purpose != "Select Function":
        add_type_with_purpose_to_lists(type_name, purpose)
        update_listbox_colors()

        type_entry.delete(0, tk.END)
        purpose_combobox.set("Select Function")
        add_button.config(state=tk.DISABLED)


def add_type_with_purpose_to_lists(type_name, purpose, recorded=False):
    type_listbox.insert(tk.END, f"{type_name}")
    purpose_listbox.insert(tk.END, f"{purpose}")
    selected_types_and_purposes.append((type_name, purpose, recorded))


def get_selected_types():
    selected_types = type_listbox.get(0, tk.END)
    selected_purposes = purpose_listbox.get(0, tk.END)
    return selected_types, selected_purposes


def show_help():
    help_message = (
        "Function:\n"
        "   - Select Text Left: Select text from right to left.\n"
        "   - Select Text Right: Select text from left to right.\n\n"
        
        "How to Use:\n"
        
        "   1. Import preset by clicking \"import preset\" located at bottom left or enter "
        "Key combinations and assign its function on the top.\n"
        
        "   2. Select one combination and click on \"Record Training Data\" (select one combination only). "
        "Will display \"Recording finished\" after recording is finished.\n"
        
        "   3. After all data are recorded, Click \"Train Model\" to train.\n"
        
        "   4. Click on \"Real-Time Test\" to start real-time test, and press \"Ctrl\" "
        "key to start classification; Click \"stop\" to stop.\n\n"

    )
    messagebox.showinfo("Help", help_message)


def start_recording_thread():
    if selected_type:
        delete_directory = path + '/' + selected_type + '*'
        for filename in glob.glob(delete_directory):
            os.remove(filename)
        # Update recorded status to False
        update_recorded_status(selected_type, False)

    # Start the real-time testing loop in a separate thread
    recording_thread = threading.Thread(target=start_recording)
    recording_thread.start()


def start_recording():
    if selected_type:
        recording_status_label.config(text=f"Recording {selected_type}...", foreground="red")
        app.update()  # Update the GUI to immediately show the label change
        record_training_data(selected_type, path)
        update_recorded_status(selected_type, True)  # Mark type as recorded
        recording_status_label.config(text=f"Recording {selected_type} finished", foreground="green")
        app.update()  # Update the GUI to immediately show the label change
        time.sleep(1)  # Add a delay of 1 second between recordings

        # Check if all types are recorded and enable/disable the "Train Model" button accordingly
        all_recorded = all(recorded for _, _, recorded in selected_types_and_purposes)
        train_button.config(state=tk.NORMAL if all_recorded else tk.DISABLED)
    else:
        messagebox.showwarning("No Type Selected", "Please select a type before recording.")
    all_recorded = all(recorded for _, _, recorded in selected_types_and_purposes)
    if all_recorded:
        recording_status_label.config(text=f"Ready To Train Model", foreground="green")
        app.update()  # Update the GUI to immediately show the label change


def on_type_selection(event):
    global selected_type
    selected_index = type_listbox.curselection()
    if selected_index:
        selected_type = type_listbox.get(selected_index[0])
        record_training_data_button.config(state=tk.NORMAL)
    else:
        selected_type = None
        record_training_data_button.config(state=tk.DISABLED)


def remove_type():
    if messagebox.askyesno("Confirmation", "Are you sure you want to remove selected gesture, its assigned function, and data?"):
        selected_indices = type_listbox.curselection()[::-1]
        for index in selected_indices:
            type_listbox.delete(index)
            purpose_listbox.delete(index)
            del selected_types_and_purposes[index]
            delete_directory = path + '/' + type_listbox.get(index) + '*'
            for filename in glob.glob(delete_directory):
                os.remove(filename)


def delete_all_recorded_data():
    if messagebox.askyesno("Confirmation", "Are you sure you want to delete all recorded data?"):
        type_name = type_listbox.get(0, tk.END)
        for i in type_name:
            print(i)
            update_recorded_status(i, False)
        update_listbox_colors()
        # Implement code to delete all recorded files
        delete_directory = path + '/*'
        for filename in glob.glob(delete_directory):
            os.remove(filename)


def remove_all_types():
    if messagebox.askyesno("Confirmation", "Are you sure you want to remove all gestures, assigned functions, and data?"):
        global selected_types_and_purposes
        selected_types_and_purposes = []
        type_listbox.delete(0, tk.END)  # Clear the listbox
        purpose_listbox.delete(0, tk.END)  # Clear the listbox
        update_listbox_colors()
        # Implement code to delete all recorded files
        delete_directory = path + '/*'
        for filename in glob.glob(delete_directory):
            os.remove(filename)


def train_model():
    recording_status_label.config(text="Training", foreground="red")
    app.update()  # Update the GUI to immediately show the label change
    global model_trained
    x_train = []
    y_train = []
    for filename in os.listdir(path + '/'):
        if filename.endswith('.wav'):
            (rate, sig) = wav.read(path + '/' + filename)
            # get mfcc
            mel = mfcc(sig, samplerate=rate,
                       numcep=64, nfilt=64, nfft=1103)
            x_train.append(mel)
            y_train.append(before(filename, "_"))

    # Unscaled test data ready in a numpy array
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Scale all data...
    scale = np.concatenate(x_train, axis=0)
    scalers = {}

    x_train_2d = x_train.reshape(x_train.shape[0], -1)

    clf = SVC()
    clf.fit(x_train_2d, y_train)

    # save the model to disk
    filename = 'withGui.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print("Model training completed")
    model_trained = True
    recording_status_label.config(text="Train Model Completed", foreground="green")
    app.update()  # Update the GUI to immediately show the label change
    app.after(3000, clear_message)
    recording_status_label.config(
        text=f"Ready to Proceed to Real-Time Testing",foreground="green")
    app.update()  # Update the GUI to immediately show the label change


def start_real_time_testing():
    global stop_testing
    stop_testing = False

    # Start the real-time testing loop in a separate thread
    testing_thread = threading.Thread(target=real_time_model)
    testing_thread.start()


def stop_real_time_testing():
    global stop_testing
    stop_testing = True
    recording_status_label.config(text="Testing stopped", foreground="red")
    app.update()  # Update the GUI to immediately show the label change


def clear_message():
    recording_status_label.config(text="")
    app.update()


def breath_fast(arr):
    current_string = None
    current_count = 0

    for string in arr:
        if string == current_string:
            current_count += 1
        else:
            if current_count >= 20:
                return False  # If any consecutive count is 20 or more, return False
            current_string = string
            current_count = 1

    # Check the last consecutive count
    if current_count >= 20:
        return False

    return True


def breath(breathing,storage):
    if len(storage) < 110:
        storage.append(breathing)
    else:
        storage.insert(0, breathing)
        storage.pop()
    storage.count("Breathing In")
    if len(storage) >= 80:
        breathing_fast = breath_fast(storage)
        if breathing_fast:
            messagebox.showinfo("Attention","You are breathing too fast! Take a break!")
        else:
            print("OK")
            recording_status_label.config(text="Breathing OK", foreground="green")
            app.update()  # Update the GUI to immediately show the label change


def real_time_model():
    global stop_testing
    stop_testing = False  # Initialize the flag before testing
    selected_length = 1000
    # Implement real-time testing logic with the selected length of time
    # You can use the selected_length to control the duration of testing
    print(f"Performing real-time testing for {selected_length} seconds")

    selected_types, selected_purposes = get_selected_types()

    filename = 'withGui.sav'
    FiveKeyModel = pickle.load(open(filename, 'rb'))
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = selected_length
    count = 0
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    vote = []  # to reduce false positive
    compare = []  # cumulative move
    storage = []  # to detect breathing pace
    word_selected = 0
    typed = []
    pressed = 0 # ctrl hot key switch
    moveLeft = 0
    moveRight = 0
    moveUp = 0
    moveDown = 0
    print('Testing')
    recording_status_label.config(text=f"Hold Ctrl Key to Detect Keyswipe", foreground="green")
    app.update()
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    for i in range(0, int(fs / chunk * seconds)):
        if stop_testing:
            break
        input_data = np.frombuffer(stream.read
                                   (chunk), dtype=np.int16)
        """
            if kb.is_pressed('ctrl') and pressed == 0:
            pressed = 1
            recording_status_label.config(text=f"Testing", foreground="green")
            app.update()
            time.sleep(1)
        elif kb.is_pressed('ctrl') and pressed == 1:
            pressed = 0
            recording_status_label.config(text=f"Test Paused", foreground="red")
            app.update()
            time.sleep(1)
        """
        if np.mean(abs(input_data)) > 50:
            result = ''
            position = ms.get_position()
            data = mfcc(input_data, samplerate=fs,
                        numcep=64, nfilt=64, nfft=1103)
            # Unscaled data ready in a numpy array
            data = np.array(data)
            # Scale all data...
            scale = np.concatenate(data, axis=0)
            x_train_2d = data.reshape(data.shape[0], -1)
            predict2 = FiveKeyModel.predict(data)
            # to store current and 2 previous result to compare
            if len(vote) < 3:
                vote.append(np.array2string(predict2))
            else:
                vote.insert(0, np.array2string(predict2))
                vote.pop()
            result1 = st.mode(vote)
            # result1 is a string with '' and [], needs to be removed
            i = 2
            check = len(result1) - 2
            while i < check:
                result = result + result1[i]
                i = i + 1
            # to reduce false positive rate
            if len(compare) < 2:
                compare.append(np.array2string(predict2))
            else:
                compare.insert(0, np.array2string(predict2))
                compare.pop()
                if compare[1] == compare[0]:
                    count = count + 1
                else:
                    count = 0
            move = selected_purposes[int(selected_types.index(result))]
            if kb.is_pressed('ctrl'):
                kb.release('ctrl')
                recording_status_label.config(text=f"Predicting {result}...", foreground="green")
                app.update()
                if move == "Move Widget Left":
                    moveLeft = moveLeft + 1
                    moveRight = 0
                    moveUp = 0
                    moveDown = 0
                    #move_cursor("left",count, amount)
                    if moveLeft > 0:
                        py.press('left', presses=1)
                        moveLeft = 0
                    word_selected = 0
                elif move == "Move Widget Right":
                    moveLeft = 0
                    moveRight = moveRight + 1
                    moveUp = 0
                    moveDown = 0
                    #move_cursor("right", count, amount)
                    if moveRight > 0:
                        py.press('right', presses=1)
                        moveRight = 0
                    word_selected = 0
                elif move == "Move Widget Up":
                    moveLeft = 0
                    moveRight = 0
                    moveUp = moveUp + 1
                    moveDown = 0
                    #move_cursor("up", count, amount)
                    if moveUp > 0:
                        py.press('up', presses=1)
                        moveUp = 0
                    word_selected = 0
                elif move == "Move Widget Down":
                    moveLeft = 0
                    moveRight = 0
                    moveUp = 0
                    moveDown = moveDown + 1
                    #move_cursor("down", count, amount)
                    if moveDown > 0:
                        py.press('down', presses=1)
                        moveDown = 0
                    word_selected = 0
                elif move == "Select Text Right" or "Select Text Left":
                    if word_selected == 0:
                        #ms.press(button='left')
                        #ms.release(button='left')
                        word_selected = 1
                        if move == "Select Text Right":  # pyautogui is much more reliable
                            py.hotkey("ctrl", "left")
                            py.keyDown('shiftleft')
                            py.keyDown('shiftright')
                            py.keyDown('ctrl')
                            py.press('right', presses=1)
                            py.keyUp('shiftleft')
                            py.keyUp('shiftright')
                        elif move == "Select Text Left":
                            py.hotkey("ctrl", "right")
                            py.keyDown('shiftleft')
                            py.keyDown('shiftright')
                            py.keyDown('ctrl')
                            py.press('left', presses=1)
                            py.keyUp('shiftleft')
                            py.keyUp('shiftright')
                        py.keyUp('ctrl')
            else:
                if move == "Volume Up":
                    py.press('volumeup', 1)
                    recording_status_label.config(text=f"Turning Volume Up...", foreground="green")
                    app.update()  # Update the GUI to immediately show the label change
                elif move == "Volume Down":
                    py.press('volumedown', 1)
                    recording_status_label.config(text=f"Turning Volume Down...", foreground="green")
                    app.update()  # Update the GUI to immediately show the label change
                elif move == "Breathing":
                    breath(move,storage)
                elif move == "Scroll Mouse Wheel Up":
                    scroll("up")
                    word_selected = 0
                elif move == "Scroll Mouse Wheel Down":
                    scroll("down")
                    word_selected = 0
                elif move == "Type":
                    if len(typed) < 2:
                        typed.append(result)
                    else:
                        typed.insert(0, result)
                        typed.pop()
                        if not typed[0] == typed[1]:
                            for x in result:
                                py.press(x)
                            py.press('space')
                        recording_status_label.config(text=f"You are saying {result}...", foreground="green")
                        app.update()  # Update the GUI to immediately show the label change

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()
    recording_status_label.config(text="Testing stopped", foreground="red")
    app.update()  # Update the GUI to immediately show the label change
    app.after(2000, clear_message)
    if not stop_testing:
        recording_status_label.config(text="")
        app.update()


def import_presets():
    filename = filedialog.askopenfilename(title="Import Presets", filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
    if filename:
        try:
            with open(filename, "r") as file:
                presets = file.readlines()

            for preset in presets:
                type_name, purpose = preset.strip().split(",")
                add_type_with_purpose(type_name, purpose)

            update_listbox_colors()
            recording_status_label.config(text=f"Preset Imported. Add more gestures or \nselect one gesture and recording training data",
                                          foreground="green")
            app.update()  # Update the GUI to immediately show the label change

        except FileNotFoundError:
            messagebox.showerror("Error", "Presets file not found.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while importing presets: {e}")


def save_presets():
    filename = filedialog.asksaveasfilename(title="Save Presets", defaultextension=".txt", filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
    if filename:
        try:
            with open(filename, "w") as file:
                for type_name, purpose, _ in selected_types_and_purposes:
                    file.write(f"{type_name},{purpose}\n")
            messagebox.showinfo("Presets Saved", "Presets have been saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving presets: {e}")



app = tk.Tk()
app.title("Guided Training Tool")

# Adjusting the window size (width x height)
app.geometry("800x600")

# Create a frame for the center part
center_frame = ttk.Frame(app)
center_frame.pack(padx=20, pady=20)

help_button = ttk.Button(app, text="Help", command=show_help, style="TButton")
help_button.place(relx=0.02, rely=0.02, anchor=tk.NW)

# Entry for type name
type_frame = ttk.LabelFrame(center_frame, text="Keys")
type_frame.pack(fill=tk.BOTH, expand=True)


type_label = ttk.Label(type_frame, text="Enter a combination of adjacent keys:")
type_label.pack(pady=5)

type_entry = ttk.Entry(type_frame)
type_entry.pack(padx=10, pady=5)

purpose_label = ttk.Label(type_frame, text="Select the Function:")
purpose_label.pack(pady=5)

purposes = [
    "Move Widget Left", "Move Widget Right", "Move Widget Up", "Move Widget Down",
    "Select Text Left", "Select Text Right",
    "Scroll Mouse Wheel Up", "Scroll Mouse Wheel Down",
    "Volume Up", "Volume Down","Breathing","Type",
]

purpose_combobox = ttk.Combobox(type_frame, values=purposes, state="readonly")
purpose_combobox.set("Select Function")
purpose_combobox.pack(padx=10, pady=5)


add_button = ttk.Button(type_frame, text="Add Gesture with Function",
                        command=lambda: add_type_with_purpose(type_entry.get(), purpose_combobox.get()), style="TButton")
add_button.pack(pady=5)
add_button.config(state=tk.DISABLED)  # Initially disable the button


# Create a frame for the listboxes
listbox_frame = ttk.Frame(center_frame)
listbox_frame.pack(fill=tk.BOTH, expand=True)

# Create a frame for the buttons
button_frame = ttk.Frame(listbox_frame)
button_frame.pack(fill=tk.BOTH)


# Button to remove selected types
remove_button = ttk.Button(button_frame, text="Remove Selected", command=remove_type, style="TButton")
remove_button.pack(side=tk.LEFT, padx=5, pady=5)

# Button to remove all types
remove_all_button = ttk.Button(button_frame, text="Remove All Gestures", command=remove_all_types, style="TButton")
remove_all_button.pack(side=tk.LEFT, padx=5, pady=5)

# Button to delete all recorded data
delete_button = ttk.Button(button_frame, text="Delete All Recordings", command=delete_all_recorded_data, style="TButton")
delete_button.pack(side=tk.LEFT, padx=5, pady=5)

# Title for the displayed types and purposes
display_title_frame = ttk.Frame(listbox_frame)
display_title_frame.pack(fill=tk.BOTH)

display_type_label = ttk.Label(display_title_frame, text="Gestures")
display_type_label.pack(side=tk.LEFT, padx=(0, 10), pady=5, fill=tk.BOTH, expand=True)

display_purpose_label = ttk.Label(display_title_frame, text="Functions")
display_purpose_label.pack(side=tk.LEFT, padx=(10, 0), pady=5, fill=tk.BOTH, expand=True)

# Listbox for displaying added types and purposes
type_listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE)
type_listbox.pack(side=tk.LEFT, padx=(0, 10), pady=10, fill=tk.BOTH, expand=True)
type_listbox.bind("<<ListboxSelect>>", on_type_selection)


purpose_listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE)
purpose_listbox.pack(side=tk.LEFT, padx=(10, 0), pady=10, fill=tk.BOTH, expand=True)


# Validate entries whenever a value changes
type_entry.bind("<KeyRelease>", lambda event: validate_entries())
purpose_combobox.bind("<<ComboboxSelected>>", lambda event: validate_entries())


# Button to start recording training data
record_training_data_button = ttk.Button(center_frame, text="Record Training Data", command=start_recording_thread, style="TButton")
record_training_data_button.pack(pady=5)
record_training_data_button.config(state=tk.DISABLED)  # Initially disable the button

train_button = ttk.Button(center_frame, text="Train Model", command=train_model, style="TButton")
train_button.pack(pady=5)
train_button.config(state=tk.DISABLED)  # Initially disable the button

# message box
message_box_frame = ttk.Frame(center_frame, borderwidth=3, relief="solid", style="Highlight.TFrame")
message_box_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Configure a custom style for the highlighted frame
style = ttk.Style()
style.configure("Highlight.TFrame", background="yellow")

# Create a label for recording status
recording_status_label = ttk.Label(center_frame, text="", font=("Helvetica", 12, "bold"))
recording_status_label.pack(pady=10)

real_time_button = ttk.Button(center_frame, text="Real-Time Testing", command=start_real_time_testing, style="TButton")
real_time_button.pack(pady=5)

stop_button = ttk.Button(center_frame, text="Stop", command=stop_real_time_testing, style="TButton")
stop_button.pack(pady=5)

# Create a frame for the buttons
button_frame = ttk.Frame(listbox_frame)
button_frame.pack(fill=tk.BOTH)

# Create a frame for the import and save buttons
import_save_frame = ttk.Frame(center_frame)
import_save_frame.pack(fill=tk.BOTH, expand=True)

# Button to import presets
import_button = ttk.Button(import_save_frame, text="Import Presets", command=import_presets, style="TButton")
import_button.pack(side=tk.LEFT, padx=5, pady=5)

# Button to save presets
save_button = ttk.Button(import_save_frame, text="Save Presets", command=save_presets, style="TButton")
save_button.pack(side=tk.RIGHT, padx=5, pady=5)


# Update the window size based on the content and layout
app.update()
app_width = center_frame.winfo_reqwidth() + 40
app_height = center_frame.winfo_reqheight() + 60
app.geometry(f"{app_width}x{app_height}")

recording_status_label.config(text=f"Enter a gesture and assign a function or \nImport Preset", foreground="green")
app.update()  # Update the GUI to immediately show the label change

app.mainloop()

