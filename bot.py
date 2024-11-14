from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk  # Requires 'pillow' library for handling images

# Load the DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize chat history
chat_history_ids = None

# Function to get the model's response
def get_response():
    global chat_history_ids
    query = user_input.get()
    
    if query.lower() in exit_conditions:
        root.quit()
    else:
        # Clear the user input box
        user_input.delete(0, tk.END)
        
        # Encode the user input and append the chat history
        new_input_ids = tokenizer.encode(query + tokenizer.eos_token, return_tensors="pt")
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

        # Generate a response
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        
        # Decode the response
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        # Display both user input and bot response in the chat box
        chat_box.config(state=tk.NORMAL)
        chat_box.insert(tk.END, "You: " + query + "\n", "user")
        chat_box.insert(tk.END, "Winter: " + response + "\n\n", "bot")
        chat_box.config(state=tk.DISABLED)
        chat_box.yview(tk.END)  # Auto-scroll to the latest message

# Set up exit conditions
exit_conditions = (":q", "quit", "exit")

# Initialize the tkinter window
root = tk.Tk()
root.title("Winter Chatbot")
root.geometry("800x700")
root.configure(bg="white")  # White background color for the whole window

# Center the window on the screen
root.eval('tk::PlaceWindow . center')

# Load and resize Cisco logo
cisco_logo = Image.open("ciscologo.jpg")
cisco_logo = cisco_logo.resize((100, 70), Image.LANCZOS)  # Resize the logo to make it small
cisco_logo_photo = ImageTk.PhotoImage(cisco_logo)

# Add Cisco logo to the top-left corner
cisco_logo_label = tk.Label(root, image=cisco_logo_photo, bg="white")
cisco_logo_label.place(x=10, y=10)  # Position the logo at the top-left corner

# Greeting label with navy blue text
greeting_label1 = tk.Label(root, text="Hello! I'm Winter.", font=("Helvetica", 18, "bold"), bg="white", fg="#000000")  # First line with font size 18
greeting_label1.pack(pady=0)  # No extra padding for the first label

greeting_label2 = tk.Label(root, text="Let me help you with Catalyst Center", font=("Helvetica", 14), bg="white", fg="#000000")  # Second line with font size 14
greeting_label2.pack(pady=10)  # Extra padding for spacing between the two lines

# Add a logo below the greeting (replace 'hackathon_logo.png' with your image path)
logo_image = Image.open("hackathon_logo.png")
logo_image = logo_image.resize((250, 250), Image.LANCZOS)
logo_photo = ImageTk.PhotoImage(logo_image)
logo_label = tk.Label(root, image=logo_photo, bg="white")
logo_label.pack(pady=5)

# Set up the chat history display (smaller size)
chat_box = scrolledtext.ScrolledText(root, state='disabled', wrap=tk.WORD, width=50, height=10, bg="white", fg="#000080", font=("Helvetica", 12))  # Navy blue text
chat_box.pack(pady=5)
chat_box.tag_config("user", foreground="#000000")  # Black for user text
chat_box.tag_config("bot", foreground="#000000")  # Black for bot text

# Set up user input box frame
user_input_frame = tk.Frame(root, bg="white")
user_input_frame.pack(pady=5)

# Sample text for the input box
sample_text = "How can I help you today?"

# Create the input box with sample text
user_input = tk.Entry(user_input_frame, width=42, font=("Helvetica", 14), bg="white", fg="gray", insertbackground="#000080")
user_input.grid(row=0, column=0, padx=(0, 0), pady=5)

# Set the default text (placeholder)
user_input.insert(0, sample_text)

# Define behavior when the user clicks the input box
def on_click(event):
    if user_input.get() == sample_text:
        user_input.delete(0, tk.END)  # Clear the placeholder text when clicked
        user_input.config(fg="gray")  # Change text color to navy blue when typing

# Define behavior when the user leaves the input box
def on_focus_out(event):
    if user_input.get() == "":
        user_input.insert(0, sample_text)  # Insert the placeholder text if nothing is entered
        user_input.config(fg="gray")  # Change text color to gray to indicate placeholder text

# Bind events to the input box
user_input.bind("<FocusIn>", on_click)  # When the user clicks into the box, clear placeholder
user_input.bind("<FocusOut>", on_focus_out)  # When the user leaves the box, reset the placeholder if necessary

user_input.bind("<Return>", lambda event: get_response())  # When the user presses Enter, get response

# Set up user button frame
user_button_frame = tk.Frame(root, bg="white")
user_button_frame.pack(pady=10)

# Add a button to send the message
send_button = tk.Button(user_button_frame, text="SEND", command=get_response, font=("Helvetica", 12), bg="#FFFFE0", fg="#000000", width=10)
send_button.grid(row=0, column=0, padx=5, pady=5)  # Adjusted grid position

# Add a spacer between the buttons
spacer = tk.Label(user_button_frame, width=2, bg="white")  # Adjust 'width' to control spacing
spacer.grid(row=0, column=1)

# Add a button to clear the history
def clear_history():
    chat_box.config(state=tk.NORMAL)
    chat_box.delete(1.0, tk.END)
    chat_box.config(state=tk.DISABLED)

clear_button = tk.Button(user_button_frame, text="Clear History", command=clear_history, font=("Helvetica", 12), bg="#FFFFE0", fg="#000000", width=15)
clear_button.grid(row=0, column=2, padx=5, pady=5)  # Adjusted grid position

# Start the Tkinter event loop
root.mainloop()
