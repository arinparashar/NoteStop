import torch
import customtkinter as ctk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tkinter import filedialog
from tkinter import messagebox
from fpdf import FPDF
import pyperclip

# Load model + tokenizer
MODEL_PATH = "./tinyllama-trained"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")

# Theme switch
current_theme = "dark"

def toggle_theme():
    global current_theme
    current_theme = "light" if current_theme == "dark" else "dark"
    ctk.set_appearance_mode(current_theme)

# Generate Function
def generate_notes():
    topic = topic_entry.get().strip()
    task = task_var.get()

    if not topic or not task:
        output_box.delete("1.0", "end")
        output_box.insert("end", "⚠️ Please enter a topic and select a task.")
        return

    task_prompts = {
        "Summary": {
            "prompt": f"<s>[INST] Provide a detailed summary (around 200 words) of {topic} [/INST]",
            "max_tokens": 350,
            "temperature": 0.7
        },
        "Notes": {
            "prompt": f"<s>[INST] Provide concise notes in 5-6 bullet points for {topic}. Start each point with a dash (-) [/INST]",
            "max_tokens": 280,
            "temperature": 0.7
        },
        "Q&A": {
            "prompt": f"<s>[INST] Generate 1-2 line Q&A for {topic} [/INST]",
            "max_tokens": 120,
            "temperature": 0.6
        },
        "Cheatsheet": {
            "prompt": f"<s>[INST] Create a cheatsheet for {topic} with important keywords and key points [/INST]",
            "max_tokens": 300,
            "temperature": 0.7
        }
    }

    task_info = task_prompts[task]
    inputs = tokenizer(task_info["prompt"], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=task_info["max_tokens"],
        temperature=task_info["temperature"],
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.2
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "[/INST]" in decoded:
        result = decoded.split("[/INST]", 1)[1].strip()
    else:
        result = decoded.strip()

    output_box.delete("1.0", "end")
    output_box.insert("end", result)

# Copy Result
def copy_to_clipboard():
    text = output_box.get("1.0", "end").strip()
    if text:
        pyperclip.copy(text)
        messagebox.showinfo("Copied", "✅ Output copied to clipboard!")

# Export to Markdown
def export_markdown():
    text = output_box.get("1.0", "end").strip()
    if not text:
        messagebox.showwarning("No Content", "⚠️ Nothing to export.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".md", filetypes=[("Markdown files", "*.md")])
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        messagebox.showinfo("Saved", f"✅ Exported to {file_path}")

# Export to PDF
def export_pdf():
    text = output_box.get("1.0", "end").strip()
    if not text:
        messagebox.showwarning("No Content", "⚠️ Nothing to export.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
    if file_path:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in text.split("\n"):
            pdf.multi_cell(0, 10, line)
        pdf.output(file_path)
        messagebox.showinfo("Saved", f"✅ Exported to {file_path}")

# GUI Setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.title("NoteStop - One Stop Destination for Notes in ML/DL")
app.geometry("780x720")
app.resizable(False, False)

# Title
title_label = ctk.CTkLabel(app, text="📚 NoteStop", font=ctk.CTkFont(size=32, weight="bold"))
title_label.pack(pady=(20, 5))

subtitle_label = ctk.CTkLabel(app, text="Your AI-powered note assistant for Machine Learning & Deep Learning topics",
                              font=ctk.CTkFont(size=15))
subtitle_label.pack(pady=(0, 20))

# Entry
topic_entry = ctk.CTkEntry(app, width=520, height=40, placeholder_text="🔍 Enter your topic (e.g., Neural Networks)",
                           font=ctk.CTkFont(size=14))
topic_entry.pack(pady=10)

# Task Selection
task_var = ctk.StringVar(value="")
task_frame = ctk.CTkFrame(app, fg_color="#1a1a1a")
task_frame.pack(pady=10, padx=10)

ctk.CTkLabel(task_frame, text="🧠 Select Task:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=10, pady=10)

for i, t in enumerate(["Summary", "Notes", "Q&A", "Cheatsheet"]):
    btn = ctk.CTkRadioButton(task_frame, text=t, variable=task_var, value=t, font=ctk.CTkFont(size=13))
    btn.grid(row=0, column=i + 1, padx=10, pady=10)

# Generate Button
generate_btn = ctk.CTkButton(app, text="⚡ Generate", command=generate_notes, width=200, height=40,
                             font=ctk.CTkFont(size=15, weight="bold"))
generate_btn.pack(pady=20)

# Output Box
output_box = ctk.CTkTextbox(app, width=700, height=280, wrap="word", font=("Consolas", 13), corner_radius=10)
output_box.pack(pady=10)

# Export & Action Buttons
btn_frame = ctk.CTkFrame(app, fg_color="transparent")
btn_frame.pack(pady=10)

ctk.CTkButton(btn_frame, text="📋 Copy", command=copy_to_clipboard, width=140).grid(row=0, column=0, padx=10)
ctk.CTkButton(btn_frame, text="📄 Export as PDF", command=export_pdf, width=140).grid(row=0, column=1, padx=10)
ctk.CTkButton(btn_frame, text="📝 Export as Markdown", command=export_markdown, width=160).grid(row=0, column=2, padx=10)
ctk.CTkButton(btn_frame, text="🌓 Toggle Theme", command=toggle_theme, width=140).grid(row=0, column=3, padx=10)

# Footer
footer = ctk.CTkLabel(app,
                      text="👩‍💻 Made by Arin Parashar (RA2211003012058) & Kratika Dariyani (RA2211003012054)",
                      font=ctk.CTkFont(size=12))
footer.pack(pady=(10, 5))

# Run
app.mainloop()
