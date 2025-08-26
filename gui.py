import tkinter as tk
from tkinter import scrolledtext, messagebox, font
import joblib
import numpy as np
import threading
import os
from PIL import Image, ImageTk
import cairosvg
import io
from dotenv import load_dotenv 

import google.generativeai as genai

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# Function from our existing data processing script
from src.data_processing import get_smiles_from_name

# Constants 
MODEL_PATH = 'models/random_forest_dili_model.pkl'
WINDOW_TITLE = "HepaTox-AI Predictor"
WINDOW_GEOMETRY = "700x700" # Increased window size for Gemini content

# Configure Gemini API 
load_dotenv() 
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Warning: Could not configure Gemini API. AI features will be disabled. Error: {e}")
    genai = None 

def generate_fingerprint_and_svg(smiles):
    """
    Generates a Morgan Fingerprint and an SVG image from a SMILES string.
    """
    if not isinstance(smiles, str):
        return None, None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(200, 200)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return list(fp), svg
    else:
        return None, None

class DiliPredictorApp:
    """
    The main class for the Tkinter GUI application.
    """
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_GEOMETRY)
        self.root.configure(bg='#f8fafc')

        self.model = self.load_model()
        self.create_widgets()

    def load_model(self):
        """Loads the trained RandomForest model from the file."""
        try:
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
            return model
        except FileNotFoundError:
            messagebox.showerror("Error", f"Model file not found at '{MODEL_PATH}'.\nPlease run train.py first.")
            self.root.destroy()
            return None

    def create_widgets(self):
        """Creates and arranges all the GUI elements in the window."""
        main_frame = tk.Frame(self.root, padx=20, pady=20, bg='#f8fafc')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        tk.Label(main_frame, text="HepaTox-AI", font=("Arial", 24, "bold"), bg='#f8fafc', fg='#1e293b').pack(pady=(0, 5))
        tk.Label(main_frame, text="Drug-Induced Liver Injury (DILI) Predictor", font=("Arial", 12), bg='#f8fafc', fg='#64748b').pack(pady=(0, 20))

        # Input Section
        input_frame = tk.Frame(main_frame, bg='#f8fafc')
        input_frame.pack(fill=tk.X, pady=10)
        
        self.drug_name_entry = tk.Entry(input_frame, font=("Arial", 14), width=30, bg='#ffffff', fg='#1e293b', relief='solid', borderwidth=1)
        self.drug_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self.drug_name_entry.bind("<Return>", self.start_prediction_thread)

        self.predict_button = tk.Button(input_frame, text="Predict Risk", font=("Arial", 12, "bold"), 
                                        bg='#2563eb', fg='white', command=self.start_prediction_thread, relief='flat', padx=15, pady=5)
        self.predict_button.pack(side=tk.LEFT, padx=(10, 0))

        # Log Section
        self.log_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, font=("Courier New", 10), 
                                                      bg='#1e293b', fg='white', height=6, relief='solid', borderwidth=1)
        self.log_text.pack(fill=tk.X, pady=10)
        self.log_text.config(state='disabled')

        # Final Result Frame
        self.result_frame = tk.Frame(main_frame, bg='#ffffff', relief='solid', borderwidth=1)
        
        # Gemini Section
        self.gemini_frame = tk.Frame(main_frame, bg='#f1f5f9', relief='solid', borderwidth=1)
        self.gemini_label = tk.Label(self.gemini_frame, text="Google Gemini", font=("Arial", 12, "bold"), bg='#f1f5f9', fg='#475569')
        self.gemini_text = scrolledtext.ScrolledText(self.gemini_frame, wrap=tk.WORD, font=("Arial", 10), 
                                                     bg='white', height=8, relief='solid', borderwidth=0)

    def log_message(self, message, clear=False):
        """Appends a message to the log text box."""
        self.log_text.config(state='normal')
        if clear:
            self.log_text.delete('1.0', tk.END)
        self.log_text.insert(tk.END, f"> {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def start_prediction_thread(self, event=None):
        """Starts the prediction in a separate thread to prevent the GUI from freezing."""
        drug_name = self.drug_name_entry.get().strip()
        if not drug_name:
            messagebox.showwarning("Input Error", "Please enter a drug name.")
            return
        
        self.predict_button.config(state='disabled', text='Predicting...')
        self.result_frame.pack_forget()
        self.gemini_frame.pack_forget() 
        
        thread = threading.Thread(target=self.handle_prediction, args=(drug_name,))
        thread.start()

    def handle_prediction(self, drug_name):
        """The core prediction logic that runs in a background thread."""
        self.log_message(f"Starting prediction for: '{drug_name}'", clear=True)

        smiles = get_smiles_from_name(drug_name)
        if smiles is None:
            self.log_message(f"ERROR: Could not find a structure for '{drug_name}'.")
            self.reset_button()
            return
        self.log_message(f"SMILES Found: {smiles[:40]}...")

        fingerprint, svg_image = generate_fingerprint_and_svg(smiles)
        if fingerprint is None:
            self.log_message("ERROR: Could not generate a fingerprint.")
            self.reset_button()
            return
        self.log_message("Fingerprint and image generated successfully.")

        fingerprint_2d = np.array(fingerprint).reshape(1, -1)
        prediction = self.model.predict(fingerprint_2d)[0]
        probability = self.model.predict_proba(fingerprint_2d)[0][1]
        self.log_message("Prediction complete.")
        
        self.display_final_result(svg_image, prediction, probability, drug_name)
        self.reset_button()

    def display_final_result(self, svg_image, prediction, probability, drug_name):
        """Clears and rebuilds the result frame with the new prediction."""
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        png_data = cairosvg.svg2png(bytestring=svg_image.encode('utf-8'))
        image = Image.open(io.BytesIO(png_data)).resize((150, 150), Image.Resampling.LANCZOS)
        self.molecule_photo = ImageTk.PhotoImage(image)

        is_concern = (prediction == 1)
        bg_color = '#fee2e2' if is_concern else '#dcfce7'
        text_color = '#991b1b' if is_concern else '#166534'
        self.result_frame.config(bg=bg_color)

        img_label = tk.Label(self.result_frame, image=self.molecule_photo, bg=bg_color)
        img_label.pack(side=tk.LEFT, padx=20, pady=20)

        details_frame = tk.Frame(self.result_frame, bg=bg_color)
        details_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)

        result_text = "DILI Concern" if is_concern else "No DILI Concern"
        tk.Label(details_frame, text=result_text, font=("Arial", 18, "bold"), bg=bg_color, fg=text_color).pack(anchor='w')
        tk.Label(details_frame, text="Confidence Score", font=("Arial", 10), bg=bg_color, fg='#475569').pack(anchor='w', pady=(10, 0))
        tk.Label(details_frame, text=f"{probability:.1%}", font=("Arial", 28, "bold"), bg=bg_color, fg='#1e293b').pack(anchor='w')
        
        # Add Gemini button if API is configured
        if genai:
            gemini_button = tk.Button(details_frame, text="Learn More with Gemini", font=("Arial", 10), 
                                      bg="#928787", fg='white', relief='flat',
                                      command=lambda: self.start_gemini_thread(drug_name, prediction))
            gemini_button.pack(anchor='w', pady=(10,0))
        
        self.result_frame.pack(fill=tk.X, pady=10)

    def start_gemini_thread(self, drug_name, prediction):
        """Starts the Gemini API call in a new thread."""
        self.gemini_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.gemini_label.pack(pady=(10,5))
        self.gemini_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        self.gemini_text.config(state='normal')
        self.gemini_text.delete('1.0', tk.END)
        self.gemini_text.insert('1.0', "Thinking...")
        self.gemini_text.config(state='disabled')
        
        thread = threading.Thread(target=self.fetch_gemini_summary, args=(drug_name, prediction))
        thread.start()

    def fetch_gemini_summary(self, drug_name, prediction):
        """Fetches and displays the summary from the Gemini API."""
        prompt = f"Provide a brief, one-paragraph summary for a layperson explaining what the drug '{drug_name}' is typically used for."
        if prediction == 1:
            prompt += "\n\nAdditionally, this drug was flagged with a potential for Drug-Induced Liver Injury (DILI). In simple, easy-to-understand terms, briefly explain what DILI is and what the general next steps are for a patient if a doctor raises this concern. Do not provide medical advice. Structure this as a second paragraph titled 'About DILI'."

        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(prompt)
            
            self.gemini_text.config(state='normal')
            self.gemini_text.delete('1.0', tk.END)
            self.gemini_text.insert('1.0', response.text)
            self.gemini_text.config(state='disabled')
        except Exception as e:
            self.gemini_text.config(state='normal')
            self.gemini_text.delete('1.0', tk.END)
            self.gemini_text.insert('1.0', f"Error fetching information from Gemini:\n{e}")
            self.gemini_text.config(state='disabled')

    def reset_button(self):
        """Resets the predict button to its original state."""
        self.predict_button.config(state='normal', text='Predict Risk')

if __name__ == "__main__":
    if not genai:
        messagebox.showwarning("Gemini API Not Configured", 
                               "The GOOGLE_API_KEY was not found in your .env file. "
                               "The application will run without AI insight features.")
    root = tk.Tk()
    app = DiliPredictorApp(root)
    root.mainloop()