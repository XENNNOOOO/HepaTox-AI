import tkinter as tk
from tkinter import scrolledtext, messagebox, font
import joblib
import numpy as np
import threading
import os
from PIL import Image, ImageTk
import cairosvg
import io

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from src.data_processing import get_smiles_from_name

MODEL_PATH = 'models/random_forest_dili_model.pkl'
WINDOW_TITLE = "HepaTox-AI GUI"
WINDOW_GEOMETRY = "650x550"

# RDKit Helper Function
def generate_fingerprint_and_svg(smiles):
    """
    Generates a Morgan Fingerprint and an SVG image from a SMILES string.
    """
    if not isinstance(smiles, str):
        return None, None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Generate fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        
        # Generate SVG image
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
        self.root.configure(bg='#f8fafc') # Light gray background

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
        # Main Frame 
        main_frame = tk.Frame(self.root, padx=20, pady=20, bg='#f8fafc')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header 
        header_font = font.Font(family="Arial", size=24, weight="bold")
        subtitle_font = font.Font(family="Arial", size=12)
        tk.Label(main_frame, text="HepaTox-AI", font=header_font, bg='#f8fafc', fg='#1e293b').pack(pady=(0, 5))
        tk.Label(main_frame, text="Drug-Induced Liver Injury (DILI) Predictor", font=subtitle_font, bg='#f8fafc', fg='#64748b').pack(pady=(0, 20))

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
                                                      bg='#1e293b', fg='white', height=8, relief='solid', borderwidth=1)
        self.log_text.pack(fill=tk.X, pady=10)
        self.log_text.config(state='disabled')

        # Final Result Frame 
        self.result_frame = tk.Frame(main_frame, bg='#ffffff', relief='solid', borderwidth=1)
        # This frame is packed later when results are ready

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
        self.result_frame.pack_forget() # Hide previous results
        
        thread = threading.Thread(target=self.handle_prediction, args=(drug_name,))
        thread.start()

    def handle_prediction(self, drug_name):
        """The core prediction logic that runs in a background thread."""
        if self.model is None:
            self.log_message("ERROR: Model is not loaded.", clear=True)
            return

        self.log_message(f"Starting prediction for: '{drug_name}'", clear=True)

        # Get SMILES 
        self.log_message("Fetching molecular structure (SMILES)...")
        smiles = get_smiles_from_name(drug_name)
        if smiles is None:
            self.log_message(f"ERROR: Could not find a structure for '{drug_name}'.")
            self.reset_button()
            return
        self.log_message(f"SMILES Found: {smiles[:40]}...")

        # Generate Fingerprint and SVG Image
        self.log_message("Generating molecular fingerprint and image...")
        fingerprint, svg_image = generate_fingerprint_and_svg(smiles)
        if fingerprint is None:
            self.log_message("ERROR: Could not generate a fingerprint.")
            self.reset_button()
            return
        self.log_message("Fingerprint and image generated successfully.")

        # Make Prediction
        self.log_message("Making prediction with the model...")
        fingerprint_2d = np.array(fingerprint).reshape(1, -1)
        
        prediction = self.model.predict(fingerprint_2d)[0]
        probability = self.model.predict_proba(fingerprint_2d)[0][1]
        self.log_message("Prediction complete.")
        
        # Display Final Result
        self.display_final_result(svg_image, prediction, probability)

        self.reset_button()

    def display_final_result(self, svg_image, prediction, probability):
        """Clears and rebuilds the result frame with the new prediction."""
        # Clear any old widgets from the frame
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        # Convert SVG to a Tkinter-compatible image
        png_data = cairosvg.svg2png(bytestring=svg_image.encode('utf-8'))
        image = Image.open(io.BytesIO(png_data))
        image = image.resize((150, 150), Image.Resampling.LANCZOS)
        self.molecule_photo = ImageTk.PhotoImage(image)

        # Configure colors based on prediction
        is_concern = (prediction == 1)
        bg_color = '#fee2e2' if is_concern else '#dcfce7'
        text_color = '#991b1b' if is_concern else '#166534'
        self.result_frame.config(bg=bg_color)

        # Molecule Image
        img_label = tk.Label(self.result_frame, image=self.molecule_photo, bg=bg_color)
        img_label.pack(side=tk.LEFT, padx=20, pady=20)

        # Details Frame
        details_frame = tk.Frame(self.result_frame, bg=bg_color)
        details_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)

        result_text = "DILI Concern" if is_concern else "No DILI Concern"
        pred_font = font.Font(family="Arial", size=18, weight="bold")
        tk.Label(details_frame, text=result_text, font=pred_font, bg=bg_color, fg=text_color).pack(anchor='w')

        tk.Label(details_frame, text="Confidence Score", font=("Arial", 10), bg=bg_color, fg='#475569').pack(anchor='w', pady=(10, 0))
        
        conf_font = font.Font(family="Arial", size=28, weight="bold")
        tk.Label(details_frame, text=f"{probability:.1%}", font=conf_font, bg=bg_color, fg='#1e293b').pack(anchor='w')
        
        self.result_frame.pack(fill=tk.X, pady=10)


    def reset_button(self):
        """Resets the predict button to its original state."""
        self.predict_button.config(state='normal', text='Predict Risk')

if __name__ == "__main__":
    root = tk.Tk()
    app = DiliPredictorApp(root)
    root.mainloop()