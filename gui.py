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
from rdkit.Chem import AllChem, Draw, Descriptors

MODEL_PATH = 'models/XGBoost22k_model.pkl'
WINDOW_TITLE = "HepaTox-AI"
WINDOW_GEOMETRY = "700x700"

# Configure Gemini API
load_dotenv() 
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Warning: Gemini API not configured. Error: {e}")
    genai = None

# Feature Engineering Functions
def get_smiles_from_name(compound_name):
    try:
        import cirpy
        return cirpy.resolve(compound_name, 'smiles')
    except Exception:
        return None

def generate_features_for_prediction(smiles):
    """
    Generates all necessary features (fingerprint, descriptors, SVG) for a single SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None, None, None

    # Fingerprint
    fingerprint = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

    # Descriptors (28)
    descriptors_list = [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol), Descriptors.NumRotatableBonds(mol),
        Descriptors.NOCount(mol), Descriptors.NumAliphaticCarbocycles(mol),
        Descriptors.NumAliphaticHeterocycles(mol), Descriptors.NumAliphaticRings(mol),
        Descriptors.NumAromaticCarbocycles(mol), Descriptors.NumAromaticHeterocycles(mol),
        Descriptors.NumAromaticRings(mol), Descriptors.NumSaturatedCarbocycles(mol),
        Descriptors.NumSaturatedHeterocycles(mol), Descriptors.NumSaturatedRings(mol),
        Descriptors.RingCount(mol), Descriptors.MolMR(mol), Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol), Descriptors.NHOHCount(mol), Descriptors.NOCount(mol),
        Descriptors.NumAliphaticRings(mol), Descriptors.NumAromaticRings(mol),
        Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol),
        Descriptors.NumHeteroatoms(mol), Descriptors.NumRotatableBonds(mol)
    ]
    
    # SVG Image
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(200, 200)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    return fingerprint, descriptors_list, svg

class DiliPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_GEOMETRY)
        self.root.configure(bg='#f8fafc')

        self.artifact = self.load_artifact()
        if self.artifact:
            self.model = self.artifact.get('model')
            self.scaler = self.artifact.get('scaler')
            if not self.model or not self.scaler:
                 messagebox.showerror("Model Error", f"The model file at '{MODEL_PATH}' is missing the model or scaler. Please re-run the training notebook.")
                 self.root.destroy()
                 return
        self.create_widgets()

    def load_artifact(self):
        try:
            artifact = joblib.load(MODEL_PATH)
            if not isinstance(artifact, dict):
                messagebox.showerror("Model Error", f"The model file at '{MODEL_PATH}' is outdated or in the wrong format. Please re-run notebook 11 to generate the correct file.")
                self.root.destroy()
                return None
            print("Model and scaler loaded successfully.")
            return artifact
        except FileNotFoundError:
            messagebox.showerror("Error", f"Model file not found at '{MODEL_PATH}'.\nPlease run notebook 11 first.")
            self.root.destroy()
            return None

    def create_widgets(self):
        main_frame = tk.Frame(self.root, padx=20, pady=20, bg='#f8fafc')
        main_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(main_frame, text="HepaTox-AI", font=("Arial", 24, "bold"), bg='#f8fafc', fg='#1e293b').pack(pady=(0, 5))
        tk.Label(main_frame, text="Drug-Induced Liver Injury (DILI) Predictor", font=("Arial", 12), bg='#f8fafc', fg='#64748b').pack(pady=(0, 20))
        input_frame = tk.Frame(main_frame, bg='#f8fafc')
        input_frame.pack(fill=tk.X, pady=10)
        self.drug_name_entry = tk.Entry(input_frame, font=("Arial", 14), width=30, bg='#ffffff', fg='#1e293b', relief='solid', borderwidth=1)
        self.drug_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self.drug_name_entry.bind("<Return>", self.start_prediction_thread)
        self.predict_button = tk.Button(input_frame, text="Predict Risk", font=("Arial", 12, "bold"), bg='#2563eb', fg='white', command=self.start_prediction_thread, relief='flat', padx=15, pady=5)
        self.predict_button.pack(side=tk.LEFT, padx=(10, 0))
        self.log_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, font=("Courier New", 10), bg='#1e293b', fg='white', height=6, relief='solid', borderwidth=1)
        self.log_text.pack(fill=tk.X, pady=10)
        self.log_text.config(state='disabled')
        self.result_frame = tk.Frame(main_frame, bg='#ffffff', relief='solid', borderwidth=1)
        self.gemini_frame = tk.Frame(main_frame, bg='#f1f5f9', relief='solid', borderwidth=1)
        self.gemini_label = tk.Label(self.gemini_frame, text="Google Gemini", font=("Arial", 12, "bold"), bg='#f1f5f9', fg='#475569')
        self.gemini_text = scrolledtext.ScrolledText(self.gemini_frame, wrap=tk.WORD, font=("Arial", 10), bg='white', height=8, relief='solid', borderwidth=0)

    def log_message(self, message, clear=False):
        self.log_text.config(state='normal')
        if clear: self.log_text.delete('1.0', tk.END)
        self.log_text.insert(tk.END, f"> {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def start_prediction_thread(self, event=None):
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
        if not self.artifact:
            self.log_message("ERROR: Model artifact not loaded.", clear=True)
            return

        self.log_message(f"Starting prediction for: '{drug_name}'", clear=True)
        smiles = get_smiles_from_name(drug_name)
        if smiles is None:
            self.log_message(f"ERROR: Could not find a structure for '{drug_name}'.")
            self.reset_button()
            return
        self.log_message(f"SMILES Found: {smiles[:40]}...")

        fingerprint, descriptors, svg_image = generate_features_for_prediction(smiles)
        if fingerprint is None:
            self.log_message("ERROR: Could not generate features.")
            self.reset_button()
            return
        self.log_message("Fingerprint and descriptors generated.")

        descriptors_scaled = self.scaler.transform(np.array(descriptors).reshape(1, -1))
        self.log_message("Descriptors scaled successfully.")

        fingerprint_np = np.array(fingerprint).reshape(1, -1)
        X_final = np.concatenate([fingerprint_np, descriptors_scaled], axis=1)
        self.log_message(f"Final feature vector created with shape: {X_final.shape}")
        
        prediction = self.model.predict(X_final)[0]
        probability = self.model.predict_proba(X_final)[0][1]
        self.log_message("Prediction complete.")
        
        self.display_final_result(svg_image, prediction, probability, drug_name)
        self.reset_button()

    def display_final_result(self, svg_image, prediction, probability, drug_name):
        for widget in self.result_frame.winfo_children(): widget.destroy()
        png_data = cairosvg.svg2png(bytestring=svg_image.encode('utf-8'))
        image = Image.open(io.BytesIO(png_data)).resize((150, 150), Image.Resampling.LANCZOS)
        self.molecule_photo = ImageTk.PhotoImage(image)
        is_concern = (prediction == 1)
        bg_color, text_color = ('#fee2e2', '#991b1b') if is_concern else ('#dcfce7', '#166534')
        self.result_frame.config(bg=bg_color)
        img_label = tk.Label(self.result_frame, image=self.molecule_photo, bg=bg_color)
        img_label.pack(side=tk.LEFT, padx=20, pady=20)
        details_frame = tk.Frame(self.result_frame, bg=bg_color)
        details_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)
        result_text = "DILI Concern" if is_concern else "No DILI Concern"
        tk.Label(details_frame, text=result_text, font=("Arial", 18, "bold"), bg=bg_color, fg=text_color).pack(anchor='w')
        tk.Label(details_frame, text="Confidence Score", font=("Arial", 10), bg=bg_color, fg='#475569').pack(anchor='w', pady=(10, 0))
        tk.Label(details_frame, text=f"{probability:.1%}", font=("Arial", 28, "bold"), bg=bg_color, fg='#1e293b').pack(anchor='w')
        if genai:
            gemini_button = tk.Button(details_frame, text="Learn More with Gemini", font=("Arial", 10), bg='#7c3aed', fg='white', relief='flat', command=lambda: self.start_gemini_thread(drug_name, prediction))
            gemini_button.pack(anchor='w', pady=(10,0))
        self.result_frame.pack(fill=tk.X, pady=10)

    def start_gemini_thread(self, drug_name, prediction):
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
        self.predict_button.config(state='normal', text='Predict Risk')

if __name__ == "__main__":
    if not genai:
        messagebox.showwarning("Gemini API Not Configured", "The GOOGLE_API_KEY was not found. AI features will be disabled.")
    root = tk.Tk()
    app = DiliPredictorApp(root)
    root.mainloop()