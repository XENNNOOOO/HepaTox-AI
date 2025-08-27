# HepaTox: Predicting Drug-Induced Liver Injury

An AI-powered toolkit for predicting Drug-Induced Liver Injury (DILI) using machine learning and graph neural networks.

---

## Introduction

One of the most significant challenges in pharmacology is predicting adverse drug reactions before a new medication reaches patients. Among the most severe and common reasons for a drug to fail in clinical trials or be withdrawn from the market is **Drug-Induced Liver Injury (DILI)**. The liver, as the body's primary metabolic engine, is uniquely vulnerable to damage from new chemical compounds. Predicting DILI early is a critical mission that can save development costs, accelerate the creation of safer medicines, and, most importantly, protect patient health.

This project, **HepaTox**, was undertaken to address this challenge by building and comparing machine learning models that can predict a drug's potential to cause liver injury based solely on its molecular structure.

---

## The DILIrank Dataset

The foundation of our model is the **DILIrank Dataset**, a publicly available, expert-curated resource provided by the U.S. Food and Drug Administration (FDA). This dataset is considered a gold standard for DILI research.

### Background

The DILIrank dataset provides a comprehensive list of FDA-approved drugs, each assigned a DILI severity and concern classification based on a thorough evaluation of clinical data and published literature. This expert labeling is what makes it possible to train a supervised machine learning model. The dataset contains 1,036 drug records, which, due to their high quality and information density, are sufficient for building a robust predictive model.

### Dataset Reference

* **Source:** U.S. Food and Drug Administration (FDA)
* **Homepage:** [Drug-Induced Liver Injury Rank (DILIrank) Dataset](https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-dataset)
* **Citation:** Chen, M., Suzuki, A., Thakkar, S. et al. DILIrank: the FDA-approved drug database for ranking drug-induced liver injury severity. *Hepatology* 64, 1579–1582 (2016).

---

## Project Execution & Findings

The development followed a rigorous, iterative process to find the best possible model.

### Phase 1: Baseline Models

Strong baselines were established using powerful, feature-based models trained on **molecular fingerprints**.

* **RandomForest:** The initial champion model proved to be highly effective, achieving a robust ROC AUC score of **0.761**.
* **XGBoost:** After resolving environment constraints, an XGBoost model was tested and fine-tuned. Through advanced hyperparameter tuning with Optuna, it achieved a slightly better ROC AUC of **0.765**, becoming the new baseline.

### Phase 2: Graph Neural Network (GNN) Exploration

Next, advanced GNN architectures were explored to determine if learning directly from the 2D molecular graph structure could improve performance.

* **GIN & Deeper GIN:** A Graph Isomorphism Network (GIN) was implemented. While it learned successfully, it did not surpass the baselines. By increasing the model's depth to four layers, its performance improved to a respectable ROC AUC of **0.739**.
* **AttentiveFP & PNA:** More complex, state-of-the-art architectures were tested. While these models trained successfully, they did not outperform the simpler GNN or the feature-based models on this specific dataset, highlighting that more complexity is not always better.

### Phase 3: Final Ensemble Model

The final experiment involved creating a **"committee of experts"** by ensembling the two best-performing models: the tuned RandomForest and the tuned XGBoost. By averaging their predictions, the individual errors were canceled out, creating a final model that was more robust and accurate than either of its components.

---

## Final Results & Conclusion

After a thorough and systematic comparison, the conclusion is clear:

| Metric      | RandomForest | Tuned XGBoost | Deeper GIN | **Ensemble Model** |
| :---------- | :----------- | :------------ | :--------- | :----------------- |
| **ROC AUC** | 0.761        | 0.765         | 0.739      | **0.768** |

The **Ensemble Model** is the winning architecture. This project demonstrates the power of a systematic, iterative approach. By establishing strong baselines, exploring advanced architectures, and finally combining the strengths of our best models, we successfully built a state-of-the-art predictor that represents the culmination of our research.

---

## How to Use the Final Model

The project has been structured into a reusable command-line tool.

**1. Install Dependencies:**
Ensure all required libraries are installed by running:

pip install -r requirements.txt

**2. Process Raw Data:**
Run the data processing script to generate the clean data.

python3 src/data_processing.py

**3. Train the Final Model:**
Run the training script to train the winning RandomForest model on the full dataset and save it.

python3 src/train.py

**4. Launch the GUI Predictor:**
Run the gui.py script to open the application. Enter a drug name and click "Predict Risk" to see the results.

python3 gui.py

### Technologies Used
- Core Language: Python
- Data Handling: Pandas, NumPy
- Chemoinformatics: RDKit, cirpy
- Machine Learning: Scikit-learn, XGBoost
- Deep Learning: PyTorch, PyTorch Geometric
- GUI: Tkinter, Pillow, CairoSVG
- Hyperparameter Tuning: Optuna
- LLM Integration: Google Gemini API