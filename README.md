# HepaTox: Predicting Drug-Induced Liver Injury

An AI-powered toolkit for predicting Drug-Induced Liver Injury (DILI) using machine learning and graph neural networks.

---

## Introduction

One of the most significant challenges in pharmacology is predicting adverse drug reactions before a new medication reaches patients. Among the most severe and common reasons for a drug to fail in clinical trials or be withdrawn from the market is **Drug-Induced Liver Injury (DILI)**. The liver, as the body's primary metabolic engine, is uniquely vulnerable to damage from new chemical compounds. Predicting DILI early is a critical mission that can save development costs, accelerate the creation of safer medicines, and, most importantly, protect patient health.

This project, **HepaTox**, was undertaken to address this challenge by building and comparing machine learning models that can predict a drug's potential to cause liver injury based solely on its molecular structure.

---

## The DILIrank Dataset

This project utilized two key datasets to progressively build a state-of-the-art model.

### Dataset Reference
1. DILIRank Dataset

The initial foundation of the model was the DILIrank Dataset, a publicly available, expert-curated resource from the U.S. Food and Drug Administration (FDA). This dataset, containing over 1,000 compounds, was used to establish strong baseline models.
* **Source:** U.S. Food and Drug Administration (FDA)
* **Homepage:** [Drug-Induced Liver Injury Rank (DILIrank) Dataset](https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-dataset)
* **Citation:** Chen, M., Suzuki, A., Thakkar, S. et al. DILIrank: the FDA-approved drug database for ranking drug-induced liver injury severity. *Hepatology* 64, 1579–1582 (2016).

2. InterDILI Dataset

To achieve state-of-the-art performance, the project transitioned to the interDILI Dataset, a large-scale database from recent academic research, which was sourced for this project. Containing over 22,000 compounds, this larger dataset was crucial for training a more accurate and generalizable final model.
* **Source:** BMC Journal of Cheminformatics
* **Homepage:** https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00796-8
* **Citation:** Lee, S., Yoo, S. InterDILI: interpretable prediction of drug-induced liver injury through permutation feature importance and attention mechanism. J Cheminform 16, 1 (2024). https://doi.org/10.1186/s13321-023-00796-8

---

## Project Execution & Findings

The development followed a rigorous, iterative process to find the best possible model.

### Phase 1: Baseline Models

Strong baselines were established using powerful, feature-based models trained on **molecular fingerprints**.

* **RandomForest:** The initial champion model proved to be highly effective, achieving a robust ROC AUC score of **0.761**.
* **XGBoost:** After resolving environment constraints, an XGBoost model was tested and fine-tuned. Through advanced hyperparameter tuning with Optuna, it achieved a slightly better ROC AUC of **0.765**, becoming the new baseline.

### Phase 2: Graph Neural Network (GNN) Exploration

Advanced GNN architectures (GIN, PNA) were explored to determine if learning directly from the 2D molecular graph structure could improve performance. While these models learned successfully, they did not surpass the feature-based baselines on the smaller DILIrank dataset.

### Phase 3: Final Ensemble Model
The final and most successful phase involved leveraging the large-scale dataset.

Feature Engineering: A comprehensive set of 28 physicochemical properties was engineered for each of the ~1,800 molecules used from the dataset.

Hybrid Feature Set: These properties were combined with the molecular fingerprints to create a rich, hybrid feature set.

Final Model: The champion XGBoost model was trained on this new, enriched dataset, resulting in a significant performance breakthrough.

---

## Results & Conclusion

After a thorough and systematic comparison, the conclusion is clear:

| Metric      | RandomForest | Tuned XGBoost | Ensemble Model | **XGBoost22k Model** |
| :---------- | :----------- | :------------ | :--------- | :----------------- |
| **ROC AUC** | 0.761        | 0.765         | 0.768      | **0.849** |

XGBoost22k Model Accuracy: **0.795**

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