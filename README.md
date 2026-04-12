

---

# Adaptive Feature-to-Qubit Mapping for High-Dimensional Data Discrimination Using Variational Quantum Neural Circuits

## 1. Project Motivation & Context
The primary challenge in Quantum Machine Learning (QML) on Near-Intermediate Scale Quantum (NISQ) devices is the "Qubit Bottleneck." Most real-world datasets (like Breast Cancer diagnostic data) have dozens of features, but current quantum simulators and hardware are limited in qubit count. 

This project implements an **Adaptive Feature-to-Qubit Mapping** strategy. It allows high-dimensional classical data to be intelligently compressed and mapped onto a **4-qubit** Variational Quantum Neural Network (VQNN). This is done to achieve maximum class discrimination while maintaining the theoretical trainability of the quantum circuit.

---

## 2. Hybrid Methodology
The core of this project is the **Hybrid LDA-PCA** mapping approach:
* **Linear Discriminant Analysis (LDA):** One component is extracted to provide maximum linear separation between classes.
* **Principal Component Analysis (PCA):** Three additional components are extracted to capture the maximum structural variance of the data.
* **Mapping:** These 4 components are then mapped to 4 qubits using $R_y$ angle encoding.
* **The VQNN:** A variational circuit consisting of $R_y/R_z$ gates and a CNOT entanglement ladder is trained to classify the data.

---

## 3. Folder Structure
Based on the project `tree`, the files are organized as follows:

* **`New_Updated_Version/`**: **CRITICAL.** Contains the mathematically rigorous and standalone scripts for the final report.
    * `feature_selection_comparison.py`: The master multi-dataset evaluation engine.
    * `stat_rigor_eval.py`: Compares VQNN against a classical SVM baseline.
    * `Barren_Plateau_Analysis.py`: Checks for vanishing gradients via parameter-shift variance.
* **`outputs/`**: Contains the visualization gallery (Radar charts, heatmaps, and accuracy bars).
* **`Journal_Ready_Package/`**: Optimized assets specifically formatted for paper submission.

---

## 4. Execution Workflow (Order of Operations)

To replicate the journal-ready results, execute the scripts in the `New_Updated_Version/` folder in this specific order:

### **Step 1: Primary Performance Evaluation**
```bash
python New_Updated_Version/feature_selection_comparison.py
```
* **What it does:** Runs 5-fold cross-validation across Breast Cancer, Wine, and Iris datasets.
* **Output:** Generates `feature_selection_results_REAL.csv`.

### **Step 2: Statistical Rigor Testing**
```bash
python New_Updated_Version/stat_rigor_eval.py
```
* **What it does:** Performs 10 independent seeded runs to compare the VQNN against a Classical SVM.
* **Output:** Generates `journal_statistical_summary.csv` and calculates the P-value for scientific significance.

### **Step 3: Theoretical Trainability (Barren Plateau)**
```bash
python New_Updated_Version/Barren_Plateau_Analysis.py
```
* **What it does:** Samples gradients at random points in the parameter space to ensure gradients do not vanish as qubits scale.
* **Output:** Generates `barren_plateau_results.csv` and the corresponding proof plot.

---

## 5. Summary of Key Findings
* **Classification Performance:** The VQNN achieved a mean accuracy of **~75.07%** on the Breast Cancer dataset using the Hybrid LDA-PCA mapping.
* **Statistical Comparison:** The difference between VQNN and classical SVM remains statistically significant (P < 0.05), highlighting current NISQ-era optimization challenges.
* **Gradient Health:** Barren Plateau analysis confirmed a gradient variance **> 0.2** for up to 8 qubits, proving that the specific $R_y/R_z$ ladder architecture remains trainable.

---

