import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind

# 1. Expand Dataset Suite
def load_datasets():
    # Breast Cancer (Binary)
    bc = load_breast_cancer()
    # Wine (Class 0 and 1 only for binary QNN)
    w = load_wine()
    w_mask = w.target < 2
    wine_data, wine_target = w.data[w_mask], w.target[w_mask]
    # Iris (Class 0 and 1 only)
    ir = load_iris()
    ir_mask = ir.target < 2
    iris_data, iris_target = ir.data[ir_mask], ir.target[ir_mask]
    
    return {
        "Breast Cancer": (bc.data, bc.target),
        "Wine": (wine_data, wine_target),
        "Iris": (iris_data, iris_target)
    }

def perform_journal_evaluation():
    datasets = load_datasets()
    summary_results = []

    for name, (X, y) in datasets.items():
        qnn_scores = []
        svm_scores = []
        
        print(f"\nEvaluating Dataset: {name}")
        
        # 2. Run 10-fold Seeded Iterations
        for seed in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
            
            # --- VQNN Logic (Simulated for structure, replace with your train_vqnn function) ---
            # Using LDA-based top 4 features as established in your best result
            q_acc = 0.93 + np.random.normal(0, 0.015) # Example placeholder
            qnn_scores.append(q_acc)
            
            # --- Classical SVM Baseline ---
            svm = SVC(kernel='rbf', C=1.0)
            svm.fit(X_train, y_train)
            svm_acc = accuracy_score(y_test, svm.predict(X_test))
            svm_scores.append(svm_acc)

        # 3. Statistical Significance (t-test)
        t_stat, p_val = ttest_ind(qnn_scores, svm_scores)
        
        summary_results.append({
            "Dataset": name,
            "VQNN_Mean": np.mean(qnn_scores),
            "VQNN_Std": np.std(qnn_scores),
            "SVM_Mean": np.mean(svm_scores),
            "P-Value": p_val
        })

    return pd.DataFrame(summary_results)

# Execute and output for the paper's Table 1
results_df = perform_journal_evaluation()
print("\n--- TABLE 1: MULTI-DATASET STATISTICAL COMPARISON ---")
print(results_df.to_string(index=False))