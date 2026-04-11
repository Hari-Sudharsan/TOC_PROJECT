"""
feature_selection_comparison.py
================================
Compares 10 Feature Selection Methods for the VQNN project:
  1.  PCA           – Principal Component Analysis
  2.  LDA           – Linear Discriminant Analysis
  3.  Kernel PCA    – Nonlinear PCA (RBF kernel)
  4.  ICA           – Independent Component Analysis
  5.  SVD           – Truncated Singular Value Decomposition
  6.  SelectKBest (ANOVA F)   – Univariate Statistical Test
  7.  SelectKBest (MI)        – Mutual Information
  8.  RFE (Random Forest)     – Recursive Feature Elimination
  9.  Random Forest Importance – Tree-based importance ranking
  10. Extra Trees Importance   – Extremely Randomized Trees

For each method:
  → Selects top-4 features (one per qubit)
  → Trains the same VQNN
  → Evaluates: Accuracy, F1, AUC, MCC
  → Generates individual + combined comparison figures
"""

import os, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA, FastICA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import (SelectKBest, mutual_info_classif,
                                       f_classif, RFE)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, matthews_corrcoef)
warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Output dirs ───────────────────────────────────────────────────────
OUT  = "./outputs"
os.makedirs(OUT, exist_ok=True)

# ── Dark theme ────────────────────────────────────────────────────────
PAL = {
    "bg":"#0d1117","panel":"#161b22","border":"#21262d",
    "accent":"#58a6ff","green":"#3fb950","red":"#f85149",
    "yellow":"#d29922","purple":"#bc8cff","orange":"#f0883e",
    "cyan":"#39d353","pink":"#ff7b72","text":"#e6edf3","muted":"#8b949e",
}
plt.rcParams.update({
    "figure.facecolor":PAL["bg"],"axes.facecolor":PAL["panel"],
    "axes.edgecolor":PAL["border"],"axes.labelcolor":PAL["text"],
    "xtick.color":PAL["muted"],"ytick.color":PAL["muted"],
    "text.color":PAL["text"],"grid.color":PAL["border"],
    "grid.linestyle":"--","font.family":"monospace",
    "legend.facecolor":PAL["panel"],"legend.edgecolor":PAL["border"],
})

# Colors per method (10 distinct)
METHOD_COLORS = [
    PAL["accent"], PAL["green"], PAL["red"],    PAL["yellow"],
    PAL["purple"], PAL["orange"],PAL["cyan"],   PAL["pink"],
    "#a5d6ff",     "#7ee787",
]

# ── Quantum simulator (same as main project) ──────────────────────────
def Ry(t): c,s=np.cos(t/2),np.sin(t/2); return np.array([[c,-s],[s,c]],dtype=complex)
def Rz(t): e=np.exp(1j*t/2); return np.array([[np.conj(e),0],[0,e]],dtype=complex)

def apply_gate(state, gate, q, n):
    ops=[np.eye(2,dtype=complex)]*n; ops[q]=gate
    full=ops[0]
    for op in ops[1:]: full=np.kron(full,op)
    return full@state

def cnot(state,ctrl,tgt,n):
    dim=2**n; mat=np.eye(dim,dtype=complex)
    for i in range(dim):
        bits=[(i>>(n-1-k))&1 for k in range(n)]
        if bits[ctrl]==1:
            bits[tgt]^=1; j=sum(b<<(n-1-k) for k,b in enumerate(bits))
            mat[i,i]=0; mat[j,i]=1
    return mat@state

def expval_Z(state,n):
    dim=2**n; ev=0.0
    for i in range(dim):
        bit=(i>>(n-1))&1; ev+=(1 if bit==0 else -1)*abs(state[i])**2
    return float(ev)

def vqnn_circuit(x, params, n):
    state=np.zeros(2**n,dtype=complex); state[0]=1.0
    for q in range(n): state=apply_gate(state,Ry(float(x[q])),q,n)
    for l in range(params.shape[0]):
        for q in range(n):
            state=apply_gate(state,Ry(float(params[l,q,0])),q,n)
            state=apply_gate(state,Rz(float(params[l,q,1])),q,n)
        for q in range(n-1): state=cnot(state,q,q+1,n)
    nm=np.linalg.norm(state)
    if nm>0: state/=nm
    return expval_Z(state,n)

def batch_predict(X,params,n):
    return np.array([vqnn_circuit(x,params,n) for x in X])

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
def classify(r): return (sigmoid(r)>=0.5).astype(int)
def bce(y,r):
    p=np.clip(sigmoid(r),1e-7,1-1e-7)
    return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))

def param_shift_grad(Xb,yb,params,n,eps=np.pi/2):
    grad=np.zeros_like(params)
    for idx in np.ndindex(params.shape):
        p1=params.copy(); p1[idx]+=eps
        p2=params.copy(); p2[idx]-=eps
        grad[idx]=(bce(yb,batch_predict(Xb,p1,n))-bce(yb,batch_predict(Xb,p2,n)))/2
    return grad

def train_vqnn(X_tr,y_tr,X_te,y_te,n_qubits=4,n_layers=2,
               epochs=40,lr=0.05,batch_size=16):
    params=np.random.uniform(-np.pi,np.pi,(n_layers,n_qubits,2))
    m=np.zeros_like(params); v=np.zeros_like(params); t=0
    b1,b2,eps=0.9,0.999,1e-8
    best_acc=-1; best_params=params.copy(); patience=0

    for ep in range(1,epochs+1):
        idx=np.random.choice(len(X_tr),min(batch_size,len(X_tr)),replace=False)
        grad=param_shift_grad(X_tr[idx],y_tr[idx],params,n_qubits)
        t+=1; m=b1*m+(1-b1)*grad; v=b2*v+(1-b2)*grad**2
        mh=m/(1-b1**t); vh=v/(1-b2**t)
        params-=lr*mh/(np.sqrt(vh)+eps)
        val_acc=accuracy_score(y_te,classify(batch_predict(X_te,params,n_qubits)))
        if val_acc>best_acc: best_acc=val_acc; best_params=params.copy(); patience=0
        else: patience+=1
        if patience>=12: break

    raw=batch_predict(X_te,best_params,n_qubits)
    yp=classify(raw)
    return {
        "accuracy": accuracy_score(y_te,yp),
        "f1":       f1_score(y_te,yp,zero_division=0),
        "auc":      roc_auc_score(y_te,sigmoid(raw)),
        "mcc":      matthews_corrcoef(y_te,yp),
        "cm":       confusion_matrix(y_te,yp),
    }

def angle_scale(X):
    mn,mx=X.min(0),X.max(0)
    rng=np.where(mx-mn==0,1,mx-mn)
    return (X-mn)/rng*np.pi

# ─────────────────────────────────────────────────────────────────────
# LOAD & PREP DATA
# ─────────────────────────────────────────────────────────────────────
print("Loading Breast Cancer dataset...")
data=load_breast_cancer()
X_raw=data.data.astype(float); y=data.target

scaler=StandardScaler()
X_sc=scaler.fit_transform(X_raw)

minmax=MinMaxScaler()
X_mm=minmax.fit_transform(X_raw)  # for chi2 (needs non-negative)

X_tr_sc,X_te_sc,y_tr,y_te=train_test_split(X_sc,y,test_size=0.25,random_state=42,stratify=y)
X_tr_mm,X_te_mm,_,_=train_test_split(X_mm,y,test_size=0.25,random_state=42,stratify=y)

N_FEATURES=4   # one per qubit

# ─────────────────────────────────────────────────────────────────────
# DEFINE ALL 10 FEATURE SELECTION METHODS
# ─────────────────────────────────────────────────────────────────────
def apply_method(name, X_tr, X_te, y_tr):
    """
    Returns (X_tr_reduced, X_te_reduced, extra_info_dict)
    All outputs have shape (n_samples, N_FEATURES)
    """
    info = {}

    if name == "PCA":
        model = PCA(n_components=N_FEATURES, random_state=42)
        Xr_tr = model.fit_transform(X_tr)
        Xr_te = model.transform(X_te)
        info["variance_retained"] = f"{model.explained_variance_ratio_.sum()*100:.1f}%"
        info["selected_features"] = "PC1-PC4 (linear combinations)"

    elif name == "LDA":
        # LDA max components = min(n_classes-1, n_features) = 1 for binary
        # We run LDA once then pad to N_FEATURES with PCA components
        lda = LDA(n_components=1)
        Xr_lda_tr = lda.fit_transform(X_tr, y_tr)
        Xr_lda_te = lda.transform(X_te)
        pca = PCA(n_components=N_FEATURES-1, random_state=42)
        Xr_pca_tr = pca.fit_transform(X_tr)
        Xr_pca_te = pca.transform(X_te)
        Xr_tr = np.hstack([Xr_lda_tr, Xr_pca_tr])
        Xr_te = np.hstack([Xr_lda_te, Xr_pca_te])
        info["note"] = "LDA (1 component) + PCA (3 components)"

    elif name == "Kernel PCA":
        model = KernelPCA(n_components=N_FEATURES, kernel="rbf",
                          gamma=0.01, random_state=42)
        Xr_tr = model.fit_transform(X_tr)
        Xr_te = model.transform(X_te)
        info["kernel"] = "RBF (γ=0.01)"

    elif name == "ICA":
        model = FastICA(n_components=N_FEATURES, random_state=42, max_iter=500)
        Xr_tr = model.fit_transform(X_tr)
        Xr_te = model.transform(X_te)
        info["note"] = "Independent Component Analysis"

    elif name == "Truncated SVD":
        model = TruncatedSVD(n_components=N_FEATURES, random_state=42)
        Xr_tr = model.fit_transform(X_tr)
        Xr_te = model.transform(X_te)
        info["variance_retained"] = f"{model.explained_variance_ratio_.sum()*100:.1f}%"

    elif name == "SelectKBest (ANOVA F)":
        model = SelectKBest(f_classif, k=N_FEATURES)
        Xr_tr = model.fit_transform(X_tr, y_tr)
        Xr_te = model.transform(X_te)
        selected = np.where(model.get_support())[0]
        info["selected_features"] = [data.feature_names[i] for i in selected]
        info["f_scores"] = model.scores_[selected].tolist()

    elif name == "SelectKBest (MI)":
        model = SelectKBest(mutual_info_classif, k=N_FEATURES)
        Xr_tr = model.fit_transform(X_tr, y_tr)
        Xr_te = model.transform(X_te)
        selected = np.where(model.get_support())[0]
        info["selected_features"] = [data.feature_names[i] for i in selected]
        info["mi_scores"] = model.scores_[selected].tolist()

    elif name == "RFE (Random Forest)":
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        model = RFE(estimator, n_features_to_select=N_FEATURES, step=2)
        Xr_tr = model.fit_transform(X_tr, y_tr)
        Xr_te = model.transform(X_te)
        selected = np.where(model.get_support())[0]
        info["selected_features"] = [data.feature_names[i] for i in selected]
        info["ranking"] = model.ranking_[selected].tolist()

    elif name == "Random Forest Importance":
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_tr, y_tr)
        top_k = np.argsort(rf.feature_importances_)[::-1][:N_FEATURES]
        Xr_tr = X_tr[:, top_k]
        Xr_te = X_te[:, top_k]
        info["selected_features"] = [data.feature_names[i] for i in top_k]
        info["importances"] = rf.feature_importances_[top_k].tolist()

    elif name == "Extra Trees Importance":
        et = ExtraTreesClassifier(n_estimators=100, random_state=42)
        et.fit(X_tr, y_tr)
        top_k = np.argsort(et.feature_importances_)[::-1][:N_FEATURES]
        Xr_tr = X_tr[:, top_k]
        Xr_te = X_te[:, top_k]
        info["selected_features"] = [data.feature_names[i] for i in top_k]
        info["importances"] = et.feature_importances_[top_k].tolist()

    return angle_scale(Xr_tr), angle_scale(Xr_te), info

# ─────────────────────────────────────────────────────────────────────
# RUN ALL METHODS
# ─────────────────────────────────────────────────────────────────────
METHODS = [
    "PCA",
    "LDA",
    "Kernel PCA",
    "ICA",
    "Truncated SVD",
    "SelectKBest (ANOVA F)",
    "SelectKBest (MI)",
    "RFE (Random Forest)",
    "Random Forest Importance",
    "Extra Trees Importance",
]

all_results   = []
method_details = {}

print("\n" + "="*60)
print("  RUNNING ALL 10 FEATURE SELECTION METHODS + VQNN")
print("="*60)

for i, method in enumerate(METHODS):
    print(f"\n[{i+1:2d}/10] {method}")
    t0 = time.time()

    # Apply feature selection
    Xr_tr, Xr_te, info = apply_method(
        method, X_tr_sc, X_te_sc, y_tr)

    # Train VQNN
    np.random.seed(42)
    metrics = train_vqnn(Xr_tr, y_tr, Xr_te, y_te,
                         n_qubits=4, n_layers=2,
                         epochs=40, lr=0.05, batch_size=16)
    elapsed = time.time() - t0

    print(f"     Acc={metrics['accuracy']*100:.2f}%  "
          f"F1={metrics['f1']:.3f}  "
          f"AUC={metrics['auc']:.3f}  "
          f"MCC={metrics['mcc']:.3f}  "
          f"({elapsed:.0f}s)")

    result = {
        "Method":   method,
        "Accuracy": round(metrics["accuracy"]*100, 2),
        "F1":       round(metrics["f1"],  3),
        "AUC":      round(metrics["auc"], 3),
        "MCC":      round(metrics["mcc"], 3),
        "Time_s":   round(elapsed, 1),
        "CM":       metrics["cm"],
        "info":     info,
    }
    all_results.append(result)
    method_details[method] = info

# Sort by accuracy
all_results.sort(key=lambda r: r["Accuracy"], reverse=True)
BEST = all_results[0]["Method"]

print(f"\n\n  ★ BEST METHOD: {BEST} "
      f"({all_results[0]['Accuracy']}% Accuracy)\n")

# ─────────────────────────────────────────────────────────────────────
# SAVE RESULTS CSV
# ─────────────────────────────────────────────────────────────────────
df_res = pd.DataFrame([{k:v for k,v in r.items()
                         if k not in ("CM","info")}
                        for r in all_results])
df_res.to_csv(os.path.join(OUT,"feature_selection_results.csv"), index=False)
print("  Saved: feature_selection_results.csv")

# ─────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────
def save(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, dpi=160, bbox_inches="tight",
                facecolor=PAL["bg"], edgecolor="none")
    plt.close()
    print(f"  Saved: {name}")

# ─────────────────────────────────────────────────────────────────────
# FIGURE A – Master Accuracy Bar Chart (all 10 methods)
# ─────────────────────────────────────────────────────────────────────
methods_sorted = [r["Method"] for r in all_results]
accs_sorted    = [r["Accuracy"] for r in all_results]
colors_sorted  = [METHOD_COLORS[METHODS.index(m)] for m in methods_sorted]

fig, ax = plt.subplots(figsize=(15, 6))
fig.patch.set_facecolor(PAL["bg"])

bars = ax.barh(range(len(methods_sorted)), accs_sorted,
               color=colors_sorted, alpha=0.88, zorder=3, height=0.65)
for i, (bar, acc, method) in enumerate(zip(bars, accs_sorted, methods_sorted)):
    star = " ★ BEST" if method == BEST else ""
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{acc:.2f}%{star}",
            va="center", ha="left",
            color=PAL["yellow"] if method == BEST else PAL["text"],
            fontsize=9, fontweight="bold" if method == BEST else "normal")

ax.set_yticks(range(len(methods_sorted)))
ax.set_yticklabels(methods_sorted, fontsize=10)
ax.set_xlabel("VQNN Test Accuracy (%)", fontsize=11)
ax.set_title("Feature Selection Method Comparison — VQNN Accuracy",
             fontsize=14, color=PAL["text"], fontweight="bold", pad=15)
ax.set_xlim(0, max(accs_sorted) + 12)
ax.axvline(np.mean(accs_sorted), color=PAL["muted"], lw=1.5, ls="--",
           label=f"Mean={np.mean(accs_sorted):.1f}%")
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3, zorder=0)
ax.invert_yaxis()
plt.tight_layout()
save("FS_A_accuracy_all_methods.png")

# ─────────────────────────────────────────────────────────────────────
# FIGURE B – Multi-Metric Comparison (Acc, F1, AUC, MCC)
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor(PAL["bg"])
axes = axes.flatten()

metrics_keys  = ["Accuracy", "F1", "AUC", "MCC"]
metrics_units = ["%", "", "", ""]
metric_colors = [PAL["accent"], PAL["green"], PAL["purple"], PAL["orange"]]

short_names = [m.replace("SelectKBest","SKB").replace("Importance","Imp.")
                .replace("Random Forest","RF").replace("Extra Trees","ET")
                .replace("(ANOVA F)","(F)").replace("(MI)","(MI)")
                .replace("Truncated ","Trunc.").replace("Kernel ","K.")
               for m in methods_sorted]

x = np.arange(len(methods_sorted))

for ax, mk, unit, col in zip(axes, metrics_keys, metrics_units, metric_colors):
    vals = [r[mk] for r in all_results]
    if mk == "Accuracy":
        vals = [v/100 for v in vals]
    bars = ax.bar(x, vals, color=col, alpha=0.82, zorder=3, width=0.65)
    best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor(PAL["yellow"])
    bars[best_idx].set_linewidth(2.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.005,
                f"{v:.3f}" if mk != "Accuracy" else f"{v*100:.1f}%",
                ha="center", va="bottom",
                color=PAL["text"], fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=8)
    ax.set_title(mk, color=PAL["text"], fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(vals)*1.2)
    ax.grid(axis="y", alpha=0.3, zorder=0)

fig.suptitle("Multi-Metric Comparison Across Feature Selection Methods",
             fontsize=15, color=PAL["text"], fontweight="bold", y=1.01)
plt.tight_layout()
save("FS_B_multi_metric_comparison.png")

# ─────────────────────────────────────────────────────────────────────
# FIGURE C – Individual Confusion Matrix per Method (2×5 grid)
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.patch.set_facecolor(PAL["bg"])
cmap_q = LinearSegmentedColormap.from_list(
    "qblue", ["#0d1117","#0c2d48","#1f77b4","#58a6ff"], N=256)

for ax, res in zip(axes.flatten(), all_results):
    cm   = res["CM"]
    col  = METHOD_COLORS[METHODS.index(res["Method"])]
    im   = ax.imshow(cm, cmap=cmap_q, vmin=0)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Mal.","Ben."], color=PAL["muted"], fontsize=8)
    ax.set_yticklabels(["Mal.","Ben."], color=PAL["muted"], fontsize=8,
                        rotation=90, va="center")
    ax.set_xlabel("Predicted", color=PAL["muted"], fontsize=7)
    ax.set_ylabel("Actual",    color=PAL["muted"], fontsize=7)
    star = " ★" if res["Method"] == BEST else ""
    ax.set_title(f"{res['Method']}{star}\nAcc={res['Accuracy']:.1f}%",
                 color=PAL["yellow"] if res["Method"]==BEST else PAL["text"],
                 fontsize=8, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color=PAL["text"], fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Confusion Matrices — All 10 Feature Selection Methods",
             fontsize=15, color=PAL["text"], fontweight="bold")
plt.tight_layout()
save("FS_C_confusion_matrices_all.png")

# ─────────────────────────────────────────────────────────────────────
# FIGURE D – Radar Chart (top 5 methods, 4 metrics)
# ─────────────────────────────────────────────────────────────────────
top5 = all_results[:5]
categories = ["Accuracy", "F1", "AUC", "MCC"]
N_cat = len(categories)
angles = [n/float(N_cat)*2*np.pi for n in range(N_cat)]
angles += angles[:1]

fig = plt.figure(figsize=(10, 9))
fig.patch.set_facecolor(PAL["bg"])
ax  = fig.add_subplot(111, polar=True)
ax.set_facecolor(PAL["panel"])
ax.spines["polar"].set_color(PAL["border"])

for i, res in enumerate(top5):
    vals = [
        res["Accuracy"]/100,
        res["F1"],
        res["AUC"],
        (res["MCC"] + 1) / 2,   # normalize MCC [-1,1] → [0,1]
    ]
    vals += vals[:1]
    col = METHOD_COLORS[METHODS.index(res["Method"])]
    ax.plot(angles, vals, "o-", lw=2, color=col,
            label=res["Method"], markersize=6)
    ax.fill(angles, vals, alpha=0.08, color=col)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(["Accuracy","F1","AUC","MCC (norm)"],
                   color=PAL["text"], fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"],
                   color=PAL["muted"], fontsize=8)
ax.tick_params(colors=PAL["muted"])
ax.grid(color=PAL["border"], alpha=0.5)

plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
           fontsize=9, facecolor=PAL["panel"],
           edgecolor=PAL["border"], labelcolor=PAL["text"])
plt.title("Radar Chart — Top 5 Feature Selection Methods",
          fontsize=14, color=PAL["text"], fontweight="bold", y=1.08)
plt.tight_layout()
save("FS_D_radar_top5.png")

# ─────────────────────────────────────────────────────────────────────
# FIGURE E – Accuracy vs Time scatter
# ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(PAL["bg"])

for i, res in enumerate(all_results):
    col = METHOD_COLORS[METHODS.index(res["Method"])]
    ax.scatter(res["Time_s"], res["Accuracy"], s=180,
               color=col, zorder=5, edgecolors=PAL["text"], lw=0.8)
    ax.text(res["Time_s"]+0.5, res["Accuracy"]+0.15,
            res["Method"].replace("SelectKBest","SKB")
                         .replace("Importance","Imp.")
                         .replace("Random Forest","RF")
                         .replace("Extra Trees","ET"),
            color=col, fontsize=8, va="bottom")

ax.set_xlabel("Computation Time (seconds)", fontsize=11)
ax.set_ylabel("VQNN Accuracy (%)", fontsize=11)
ax.set_title("Accuracy vs Computation Time Trade-off",
             fontsize=14, color=PAL["text"], fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout()
save("FS_E_accuracy_vs_time.png")

# ─────────────────────────────────────────────────────────────────────
# FIGURE F – Heatmap: methods × metrics
# ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(PAL["bg"])

heat_data = np.array([
    [r["Accuracy"]/100, r["F1"], r["AUC"], (r["MCC"]+1)/2]
    for r in all_results
])
cmap_heat = LinearSegmentedColormap.from_list(
    "heat", [PAL["bg"], PAL["purple"], PAL["accent"], PAL["green"]], N=256)
im = ax.imshow(heat_data, cmap=cmap_heat, aspect="auto", vmin=0, vmax=1)

short_labels = [m.replace("SelectKBest","SKB").replace("Importance","Imp.")
                 .replace("Random Forest","RF").replace("Extra Trees","ET")
                 .replace("(ANOVA F)","(F)").replace("Truncated ","")
                 .replace("Kernel ","K.")
                for m in [r["Method"] for r in all_results]]

ax.set_xticks(range(4))
ax.set_xticklabels(["Accuracy","F1","AUC","MCC (norm)"],
                   fontsize=11, color=PAL["text"])
ax.set_yticks(range(len(all_results)))
ax.set_yticklabels(short_labels, fontsize=10)

for i in range(len(all_results)):
    for j in range(4):
        ax.text(j, i, f"{heat_data[i,j]:.3f}",
                ha="center", va="center",
                color=PAL["text"] if heat_data[i,j]<0.8 else "#0d1117",
                fontsize=9, fontweight="bold")

plt.colorbar(im, ax=ax, label="Normalized Score", fraction=0.025)
ax.set_title("Performance Heatmap — All Methods × All Metrics",
             fontsize=14, color=PAL["text"], fontweight="bold", pad=12)
plt.tight_layout()
save("FS_F_performance_heatmap.png")

# ─────────────────────────────────────────────────────────────────────
# FIGURE G – Feature Importance bars for tree-based methods
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(PAL["bg"])

tree_methods = ["Random Forest Importance", "Extra Trees Importance",
                "RFE (Random Forest)"]

for ax, tm in zip(axes, tree_methods):
    info = method_details.get(tm, {})
    if "selected_features" in info and "importances" in info:
        feats = info["selected_features"]
        imps  = info["importances"]
        bars = ax.bar(range(len(feats)), imps,
                      color=METHOD_COLORS[METHODS.index(tm)], alpha=0.85,
                      zorder=3)
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels([f.replace("mean ","").replace(" error","_err")
                             for f in feats],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Feature Importance", fontsize=10)
        for bar, v in zip(bars, imps):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.003,
                    f"{v:.3f}", ha="center", va="bottom",
                    color=PAL["text"], fontsize=8)
    elif "selected_features" in info:
        feats = info["selected_features"]
        ax.bar(range(len(feats)), [1]*len(feats),
               color=METHOD_COLORS[METHODS.index(tm)], alpha=0.85, zorder=3)
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels([f.replace("mean ","") for f in feats],
                           rotation=30, ha="right", fontsize=9)
    ax.set_title(tm, color=PAL["text"], fontweight="bold", fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)

fig.suptitle("Selected Features & Importances (Tree-Based Methods)",
             fontsize=14, color=PAL["text"], fontweight="bold")
plt.tight_layout()
save("FS_G_feature_importances.png")

# ─────────────────────────────────────────────────────────────────────
# FIGURE H – Statistical filter scores (SelectKBest methods)
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(PAL["bg"])

skb_methods = [("SelectKBest (ANOVA F)", "f_scores", "ANOVA F-Score", PAL["red"]),
               ("SelectKBest (MI)",      "mi_scores","Mutual Information",PAL["purple"])]

for ax, (tm, score_key, ylabel, col) in zip(axes, skb_methods):
    info = method_details.get(tm, {})
    if "selected_features" in info and score_key in info:
        feats  = info["selected_features"]
        scores = info[score_key]
        bars   = ax.bar(range(len(feats)), scores, color=col, alpha=0.85, zorder=3)
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels([f.replace("mean ","").replace(" error","_err")
                             for f in feats],
                           rotation=25, ha="right", fontsize=9)
        for bar, v in zip(bars, scores):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+max(scores)*0.01,
                    f"{v:.2f}", ha="center", va="bottom",
                    color=PAL["text"], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(tm, color=PAL["text"], fontweight="bold", fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)

fig.suptitle("Statistical Filter Scores — Top 4 Selected Features",
             fontsize=14, color=PAL["text"], fontweight="bold")
plt.tight_layout()
save("FS_H_statistical_filter_scores.png")

# ─────────────────────────────────────────────────────────────────────
# FIGURE I – Ranked summary table (visual)
# ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(PAL["bg"]); ax.axis("off")

headers = ["Rank","Method","Accuracy","F1","AUC","MCC","Time(s)","Category"]
rows = []
categories = {
    "PCA":"Projection","LDA":"Projection","Kernel PCA":"Projection",
    "ICA":"Projection","Truncated SVD":"Projection",
    "SelectKBest (ANOVA F)":"Filter","SelectKBest (MI)":"Filter",
    "RFE (Random Forest)":"Wrapper","Random Forest Importance":"Embedded",
    "Extra Trees Importance":"Embedded",
}
for rank, res in enumerate(all_results, 1):
    rows.append([
        f"{'★ ' if rank==1 else ''}{rank}",
        res["Method"],
        f"{res['Accuracy']:.2f}%",
        f"{res['F1']:.3f}",
        f"{res['AUC']:.3f}",
        f"{res['MCC']:.3f}",
        f"{res['Time_s']:.0f}s",
        categories.get(res["Method"],"—"),
    ])

table = ax.table(cellText=rows, colLabels=headers,
                 cellLoc="center", loc="center")
table.auto_set_font_size(False); table.set_fontsize(9.5)
table.scale(1, 1.6)

# Style header
for j in range(len(headers)):
    table[0, j].set_facecolor(PAL["accent"])
    table[0, j].set_text_props(color="white", fontweight="bold")

# Style rows
for i, res in enumerate(all_results, 1):
    col = METHOD_COLORS[METHODS.index(res["Method"])]
    base_bg = "#1a1f2e" if i % 2 == 0 else PAL["panel"]
    for j in range(len(headers)):
        table[i, j].set_facecolor(
            PAL["yellow"].replace("d2","1a").replace("99","15").replace("22","0a")
            if i == 1 else base_bg)
        table[i, j].set_text_props(
            color=PAL["yellow"] if i==1 else PAL["text"])

ax.set_title("Ranked Summary — Feature Selection Methods for VQNN",
             fontsize=14, color=PAL["text"], fontweight="bold",
             pad=20)
plt.tight_layout()
save("FS_I_ranked_summary_table.png")

# ─────────────────────────────────────────────────────────────────────
# FIGURE J – Individual method accuracy charts (10 separate bars)
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.patch.set_facecolor(PAL["bg"])

for ax, res in zip(axes.flatten(), all_results):
    col   = METHOD_COLORS[METHODS.index(res["Method"])]
    mets  = ["Accuracy","F1","AUC","MCC"]
    raw_v = [res["Accuracy"]/100, res["F1"], res["AUC"],
             (res["MCC"]+1)/2]
    bars  = ax.bar(mets, raw_v, color=[col,PAL["green"],
                                        PAL["purple"],PAL["orange"]],
                   alpha=0.85, zorder=3)
    for bar, v in zip(bars, raw_v):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.02,
                f"{v:.3f}", ha="center", va="bottom",
                color=PAL["text"], fontsize=8, fontweight="bold")
    star = " ★" if res["Method"] == BEST else ""
    ax.set_title(f"#{all_results.index(res)+1} {res['Method']}{star}",
                 color=PAL["yellow"] if res["Method"]==BEST else PAL["text"],
                 fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.25)
    ax.grid(axis="y", alpha=0.3, zorder=0)

fig.suptitle("Individual Method Performance Cards — All 10 Feature Selection Methods",
             fontsize=14, color=PAL["text"], fontweight="bold")
plt.tight_layout()
save("FS_J_individual_method_cards.png")

# ─────────────────────────────────────────────────────────────────────
# PRINT FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("  FINAL SUMMARY — FEATURE SELECTION COMPARISON")
print("="*62)
print(f"  {'Rank':>4}  {'Method':30s}  {'Acc':>7}  {'F1':>6}  {'AUC':>6}  {'MCC':>6}")
print("  " + "─"*58)
for rank, res in enumerate(all_results, 1):
    star = " ★" if rank == 1 else ""
    print(f"  {rank:>4}  {res['Method']:30s}  "
          f"{res['Accuracy']:>6.2f}%  {res['F1']:>6.3f}  "
          f"{res['AUC']:>6.3f}  {res['MCC']:>6.3f}{star}")

print(f"\n  ★ BEST METHOD : {all_results[0]['Method']}")
print(f"  ★ BEST ACC    : {all_results[0]['Accuracy']:.2f}%")
print(f"  ★ BEST AUC    : {all_results[0]['AUC']:.3f}")
print(f"\n  Figures saved (10 total):")
for fig_name in ["FS_A_accuracy_all_methods.png",
                 "FS_B_multi_metric_comparison.png",
                 "FS_C_confusion_matrices_all.png",
                 "FS_D_radar_top5.png",
                 "FS_E_accuracy_vs_time.png",
                 "FS_F_performance_heatmap.png",
                 "FS_G_feature_importances.png",
                 "FS_H_statistical_filter_scores.png",
                 "FS_I_ranked_summary_table.png",
                 "FS_J_individual_method_cards.png"]:
    print(f"    {fig_name}")
print("="*62)
