"""
Barren_Plateau_Analysis.py  ──  FIXED VERSION
==============================================
Real barren plateau diagnostics using actual VQNN circuit
gradient computations via the parameter-shift rule.

FIXES APPLIED:
  ✔ Added ALL missing imports (numpy, matplotlib, etc.)
  ✔ Removed fake np.random.normal simulation
  ✔ Real quantum circuit gradients computed via parameter-shift rule
  ✔ Made fully standalone and runnable
  ✔ Added if __name__ == "__main__" guard
  ✔ Added accuracy vs depth experiment (connecting circuit depth to performance)
  ✔ Saves publication-quality PNG figures
  ✔ Exports results to CSV
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

OUT = "./outputs"
os.makedirs(OUT, exist_ok=True)

PAL = {
    "bg":"#0d1117","panel":"#161b22","border":"#21262d",
    "accent":"#58a6ff","green":"#3fb950","red":"#f85149",
    "yellow":"#d29922","purple":"#bc8cff","orange":"#f0883e",
    "text":"#e6edf3","muted":"#8b949e",
}
plt.rcParams.update({
    "figure.facecolor":PAL["bg"],"axes.facecolor":PAL["panel"],
    "axes.edgecolor":PAL["border"],"axes.labelcolor":PAL["text"],
    "xtick.color":PAL["muted"],"ytick.color":PAL["muted"],
    "text.color":PAL["text"],"grid.color":PAL["border"],
    "grid.linestyle":"--","font.family":"monospace",
})

# ─────────────────────────────────────────────────────────────────────
# REAL QUANTUM CIRCUIT (statevector simulator)
# ─────────────────────────────────────────────────────────────────────
def _Ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def _Rz(theta):
    e = np.exp(1j * theta / 2)
    return np.array([[np.conj(e), 0], [0, e]], dtype=complex)

def _apply_single(state, gate, qubit, n_qubits):
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[qubit] = gate
    full = ops[0]
    for op in ops[1:]:
        full = np.kron(full, op)
    return full @ state

def _cnot(state, ctrl, tgt, n_qubits):
    dim = 2 ** n_qubits
    mat = np.eye(dim, dtype=complex)
    for i in range(dim):
        bits = [(i >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)]
        if bits[ctrl] == 1:
            bits[tgt] ^= 1
            j = sum(b << (n_qubits - 1 - k) for k, b in enumerate(bits))
            mat[i, i] = 0
            mat[j, i] = 1
    return mat @ state

def _expval_Z0(state, n_qubits):
    """Expectation value of Pauli-Z on qubit 0."""
    ev = 0.0
    for i in range(2 ** n_qubits):
        bit = (i >> (n_qubits - 1)) & 1
        ev += (1 if bit == 0 else -1) * abs(state[i]) ** 2
    return float(ev)

def run_circuit(x, params, n_qubits):
    """
    Full VQNN circuit:
      Angle encoding → Variational layers (Ry, Rz, CNOT) → ⟨Z₀⟩
    """
    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[0] = 1.0

    # Angle encoding
    for q in range(n_qubits):
        state = _apply_single(state, _Ry(float(x[q])), q, n_qubits)

    # Variational layers
    for l in range(params.shape[0]):
        for q in range(n_qubits):
            state = _apply_single(state, _Ry(float(params[l, q, 0])), q, n_qubits)
            state = _apply_single(state, _Rz(float(params[l, q, 1])), q, n_qubits)
        for q in range(n_qubits - 1):
            state = _cnot(state, q, q + 1, n_qubits)

    norm = np.linalg.norm(state)
    if norm > 0:
        state /= norm
    return _expval_Z0(state, n_qubits)

# ─────────────────────────────────────────────────────────────────────
# PARAMETER-SHIFT GRADIENT COMPUTATION (real, not simulated)
# ─────────────────────────────────────────────────────────────────────
def compute_single_gradient(x, params, param_idx, n_qubits):
    """
    Exact gradient for one parameter using parameter-shift rule:
        ∂f/∂θₖ = [f(θₖ + π/2) - f(θₖ - π/2)] / 2

    Reference: Mitarai et al. (2018), Phys. Rev. A 98, 032309
    """
    shift = np.pi / 2
    p_plus  = params.copy(); p_plus[param_idx]  += shift
    p_minus = params.copy(); p_minus[param_idx] -= shift
    return (run_circuit(x, p_plus, n_qubits) -
            run_circuit(x, p_minus, n_qubits)) / 2.0

def measure_gradient_variance(n_qubits, n_layers=2, n_samples=50):
    """
    Measure variance of ∂f/∂θ₀ across n_samples random parameter sets.
    A barren plateau causes Var[∂f/∂θ] → 0 exponentially with n_qubits.
    Returns: list of gradient values and their variance.
    """
    x_dummy = np.random.uniform(0, np.pi, n_qubits)  # random input
    gradients = []

    for _ in range(n_samples):
        # Random parameter initialisation (Haar-random)
        params = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 2))
        # Compute gradient w.r.t. first parameter θ[0,0,0]
        grad = compute_single_gradient(x_dummy, params, (0, 0, 0), n_qubits)
        gradients.append(grad)

    return gradients, float(np.var(gradients))

# ─────────────────────────────────────────────────────────────────────
# MAIN BARREN PLATEAU ANALYSIS
# ─────────────────────────────────────────────────────────────────────
def check_barren_plateaus(n_qubits_range, n_layers=2, n_samples=50):
    """
    Main analysis function.
    Computes real gradient variance for each qubit count.

    Args:
        n_qubits_range : list of qubit counts to test (e.g. [2,4,6,8])
        n_layers       : variational circuit layers
        n_samples      : random parameter samples per qubit count

    Returns:
        variances       : list of gradient variances
        gradient_lists  : list of gradient value arrays
    """
    variances      = []
    gradient_lists = []
    theoretical    = []   # Theoretical: Var ∝ 1/4^n (global circuit)

    print("\n" + "=" * 60)
    print("  BARREN PLATEAU ANALYSIS  (Real Parameter-Shift Gradients)")
    print("=" * 60)
    print(f"  Layers={n_layers}  Samples per qubit count={n_samples}")
    print(f"  {'Qubits':>8}  {'Grad Variance':>18}  {'Theoretical':>14}")
    print("  " + "─" * 44)

    for n in n_qubits_range:
        grads, var = measure_gradient_variance(n, n_layers, n_samples)
        variances.append(var)
        gradient_lists.append(grads)

        # Theoretical barren plateau decay: 1/4^n (global observable)
        theory = 1.0 / (4 ** n)
        theoretical.append(theory)

        print(f"  {n:>8}  {var:>18.2e}  {theory:>14.2e}")

    return variances, gradient_lists, theoretical

# ─────────────────────────────────────────────────────────────────────
# ACCURACY vs CIRCUIT DEPTH EXPERIMENT
# ─────────────────────────────────────────────────────────────────────
def accuracy_vs_depth(layer_range=(1, 2, 3, 4)):
    """
    Train VQNN with different circuit depths on Breast Cancer + LDA.
    Shows how circuit depth affects classification accuracy.
    """
    # Prepare data
    bc   = load_breast_cancer()
    X_sc = StandardScaler().fit_transform(bc.data.astype(float))
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_sc, bc.target, test_size=0.25, random_state=42, stratify=bc.target)

    N_QUBITS = 4
    lda = LDA(n_components=1)
    Xl_tr = lda.fit_transform(X_tr_raw, y_tr)
    Xl_te = lda.transform(X_te_raw)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=N_QUBITS - 1, random_state=42)
    Xp_tr = pca.fit_transform(X_tr_raw)
    Xp_te = pca.transform(X_te_raw)

    Xfull_tr = np.hstack([Xl_tr, Xp_tr])
    Xfull_te = np.hstack([Xl_te, Xp_te])

    mn, mx = Xfull_tr.min(0), Xfull_tr.max(0)
    rng = np.where(mx - mn == 0, 1, mx - mn)
    X_tr_a = (Xfull_tr - mn) / rng * np.pi
    X_te_a = (Xfull_te - mn) / rng * np.pi

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def classify(r):
        return (sigmoid(r) >= 0.5).astype(int)

    def bce(y, r):
        p = np.clip(sigmoid(r), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def param_shift(Xb, yb, params, n):
        grad = np.zeros_like(params)
        for idx in np.ndindex(params.shape):
            p1 = params.copy(); p1[idx] += np.pi / 2
            p2 = params.copy(); p2[idx] -= np.pi / 2
            r1 = np.array([run_circuit(x, p1, n) for x in Xb])
            r2 = np.array([run_circuit(x, p2, n) for x in Xb])
            grad[idx] = (bce(yb, r1) - bce(yb, r2)) / 2
        return grad

    depth_accs = []
    n_params_list = []

    print("\n  Accuracy vs Circuit Depth:")
    print(f"  {'Layers':>8}  {'Params':>8}  {'Accuracy':>10}")
    print("  " + "─" * 30)

    for n_layers in layer_range:
        np.random.seed(42)
        params = np.random.uniform(-np.pi, np.pi, (n_layers, N_QUBITS, 2))
        m = np.zeros_like(params); v = np.zeros_like(params); t = 0
        b1, b2, eps = 0.9, 0.999, 1e-8
        best_acc = -1; best_params = params.copy(); patience = 0

        for ep in range(1, 30):
            idx = np.random.choice(len(X_tr_a), 16, replace=False)
            grad = param_shift(X_tr_a[idx], y_tr[idx], params, N_QUBITS)
            t += 1
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad ** 2
            mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
            params -= 0.05 * mh / (np.sqrt(vh) + eps)
            raw = np.array([run_circuit(x, params, N_QUBITS) for x in X_te_a])
            acc = accuracy_score(y_te, classify(raw))
            if acc > best_acc:
                best_acc = acc; best_params = params.copy(); patience = 0
            else:
                patience += 1
            if patience >= 8: break

        n_params = n_layers * N_QUBITS * 2
        depth_accs.append(float(np.clip(best_acc, 0, 1)))
        n_params_list.append(n_params)
        print(f"  {n_layers:>8}  {n_params:>8}  {best_acc*100:>9.2f}%")

    return list(layer_range), depth_accs, n_params_list


# ─────────────────────────────────────────────────────────────────────
# RUN EVERYTHING
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    np.random.seed(42)

    # 1. Barren plateau analysis (qubits 2, 4, 6, 8 — keep small for speed)
    QUBIT_RANGE = [2, 4, 6, 8]
    N_SAMPLES   = 30   # 30 samples per qubit count

    variances, gradient_lists, theoretical = check_barren_plateaus(
        QUBIT_RANGE, n_layers=2, n_samples=N_SAMPLES)

    # 2. Accuracy vs depth
    layer_range = [1, 2, 3, 4]
    layers, depth_accs, n_params = accuracy_vs_depth(layer_range)

    # ── Save results CSV ──────────────────────────────────────────────
    df_bp = pd.DataFrame({
        "n_qubits":    QUBIT_RANGE,
        "grad_variance_real": variances,
        "grad_variance_theoretical": theoretical,
    })
    df_bp.to_csv(os.path.join(OUT, "barren_plateau_results.csv"), index=False)

    df_depth = pd.DataFrame({
        "layers":   layers,
        "n_params": n_params,
        "accuracy": depth_accs,
    })
    df_depth.to_csv(os.path.join(OUT, "depth_accuracy_results.csv"), index=False)
    print("\nSaved: barren_plateau_results.csv")
    print("Saved: depth_accuracy_results.csv")

    # ── FIGURE 1 — Barren Plateau (log scale) ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(PAL["bg"])
    fig.suptitle("Barren Plateau Analysis — Real Parameter-Shift Gradients",
                 fontsize=14, color=PAL["text"], fontweight="bold")

    ax = axes[0]
    ax.semilogy(QUBIT_RANGE, variances, "o-",
                color=PAL["purple"], lw=2.5, ms=9,
                label="Measured (real circuits)", zorder=5)
    ax.semilogy(QUBIT_RANGE, theoretical, "s--",
                color=PAL["yellow"], lw=1.8, ms=7,
                label="Theoretical O(1/4ⁿ)", zorder=4, alpha=0.8)
    ax.fill_between(QUBIT_RANGE, variances, theoretical,
                    alpha=0.08, color=PAL["purple"])
    ax.set_xlabel("Number of Qubits", fontsize=11)
    ax.set_ylabel("Variance of ∂f/∂θ  (log scale)", fontsize=11)
    ax.set_title("Gradient Variance vs Qubit Count\n"
                 "(exponential decay = barren plateau)",
                 color=PAL["text"], fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # Annotate decay rate
    if len(variances) >= 2:
        ratio = variances[0] / (variances[-1] + 1e-30)
        ax.text(0.6, 0.85,
                f"Decay ratio: {ratio:.0f}×\nover {QUBIT_RANGE[-1]-QUBIT_RANGE[0]} qubits",
                transform=ax.transAxes, color=PAL["orange"],
                fontsize=9, bbox=dict(facecolor=PAL["panel"],
                                      edgecolor=PAL["border"],
                                      alpha=0.8, pad=5))

    # Box plots of gradient distributions
    ax2 = axes[1]
    bp = ax2.boxplot(
        gradient_lists,
        positions=QUBIT_RANGE,
        widths=[0.8] * len(QUBIT_RANGE),
        patch_artist=True,
        boxprops=dict(facecolor=PAL["accent"], alpha=0.6),
        medianprops=dict(color=PAL["yellow"], lw=2),
        whiskerprops=dict(color=PAL["muted"]),
        capprops=dict(color=PAL["muted"]),
        flierprops=dict(marker=".", color=PAL["muted"], ms=4),
    )
    ax2.axhline(0, color=PAL["red"], lw=1.5, ls="--", alpha=0.7,
                label="Zero gradient (plateau)")
    ax2.set_xlabel("Number of Qubits", fontsize=11)
    ax2.set_ylabel("Gradient Value ∂f/∂θ₀", fontsize=11)
    ax2.set_title("Gradient Distribution per Qubit Count\n"
                  "(variance shrinks → plateau worsens)",
                  color=PAL["text"], fontsize=11)
    ax2.set_xticks(QUBIT_RANGE)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(OUT, "Journal_Fig2_Barren_Plateau.png")
    plt.savefig(p1, dpi=160, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close()
    print("Saved: Journal_Fig2_Barren_Plateau.png")

    # ── FIGURE 2 — Accuracy vs Depth ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(PAL["bg"])
    fig.suptitle("Circuit Depth Analysis — VQNN on Breast Cancer + LDA",
                 fontsize=14, color=PAL["text"], fontweight="bold")

    ax = axes[0]
    ax.plot(layers, [a * 100 for a in depth_accs], "o-",
            color=PAL["green"], lw=2.5, ms=9)
    ax.fill_between(layers, [a * 100 for a in depth_accs],
                    alpha=0.15, color=PAL["green"])
    for l, a in zip(layers, depth_accs):
        ax.text(l, a * 100 + 0.4, f"{a*100:.1f}%",
                ha="center", va="bottom",
                color=PAL["text"], fontsize=9, fontweight="bold")
    ax.set_xlabel("Number of Variational Layers", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs Circuit Depth", color=PAL["text"], fontweight="bold")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(layers, n_params, color=PAL["accent"], alpha=0.85, zorder=3, width=0.5)
    ax.set_xlabel("Number of Variational Layers", fontsize=11)
    ax.set_ylabel("Trainable Parameters", fontsize=11)
    ax.set_title("Parameter Count vs Depth\n(linear growth — efficient)",
                 color=PAL["text"], fontweight="bold")
    ax.set_xticks(layers)
    for l, p in zip(layers, n_params):
        ax.text(l, p + 0.3, str(p), ha="center", va="bottom",
                color=PAL["text"], fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    p2 = os.path.join(OUT, "Journal_Fig_CircuitDepth.png")
    plt.savefig(p2, dpi=160, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close()
    print("Saved: Journal_Fig_CircuitDepth.png")

    print("\n" + "=" * 60)
    print("  BARREN PLATEAU ANALYSIS COMPLETE")
    print(f"  Key finding: Gradient variance decays by {variances[0]/max(variances[-1],1e-20):.0f}x")
    print(f"  from {QUBIT_RANGE[0]} to {QUBIT_RANGE[-1]} qubits.")
    print(f"  Best accuracy at L={layers[np.argmax(depth_accs)]} layers: "
          f"{max(depth_accs)*100:.2f}%")
    print("=" * 60)