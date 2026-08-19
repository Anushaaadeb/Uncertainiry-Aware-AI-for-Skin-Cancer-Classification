import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from model import DualModeClassifier
from inference_engine import predict_standard, predict_bnn_mcdropout, DEFAULT_UNC_THRESHOLDS, CLASSES
from train import HAM10000Dataset, SEED
from labels import NUM_CLASSES
from config import IMG_SIZE

OUTDIR = "eval_plots"


def calculate_ece(probs, targets, n_bins=10):
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(targets)
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        in_bin = confidences.gt(bin_boundaries[i]) * confidences.le(bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean().item()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_conf_in_bin = confidences[in_bin].mean().item()
            ece += np.abs(avg_conf_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def reliability_bins(probs, targets, n_bins=10):
    """Returns (bin_centers, bin_accuracy, bin_confidence, bin_count) for a reliability diagram."""
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(targets)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        in_bin = confidences.gt(bin_boundaries[i]) * confidences.le(bin_boundaries[i + 1])
        count = in_bin.sum().item()
        centers.append((bin_boundaries[i].item() + bin_boundaries[i + 1].item()) / 2)
        counts.append(count)
        if count > 0:
            accs.append(accuracies[in_bin].float().mean().item())
            confs.append(confidences[in_bin].mean().item())
        else:
            accs.append(np.nan)
            confs.append(np.nan)
    return np.array(centers), np.array(accs), np.array(confs), np.array(counts)


def plot_metrics_bar(std_acc, std_f1, std_ece, mcd_acc, mcd_f1, mcd_ece, path):
    metrics = ["Accuracy", "Macro F1", "ECE \u2193"]
    std_vals = [std_acc, std_f1, std_ece]
    mcd_vals = [mcd_acc, mcd_f1, mcd_ece]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - width / 2, std_vals, width, label="Standard", color="#3B82F6")
    b2 = ax.bar(x + width / 2, mcd_vals, width, label="Bayesian (MCD + TTA)", color="#10B981")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Standard vs Bayesian Model \u2014 Headline Metrics", fontweight="bold")
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=9)
    ax.legend()
    ax.margins(y=0.15)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved '{path}'")


def plot_uncertainty_kde(correct_unc, incorrect_unc, metric_name, threshold, path):
    plt.figure(figsize=(8, 4.5))
    sns.kdeplot(correct_unc, color="#10B981", label="Correct Predictions", fill=True, alpha=0.35)
    sns.kdeplot(incorrect_unc, color="#EF4444", label="Incorrect Predictions", fill=True, alpha=0.35)
    plt.axvline(x=threshold, color="#F59E0B", linestyle="--", linewidth=2,
                label=f"Safeguard Cutoff ({threshold:g})")
    plt.title(f"MC Dropout Uncertainty Separation \u2014 {metric_name}", fontsize=12, fontweight="bold")
    plt.xlabel(metric_name)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved '{path}'")


def plot_reliability_diagram(centers, accs, confs, counts, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    width = centers[1] - centers[0] if len(centers) > 1 else 0.1
    ax.bar(centers, accs, width=width * 0.9, color="#3B82F6", alpha=0.7,
           edgecolor="black", label="Accuracy in bin")
    ax.plot(centers, confs, marker="o", color="#EF4444", label="Avg. confidence in bin")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved '{path}'")


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", cbar=True,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved '{path}'")


def plot_threshold_sensitivity(uncertainty, correct_mask, thresholds, metric_name, path):
    """
    For each candidate threshold: what fraction of samples get rejected
    (flagged), and what fraction of the *errors* does that rejection
    actually catch? Helps pick a threshold instead of guessing.
    """
    reject_rate = []
    error_catch_rate = []
    total_errors = max((~correct_mask).sum(), 1)
    for t in thresholds:
        rejected = uncertainty > t
        reject_rate.append(rejected.mean())
        caught = (rejected & (~correct_mask)).sum()
        error_catch_rate.append(caught / total_errors)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, reject_rate, label="Fraction of all samples rejected", color="#F59E0B")
    ax.plot(thresholds, error_catch_rate, label="Fraction of errors caught by rejection", color="#EF4444")
    ax.set_xlabel(f"{metric_name} threshold")
    ax.set_ylabel("Rate")
    ax.set_title(f"Threshold Sensitivity \u2014 {metric_name}", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved '{path}'")


def evaluate_both():
    import os
    os.makedirs(OUTDIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualModeClassifier(num_classes=7, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load("fine_tuned_model.pth", map_location=device))

    dataset = HAM10000Dataset("ham10000/HAM10000_metadata.csv", "ham10000")

    # Same stratified split (and same SEED) as train.py, so rare classes
    # (df, vasc) aren't starved from the evaluation set. NOTE: this
    # reproduces the same split by construction (same seed, same
    # stratify), it isn't reading train.py's actual saved indices --
    # if you ever change the split logic in train.py, mirror it here too.
    labels_all = dataset.df['label'].values
    indices = np.arange(len(dataset))
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=labels_all)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    std_preds, std_probs_list = [], []
    mcd_preds, mcd_probs_list = [], []
    unc_variance, unc_pred_entropy, unc_mutual_info = [], [], []
    targets_list = []

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Running comparative evaluation...")
    num_samples_to_test = min(200, len(test_dataset))

    for idx in range(num_samples_to_test):
        img, label = test_dataset[idx]
        img_tensor = eval_transform(img).unsqueeze(0)

        # Standard Pass
        res_std = predict_standard(model, img_tensor, device)
        std_preds.append(res_std["pred_class"])
        std_probs_list.append(res_std["probs"].numpy())

        # BNN Pass (with TTA enabled) -- compute all uncertainty metrics
        res_mcd = predict_bnn_mcdropout(model, img_tensor, num_samples=30, use_tta=True,
                                         device=device, uncertainty_metric="mutual_info")
        mcd_preds.append(res_mcd["pred_class"])
        mcd_probs_list.append(res_mcd["mean_probs"].numpy())
        unc_variance.append(res_mcd["uncertainty_all"]["variance"])
        unc_pred_entropy.append(res_mcd["uncertainty_all"]["predictive_entropy"])
        unc_mutual_info.append(res_mcd["uncertainty_all"]["mutual_info"])

        targets_list.append(label)

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{num_samples_to_test} samples done")

    y_true = np.array(targets_list)
    std_preds = np.array(std_preds)
    mcd_preds = np.array(mcd_preds)
    std_probs = torch.tensor(np.array(std_probs_list))
    mcd_probs = torch.tensor(np.array(mcd_probs_list))
    unc_variance = np.array(unc_variance)
    unc_pred_entropy = np.array(unc_pred_entropy)
    unc_mutual_info = np.array(unc_mutual_info)

    # Headline metrics
    std_acc = accuracy_score(y_true, std_preds)
    mcd_acc = accuracy_score(y_true, mcd_preds)
    _, _, std_f1, _ = precision_recall_fscore_support(y_true, std_preds, average='macro')
    _, _, mcd_f1, _ = precision_recall_fscore_support(y_true, mcd_preds, average='macro')
    std_ece = calculate_ece(std_probs, torch.tensor(y_true))
    mcd_ece = calculate_ece(mcd_probs, torch.tensor(y_true))

    df = pd.DataFrame({
        "Metric": ["Accuracy", "Macro F1-Score", "ECE (Calibration Error) \u2193"],
        "Standard Fine-Tuned Model": [f"{std_acc:.4f}", f"{std_f1:.4f}", f"{std_ece:.4f}"],
        "Bayesian Model (MCD + TTA)": [f"{mcd_acc:.4f}", f"{mcd_f1:.4f}", f"{mcd_ece:.4f}"]
    })
    print("\n" + "=" * 65)
    print("                     EVALUATION RESULTS")
    print("=" * 65)
    print(df.to_string(index=False))
    print("=" * 65 + "\n")

    correct_mask = (mcd_preds == y_true)

    # --- Plot 1: headline metrics bar chart ---
    plot_metrics_bar(std_acc, std_f1, std_ece, mcd_acc, mcd_f1, mcd_ece,
                      f"{OUTDIR}/01_metrics_comparison.png")

    # --- Plots 2-4: uncertainty separation, one file per metric ---
    plot_uncertainty_kde(unc_variance[correct_mask], unc_variance[~correct_mask],
                          "Softmax Variance", DEFAULT_UNC_THRESHOLDS["variance"],
                          f"{OUTDIR}/02_uncertainty_variance.png")
    plot_uncertainty_kde(unc_pred_entropy[correct_mask], unc_pred_entropy[~correct_mask],
                          "Predictive Entropy", DEFAULT_UNC_THRESHOLDS["predictive_entropy"],
                          f"{OUTDIR}/03_uncertainty_predictive_entropy.png")
    plot_uncertainty_kde(unc_mutual_info[correct_mask], unc_mutual_info[~correct_mask],
                          "Mutual Information (BALD)", DEFAULT_UNC_THRESHOLDS["mutual_info"],
                          f"{OUTDIR}/04_uncertainty_mutual_info.png")

    # --- Plots 5-6: reliability diagrams ---
    c, a, cf, _ = reliability_bins(std_probs, torch.tensor(y_true))
    plot_reliability_diagram(c, a, cf, _, "Reliability Diagram \u2014 Standard Model",
                              f"{OUTDIR}/05_reliability_standard.png")
    c, a, cf, _ = reliability_bins(mcd_probs, torch.tensor(y_true))
    plot_reliability_diagram(c, a, cf, _, "Reliability Diagram \u2014 Bayesian Model",
                              f"{OUTDIR}/06_reliability_bayesian.png")

    # --- Plots 7-8: confusion matrices ---
    plot_confusion(y_true, std_preds, "Confusion Matrix \u2014 Standard Model",
                   f"{OUTDIR}/07_confusion_standard.png")
    plot_confusion(y_true, mcd_preds, "Confusion Matrix \u2014 Bayesian Model",
                   f"{OUTDIR}/08_confusion_bayesian.png")

    # --- Plot 9: threshold sensitivity for the recommended metric (mutual info) ---
    thresholds = np.linspace(unc_mutual_info.min(), unc_mutual_info.max(), 50)
    plot_threshold_sensitivity(unc_mutual_info, correct_mask, thresholds, "Mutual Information (BALD)",
                                f"{OUTDIR}/09_threshold_sensitivity_mutual_info.png")

    print(f"\nAll plots saved to ./{OUTDIR}/")


if __name__ == "__main__":
    evaluate_both()