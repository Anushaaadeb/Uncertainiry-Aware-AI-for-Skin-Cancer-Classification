import torch
import torch.nn.functional as F
from torchvision import transforms

# HAM10000 class names -- imported from labels.py, the single source of
# truth also used by train.py to build training labels. Do NOT redefine
# this list locally: train.py and this file must agree on which index
# means which class, or predictions get mislabeled (e.g. melanoma
# reported as a benign nevus).
from labels import CLASSES

# TTA Augmentations applied on each Monte Carlo pass
tta_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
])

def predict_standard(model, image_tensor, device="cuda"):
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=-1)
        
    pred_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, pred_class].item()
    return {
        "pred_class": pred_class,
        "class_name": CLASSES[pred_class],
        "confidence": confidence,
        "probs": probs[0].cpu()
    }

# Default alarm thresholds per uncertainty metric. These live on very
# different numeric scales, so a threshold tuned for one is meaningless
# for another. Treat these as starting points -- re-tune against the
# valley of the correct/incorrect KDE separation plot for your data.
DEFAULT_UNC_THRESHOLDS = {
    "variance": 0.012,
    "predictive_entropy": 0.5,
    "mutual_info": 0.05,
}

def predict_bnn_mcdropout(model, image_tensor, num_samples=30, use_tta=True,
                           device="cuda", uncertainty_metric="mutual_info"):
    """
    Runs MC Dropout with optional Test-Time Augmentation (TTA).

    uncertainty_metric selects which epistemic uncertainty score is used
    for pred_class's "uncertainty" field / triage decision:
      - "variance":  mean per-class variance across MC passes (original
        metric). Simple, but treats variance from geometric TTA jitter
        the same as variance from genuine model disagreement, and is
        diluted by near-zero-probability classes.
      - "predictive_entropy": entropy of the *averaged* probability
        vector. Captures total (epistemic + aleatoric) uncertainty --
        an image that is just genuinely ambiguous will also score high
        here, not only images the model is unsure about.
      - "mutual_info": predictive_entropy minus the average per-pass
        entropy (the BALD score). This isolates epistemic uncertainty
        (disagreement *between* MC passes) from aleatoric uncertainty
        (each pass being individually unsure). Usually the metric you
        want for an OOD/artifact safeguard, since it stays low when the
        image is just inherently ambiguous but the model is consistent.

    All three are computed and returned in "uncertainty_all" regardless
    of which one drives the triage decision, so you can compare them.
    """
    model.eval()
    # Keep Dropout enabled while keeping BatchNorm/LayerNorm in eval mode
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
    samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Apply dynamic TTA per pass if enabled
            if use_tta:
                input_img = tta_transforms(image_tensor.squeeze(0)).unsqueeze(0).to(device)
            else:
                input_img = image_tensor.to(device)
                
            logits = model(input_img)
            probs = F.softmax(logits, dim=-1)
            samples.append(probs)
            
    # Stack passes to shape: (num_samples, num_classes)
    probs_tensor = torch.stack(samples, dim=0).squeeze(1)
    
    mean_probs = torch.mean(probs_tensor, dim=0)
    pred_class = torch.argmax(mean_probs).item()
    confidence = mean_probs[pred_class].item()

    # --- Metric 1: raw softmax variance (original) ---
    class_variances = torch.var(probs_tensor, dim=0)
    variance_unc = torch.mean(class_variances).item()

    # --- Metric 2 & 3: entropy-based (BALD decomposition) ---
    eps = 1e-12
    predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps)).item()
    per_pass_entropy = -torch.sum(probs_tensor * torch.log(probs_tensor + eps), dim=1)
    expected_entropy = torch.mean(per_pass_entropy).item()
    mutual_info = max(predictive_entropy - expected_entropy, 0.0)

    uncertainty_all = {
        "variance": variance_unc,
        "predictive_entropy": predictive_entropy,
        "mutual_info": mutual_info,
    }
    epistemic_uncertainty = uncertainty_all[uncertainty_metric]
    
    # Run through the Tri-Color Triage Safeguard
    triage = get_triage_status(confidence, epistemic_uncertainty, metric=uncertainty_metric)
    
    return {
        "pred_class": pred_class,
        "class_name": CLASSES[pred_class],
        "confidence": confidence,
        "uncertainty": epistemic_uncertainty,
        "uncertainty_metric": uncertainty_metric,
        "uncertainty_all": uncertainty_all,
        "mean_probs": mean_probs.cpu(),
        "triage": triage
    }

def get_triage_status(confidence, uncertainty, metric="mutual_info", unc_threshold=None):
    """
    Classifies model outputs into safety triage levels based on uncertainty.

    unc_threshold defaults to DEFAULT_UNC_THRESHOLDS[metric] if not given
    explicitly -- pass it explicitly once you've picked a real cutoff from
    the separation plots.
    """
    if unc_threshold is None:
        unc_threshold = DEFAULT_UNC_THRESHOLDS.get(metric, 0.012)

    if uncertainty > unc_threshold:
        return {
            "status": "RED LIGHT",
            "action": "REJECT - High uncertainty artifact or out-of-distribution image. Retake photo.",
            "color": "#EF4444"
        }
    elif confidence < 0.60:
        return {
            "status": "YELLOW LIGHT",
            "action": "FLAG FOR REVIEW - Low prediction confidence. Route to specialist.",
            "color": "#F59E0B"
        }
    else:
        return {
            "status": "GREEN LIGHT",
            "action": "ACCEPT - High certainty prediction.",
            "color": "#10B981"
        }