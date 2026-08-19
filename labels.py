# Single source of truth for HAM10000 class ordering.
#
# Every other file (train.py, inference_engine.py, comapre.py) imports
# from here instead of building its own label map. Previously,
# train.py derived label indices from sorted(dx_code) at runtime while
# inference_engine.py hardcoded a *different* index order for display
# names -- indices 4 and 5 (melanoma / nevus) ended up swapped between
# training and display. Fixing that bug by hand in one file would just
# reintroduce it the next time someone edits the other file, so instead
# there is exactly one place this ordering is defined.

# dx code -> human-readable class name, in the fixed index order used
# everywhere: training labels, model output logits, and display names
# all agree with this order.
DX_CODE_TO_NAME = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "vasc": "Vascular lesions",
}

DX_CODES = list(DX_CODE_TO_NAME.keys())     # fixed order, e.g. index 4 == "nv"
CLASSES = list(DX_CODE_TO_NAME.values())    # fixed order, e.g. index 4 == "Melanocytic nevi"
DX_CODE_TO_LABEL = {code: i for i, code in enumerate(DX_CODES)}
NUM_CLASSES = len(DX_CODES)

# Display copy for the Streamlit UI (hover cards / chart tooltips).
# Keys must match CLASSES exactly. This is not used for training labels.
DISEASE_DETAILS = {
    "Actinic keratoses": {
        "code": "akiec",
        "also_called": "Solar keratosis; some cases are squamous cell carcinoma in situ (Bowen disease).",
        "category": "Precancerous",
        "summary": (
            "Rough, scaly patches caused by chronic sun damage. They are not invasive "
            "cancer yet, but a subset can progress to squamous cell carcinoma."
        ),
        "looks_like": "Dry, sandpaper-like pink or brown patches, often on the face, scalp, or hands.",
        "why_it_matters": "Worth clinical follow-up because of the risk of progression.",
    },
    "Basal cell carcinoma": {
        "code": "bcc",
        "also_called": "BCC; the most common form of skin cancer.",
        "category": "Malignant",
        "summary": (
            "A slow-growing skin cancer that rarely spreads to distant organs, but can "
            "invade nearby tissue if left untreated."
        ),
        "looks_like": "Pearly or translucent bump, pink patch, or a sore that does not heal; may show rolled edges or tiny blood vessels.",
        "why_it_matters": "Usually highly treatable when caught early; still needs definitive care.",
    },
    "Benign keratosis": {
        "code": "bkl",
        "also_called": "Seborrheic keratosis, solar lentigo, or lichen-planus-like keratosis.",
        "category": "Benign",
        "summary": (
            "A group of non-cancerous keratinocyte growths. They can look irregular "
            "and are a common source of confusion with melanoma."
        ),
        "looks_like": "Stuck-on waxy plaques, sun spots, or inflamed scaly patches.",
        "why_it_matters": "Typically harmless, but atypical appearance still warrants a specialist look.",
    },
    "Dermatofibroma": {
        "code": "df",
        "also_called": "Benign fibrous histiocytoma.",
        "category": "Benign",
        "summary": (
            "A small, firm, fibrous nodule in the dermis, often after a minor injury "
            "such as an insect bite."
        ),
        "looks_like": "A hard, dimpling bump that is brown, pink, or skin-colored; the 'dimple sign' when pinched is classic.",
        "why_it_matters": "Almost always benign; biopsy is sometimes done only to rule out other lesions.",
    },
    "Melanocytic nevi": {
        "code": "nv",
        "also_called": "Common moles (including atypical / dysplastic nevi in this dataset grouping).",
        "category": "Benign",
        "summary": (
            "Clusters of melanocytes that form moles. Most are harmless, and they are "
            "by far the most common class in HAM10000."
        ),
        "looks_like": "Round or oval brown spots with relatively even color and a stable shape over time.",
        "why_it_matters": "Changing, asymmetric, or multi-colored moles should be checked to exclude melanoma.",
    },
    "Melanoma": {
        "code": "mel",
        "also_called": "Malignant melanoma.",
        "category": "Malignant",
        "summary": (
            "Cancer of melanocytes. It is less common than BCC but much more likely "
            "to spread, which is why early detection matters."
        ),
        "looks_like": "An evolving mole that may be asymmetric, have irregular borders, mixed colors, or a diameter larger than about 6 mm (ABCDE rule).",
        "why_it_matters": "Highest-stakes class in this demo — high model uncertainty should trigger human review, not a confident auto-label.",
    },
    "Vascular lesions": {
        "code": "vasc",
        "also_called": "Cherry angiomas, angiokeratomas, pyogenic granulomas, and similar blood-vessel lesions.",
        "category": "Usually benign",
        "summary": (
            "Growths made of blood vessels rather than pigment cells. Most are benign, "
            "though some bleed easily or grow quickly."
        ),
        "looks_like": "Bright red, purple, or blue papules; may blanch or bleed.",
        "why_it_matters": "Usually not cancer, but can mimic pigmented lesions on a photo, so the model may be uncertain.",
    },
}

# Known HAM10000 class counts (from the dataset's metadata distribution),
# keyed by dx code so they can never end up in the wrong array position
# regardless of how DX_CODES is ordered above.
HAM_COUNTS_BY_CODE = {
    "akiec": 327,
    "bcc": 514,
    "bkl": 1099,
    "df": 115,
    "nv": 6705,
    "mel": 1113,
    "vasc": 142,
}
# Same fixed order as DX_CODES/CLASSES -- safe to index directly against
# model outputs or CLASSES.
HAM_COUNTS = [HAM_COUNTS_BY_CODE[code] for code in DX_CODES]