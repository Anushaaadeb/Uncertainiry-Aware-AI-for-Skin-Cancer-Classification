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