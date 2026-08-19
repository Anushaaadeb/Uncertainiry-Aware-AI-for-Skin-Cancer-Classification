# Central model/resolution config. Every file that builds the model or
# its preprocessing transform imports from here, so switching backbones
# is a one-line change instead of hunting down hardcoded "224" and
# "efficientnet_b0" across train.py / inference_engine.py / app.py /
# comapre.py.
#
# IMPORTANT: if you change these, you must retrain -- a checkpoint
# trained with one backbone/resolution will not load into a different
# one (different feature dimensions), and old fine_tuned_model.pth
# files become incompatible.

BACKBONE_NAME = "efficientnet_b2"  
IMG_SIZE = 260                    