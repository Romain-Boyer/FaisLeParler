import os

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, "data")
MODELS_PATH = os.path.join(CURRENT_PATH, "models")

LACLASSEAMERICAINE_PATH = os.path.join(DATA_PATH, "laclasseamericaine.csv")

WIKIFR_PATH = os.path.join(MODELS_PATH, "wiki.fr.vec")
