from monai.transforms import LoadImage
from monai.data.image_reader import PILReader

# Exemple de chemin vers l'image
image_path = "/work/bigo/data/Result_mapper_slicerSALT/Result/119906/V06/EACSF/left_eacsf_flat.png"

# Charger l'image avec PIL via Monai
loader = LoadImage(image_only=True, reader=PILReader())

# Essayer de charger l'image
try:
    image = loader(image_path)
    print(f"Image {image_path} chargée avec succès.")
    print(f"Forme de l'image : {image.shape}")  # Afficher la forme de l'image
except Exception as e:
    print(f"Erreur lors du chargement de l'image {image_path}: {e}")
