from monai.transforms import Compose, LoadImage, EnsureChannelFirst, NormalizeIntensity, EnsureType, RandRotate, RandFlip, RandZoom
import os
import torch
from monai.data.image_reader import PILReader
import numpy as np

# Définir le pipeline de transformations
transform_pipeline = Compose([
    EnsureChannelFirst(channel_dim='no_channel'),  # Assure que le canal est bien au début
    NormalizeIntensity(),  # Normalisation de l'intensité
    RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),  # Rotation aléatoire
    RandFlip(spatial_axis=0, prob=0.5),  # Flip aléatoire
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),  # Zoom aléatoire
    EnsureType(data_type="tensor"),  # Conversion explicite en torch.Tensor
])


image_files = [
    "/work/bigo/data/Result_mapper/101247/V06/eacsf/101247_V06_right_EACSF_Smoothed_flat.png", 
   
]


for idx, image_path in enumerate(image_files):
    try:
        # Vérifier si le fichier existe
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Charger l'image (sans métadonnées)
        loader = LoadImage(image_only=True, reader=PILReader())
        image = loader(image_path)
        print(f"Image {idx} loaded successfully with shape {image.shape} (type: {type(image)})")
        
        
        transformed_data = transform_pipeline(image)
        print(f"Transformed data for image {idx}: {transformed_data.shape} (type: {type(transformed_data)})")
    
    except Exception as e:
        print(f"Error with image {idx} ({image_path}): {e}")
