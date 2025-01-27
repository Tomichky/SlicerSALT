import os

def rename_files_in_directory(directory):
    # Parcourt tous les sous-dossiers et fichiers du dossier
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Vérifie si 'eacsf' est dans le nom du fichier
            if 'eacsf' in file:
                # Crée le nouveau nom avec 'EACSF' à la place de 'eacsf'
                new_name = file.replace('eacsf', 'EACSF')
                # Crée le chemin complet du fichier actuel et du nouveau fichier
                old_file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, new_name)
                # Renomme le fichier
                os.rename(old_file_path, new_file_path)
                print(f"Renommé : {old_file_path} -> {new_file_path}")

# Exemple d'utilisation
directory = '/work/bigo/data/Result_mapper_slicerSALT/Result'  # Remplacez par le chemin de votre dossier
rename_files_in_directory(directory)
