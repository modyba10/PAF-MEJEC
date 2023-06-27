from PIL import Image


def compress_jpeg(input_image_path, output_image_path, quality):
    # Ouvrir l'image en utilisant Pillow
    image = Image.open(input_image_path)

    # Convertir l'image en mode RVB si elle n'est pas déjà en RVB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Effectuer la compression JPEG en enregistrant l'image avec la qualité spécifiée
    image.save(output_image_path, format="JPEG", quality=quality,subsampling=1)


# Chemins d'entrée et de sortie des images
input_path = "/Users/modyba/Desktop/PAF_MEJEC-COMPRESSION/essai2.jpg"
output_path = "essai2_compressed_jpeg_quality=75.jpeg"

# Qualité de compression (valeur entre 1 et 95)
quality =75

# Appeler la fonction de compression JPEG
compress_jpeg(input_path, output_path, quality)
