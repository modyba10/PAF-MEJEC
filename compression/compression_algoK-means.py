import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

""" le changement defini la qualité de compression """
def compress_image(image_path, k):
    # Chargement de l'image
    image = Image.open(image_path)

    # Conversion de l'image en un tableau numpy
    pixels = np.array(image)

    # Obtention des dimensions de l'image
    width, height, _ = pixels.shape

    # Aplatir l'image en une liste de pixels
    flattened_pixels = pixels.reshape(width * height, -1)

    # Appliquer l'algorithme K-means
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(flattened_pixels)

    # Assigner chaque pixel au centroïde le plus proche
    compressed_pixels = kmeans.labels_

    # Récupération des valeurs des centroïdes
    centroid_values = kmeans.cluster_centers_

    # Création de la nouvelle image compressée
    compressed_image = centroid_values[compressed_pixels].reshape(width, height, -1)

    # Conversion de l'array numpy en image PIL
    compressed_image = Image.fromarray(np.uint8(compressed_image))

    # Retourner l'image compressée
    return compressed_image


# Test de la fonction compress_image
image_path = 'archive-3/ISIC_0541318.JPG'  # Chemin vers l'image à compresser
k = 8  # Nombre de clusters

# Appel de la fonction compress_image
compressed_image = compress_image(image_path, k)

# Sauvegarde de l'image compressée
compressed_image.save('image_compressée2.jpg')
