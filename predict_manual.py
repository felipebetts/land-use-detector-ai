import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from constants import TILE_SIZE

# Definições das classes
class_labels = ["Urbano", "Agricultura", "Pastagem", "Floresta", "Água", "Descampado", "Desconhecido"]
class_colors_rgb = [
    (0,255,255), (255,255,0), (255,0,255),
    (0,255,0),   (0,0,255),   (255,255,255),
    (0,0,0),
]
class_colors = [tuple(c/255 for c in rgb) for rgb in class_colors_rgb]
cmap = mcolors.ListedColormap(class_colors, name='LandCoverMap')
norm = mcolors.BoundaryNorm(range(len(class_labels)+1), len(class_labels))

# Carregar o modelo
model = load_model('trained_models/meu_modelo.h5')

# Função para processar imagens PNG
def process_image_png(image_path):
    img = Image.open(image_path).convert('RGB')
    print(f"Dimensões da imagem original: {img.size}")
    img = img.resize((TILE_SIZE, TILE_SIZE))  # Ajuste ao tamanho de entrada do modelo
    img_array = np.array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão do batch
    return img_array

# Função para processar imagens TIF
def process_image_tif(image_path):
    with rasterio.open(image_path) as src:
        print(f"Dimensões da imagem original: {src.width}x{src.height}")
        img = src.read([4, 3, 2])  # Carregar bandas RGB
        img = np.moveaxis(img, 0, -1)  # Rearranjar para [Altura, Largura, Bandas]
        img = np.clip(img, 0, 255) / 255.0  # Normalizar
        img = np.resize(img, (TILE_SIZE, TILE_SIZE, 3))  # Redimensionar
    return np.expand_dims(img, axis=0)  # Adicionar dimensão do batch

# Função para fazer predição e visualizar o resultado
def predict_and_visualize(image_path, process_function):
    # Processar a imagem
    input_image = process_function(image_path)

    # Fazer a predição
    prediction = model.predict(input_image)
    predicted_class = np.argmax(prediction, axis=-1)

    # Visualizar
    plt.figure(figsize=(14, 6))

    # Imagem original
    if process_function == process_image_png:
        original_image = Image.open(image_path)
        plt.subplot(1, 3, 1)
        plt.title("Imagem Original (PNG)")
        plt.imshow(original_image)
        plt.axis('off')
    elif process_function == process_image_tif:
        with rasterio.open(image_path) as src:
            original_image = src.read([4, 3, 2])
            original_image = np.moveaxis(original_image, 0, -1)
            plt.subplot(1, 3, 1)
            plt.title("Imagem Original (TIF)")
            plt.imshow(original_image / 255.0)
            plt.axis('off')

    # Máscara predita
    plt.subplot(1, 3, 2)
    plt.title("Máscara Predita")
    img_plot = plt.imshow(predicted_class[0], cmap=cmap, norm=norm)
    plt.axis('off')

    # Legenda (Colorbar)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    cbar = plt.colorbar(img_plot, ticks=range(len(class_labels)), orientation='vertical')
    cbar.ax.set_yticklabels(class_labels)

    plt.tight_layout()
    plt.show()

# # Caminho da imagem (substitua pelos seus arquivos)
# image_path_png = 'exports/Composto_SENTINEL_2_Piabanha_2023-01-01_to_2023-12-31_tile_7.png'
# image_path_tif = 'exports/Composto_SENTINEL_2_Piabanha_2023-01-01_to_2023-12-31_tile_7.tif'

# Caminho da imagem (substitua pelos seus arquivos)
image_path_png = 'mosaico_tiles/tile_0_1.png'
image_path_tif = 'exports/Composto_SENTINEL_2_Piabanha_2023-01-01_to_2023-12-31_tile_7.tif'

# Predição para PNG
print("Predição para imagem PNG:")
predict_and_visualize(image_path_png, process_image_png)

# Predição para TIF
print("Predição para imagem TIF:")
predict_and_visualize(image_path_tif, process_image_tif)