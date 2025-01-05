import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



from divide_mosaic import divide_image_with_padding, recompose_image

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
model_default = tf.keras.models.load_model('trained_models/meu_modelo_v2.h5')

def file_basename(path):
    return os.path.splitext(os.path.basename(path))[0]

def process_image_png(image_path):
    img = Image.open(image_path).convert('RGB')
    print(f"Dimensões da imagem original: {img.size}")
    # img = img.resize((256, 256))  # Ajuste ao tamanho de entrada do modelo
    img_array = np.array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão do batch
    return img_array


def visualize(image_path, predicted_class):
    # Visualizar
    plt.figure(figsize=(14, 6))

    # Imagem original
    original_image = Image.open(image_path)
    plt.subplot(1, 3, 1)
    plt.title("Imagem Original (PNG)")
    plt.imshow(original_image)
    plt.axis('off')

    # Máscara predita
    plt.subplot(1, 3, 2)
    plt.title("Máscara Predita")
    img_plot = plt.imshow(predicted_class, cmap=cmap, norm=norm)
    plt.axis('off')

    # Legenda (Colorbar)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    cbar = plt.colorbar(img_plot, ticks=range(len(class_labels)), orientation='vertical')
    cbar.ax.set_yticklabels(class_labels)

    plt.tight_layout()
    plt.show()

def save_mask_as_png(mask, output_path):
    """
    Salva a máscara como uma imagem PNG.
    :param mask: Array 2D representando as classes preditas.
    :param output_path: Caminho onde o arquivo PNG será salvo.
    """
    # Cria uma imagem RGB aplicando o mapeamento de cores
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for idx, color in enumerate(class_colors_rgb):
        color_mask[mask == idx] = color

    # Converte para uma imagem PIL e salva
    image = Image.fromarray(color_mask)
    image.save(output_path)
    print(f"Máscara salva em: {output_path}")

def predict(input_path, model=model_default):
    output_folder = os.path.join('exports', f"{file_basename(input_path)}_predictions")
    originals_folder = os.path.join(output_folder, "originals")
    tiles_paths = divide_image_with_padding(input_path, originals_folder)
    print('tiles_paths:', tiles_paths)

    masks_folder = os.path.join(output_folder, 'masks')
    os.makedirs(masks_folder, exist_ok=True)
    
    masks_paths = []

    for tile_path in tiles_paths:
        tile_image = process_image_png(tile_path)

        prediction = model.predict(tile_image)
        predicted_class = np.argmax(prediction, axis=-1)[0]  # Remove dimensão extra
        print(f"Predição realizada para o tile: {tile_path}")

        # Extrair o nome base do arquivo sem extensão
        base_name = file_basename(tile_path)

        output_png_path = os.path.join(output_folder, "masks", f"{base_name}.png")

        # Salvar a máscara predita como PNG
        save_mask_as_png(predicted_class, output_png_path)
        masks_paths.append(output_png_path)

        # Visualizar (opcional)
        # visualize(tile_path, predicted_class)
    output_path = os.path.join(output_folder, 'output.png')
    recompose_image(masks_folder, output_path, (1668,4469))
    return output_path




if __name__ == '__main__':
    predict('exports/mosaico_final.png')