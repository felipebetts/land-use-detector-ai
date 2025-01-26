import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import keras
from PIL import Image
from data_loader import get_dataset

def visualize_prediction(model, dataset, num_samples=3):
    
    # Definições das classes
    class_labels = ["Urbano", "Agricultura", "Pastagem", "Floresta", "Água", "Descampado", "Desconhecido"]
    class_colors_rgb = [
        (0,255,255), (255,255,0), (255,0,255),
        (0,255,0),   (0,0,255),   (255,255,255),
        (0,0,0),
    ]
    class_colors = [tuple(c/255 for c in rgb) for rgb in class_colors_rgb]
    cmap = mcolors.ListedColormap(class_colors, name='LandCoverMap')
    
    num_classes = len(class_labels)
    norm = mcolors.BoundaryNorm(range(num_classes+1), num_classes)
    
    for images, masks in dataset.take(1):
        preds = model.predict(images)         # [batch, H, W, num_classes]
        preds_argmax = np.argmax(preds, axis=-1)  # [batch, H, W]
        true_masks = masks.numpy()                # [batch, H, W, 1]

        # Selecionar índices aleatórios
        random_indices = np.random.choice(len(images), size=num_samples, replace=False)

        for i in random_indices:
            plt.figure(figsize=(14,4))

            # 1) Imagem original
            plt.subplot(1,4,1)
            plt.title("Imagem")
            plt.imshow(images[i])  # assumindo [0..1] float

            # 2) Máscara verdadeira
            plt.subplot(1,4,2)
            plt.title("Máscara Real")
            plt.imshow(true_masks[i, :, :, 0], cmap=cmap, norm=norm)
            
            # Se quiser colorbar também para a máscara real:
            # cbar_real = plt.colorbar()
            # cbar_real.set_ticks(np.arange(num_classes))
            # cbar_real.set_ticklabels(class_labels)

            # 3) Predição
            plt.subplot(1,4,3)
            plt.title("Predição")
            # usar a colormap + norm
            img_plot = plt.imshow(preds_argmax[i], cmap=cmap, norm=norm)
            
            # 4) Colorbar / Legenda
            plt.subplot(1,4,4)
            plt.axis('off')  # opcional
            cbar = plt.colorbar(img_plot, ticks=range(num_classes))
            cbar.ax.set_yticklabels(class_labels)

            plt.tight_layout()
            plt.show()



def plot_image_and_mask(image_path, mask_path):
    """
    Plota uma imagem e sua máscara predita com a legenda correspondente.
    
    Args:
        image_path (str): Caminho para a imagem original.
        mask_array (np.ndarray): Máscara predita (array 2D com valores de classe).
    """
    # Definições das classes
    class_labels = ["Urbano", "Agricultura", "Pastagem", "Floresta", "Água", "Descampado", "Desconhecido"]
    class_colors_rgb = [
        (0, 255, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 0), (0, 0, 255), (255, 255, 255),
        (0, 0, 0),
    ]
    class_colors = [tuple(c/255 for c in rgb) for rgb in class_colors_rgb]
    cmap = mcolors.ListedColormap(class_colors, name='LandCoverMap')
    norm = mcolors.BoundaryNorm(range(len(class_labels) + 1), len(class_labels))

    # Carregar a imagem original
    image = Image.open(image_path)

    # Carregar a máscara
    mask_image = Image.open(mask_path)
    mask_array = np.array(mask_image)

    plt.figure(figsize=(14, 6))

    # Subplot para a imagem original
    plt.subplot(1, 3, 1)
    plt.title("Imagem de Satélite")
    plt.imshow(image, aspect='equal')
    plt.axis('off')

    # Subplot para a máscara predita
    plt.subplot(1, 3, 2)
    plt.title("Máscara")
    img_plot = plt.imshow(mask_array, cmap=cmap, norm=norm, aspect='equal')
    plt.axis('off')

    # Subplot para a legenda (colorbar)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    cbar = plt.colorbar(img_plot, ticks=range(len(class_labels)), orientation='vertical')
    cbar.ax.set_yticklabels(class_labels)

    plt.tight_layout()
    plt.show()

def plot_mask_and_label(mask_path):
    """
    Plota uma imagem e sua máscara predita com a legenda correspondente.
    
    Args:
        image_path (str): Caminho para a imagem original.
        mask_array (np.ndarray): Máscara predita (array 2D com valores de classe).
    """
    # Definições das classes
    class_labels = ["Urbano", "Agricultura", "Pastagem", "Floresta", "Água", "Descampado", 
                    # "Desconhecido"
                    ]
    class_colors_rgb = [
        (0, 255, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 0), (0, 0, 255), (255, 255, 255),
        # (0, 0, 0),
    ]
    class_colors = [tuple(c/255 for c in rgb) for rgb in class_colors_rgb]
    cmap = mcolors.ListedColormap(class_colors, name='LandCoverMap')
    norm = mcolors.BoundaryNorm(range(len(class_labels) + 1), len(class_labels))

    # Carregar a imagem original
    # image = Image.open(image_path)

    # Carregar a máscara
    mask_image = Image.open(mask_path)
    mask_array = np.array(mask_image)

    plt.figure(figsize=(14, 6))

    # # Subplot para a imagem original
    # plt.subplot(1, 3, 1)
    # plt.title("Imagem de Satélite")
    # plt.imshow(image)
    # plt.axis('off')

    # Subplot para a máscara predita
    plt.subplot(1, 3, 2)
    plt.title("Máscara")
    img_plot = plt.imshow(mask_array, cmap=cmap, norm=norm)
    plt.axis('off')

    # Subplot para a legenda (colorbar)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    cbar = plt.colorbar(img_plot, ticks=range(len(class_labels)), orientation='vertical')
    cbar.ax.set_yticklabels(class_labels)

    plt.tight_layout()
    plt.show()


# saved_model = keras.models.load_model('trained_models/model_30_epochs.h5')
# _, test_dataset = get_dataset()

# # # Visualizar alguns resultados
# visualize_prediction(saved_model, test_dataset, num_samples=6)


def main():
    model = keras.models.load_model('trained_models/model_30_epochs.h5')
    _, test_dataset = get_dataset()
    visualize_prediction(model, test_dataset, num_samples=6)


if __name__ == "__main__":
    # image_id = "21717"
    # image_path = f"/home/felipebetts/.cache/kagglehub/datasets/balraj98/deepglobe-land-cover-classification-dataset/versions/2/train/{image_id}_sat.jpg"
    # mask_path = f"/home/felipebetts/.cache/kagglehub/datasets/balraj98/deepglobe-land-cover-classification-dataset/versions/2/train/{image_id}_mask.png"
    # plot_image_and_mask(image_path, mask_path)

    # mask_path = "exports/model_batch_12_epochs_250_v3_predictions/model_batch_12_epochs_250_v3_cropped.png"
    # plot_mask_and_label(mask_path)

    # tile = 'tile_15_3'
    # image_path = f"exports/model_batch_12_epochs_250_v3_predictions/originals/{tile}.png"
    # mask_path = f"exports/model_batch_12_epochs_250_v3_predictions/masks/{tile}.png"
    # plot_image_and_mask(image_path, mask_path)
    main()