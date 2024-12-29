import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import keras
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

        for i in range(num_samples):
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


saved_model = keras.models.load_model('trained_models/meu_modelo_v2_20_epochs.h5')
_, test_dataset = get_dataset()

# Visualizar alguns resultados
visualize_prediction(saved_model, test_dataset, num_samples=6)
