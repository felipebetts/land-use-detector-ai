import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from data_loader import get_dataset


# Definições das classes e cores
class_labels = [
    "Urbano", "Agricultura", "Pastagem", "Floresta", "Água", "Descampado", "Desconhecido"
]
num_classes = len(class_labels)

def calculate_metrics(y_true, y_pred, num_classes):
    """
    Calcula métricas de avaliação para segmentação semântica.

    Args:
        y_true: Ground truth (forma [batch_size, height, width])
        y_pred: Predições do modelo (forma [batch_size, height, width])
        num_classes: Número de classes no modelo.

    Returns:
        metrics: Dicionário com IoU, F1-Score, Pixel Accuracy e Mean Pixel Accuracy.
    """
    metrics = {}
    
    # Flatten para cálculo de métricas pixel-wise
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Matriz de Confusão
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(num_classes))
    
    # IoU por classe
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou_per_class = intersection / (union + 1e-6)  # Evita divisão por zero

    metrics['IoU'] = iou_per_class
    metrics['Mean IoU'] = np.mean(iou_per_class)

    # F1-Score por classe
    precision = np.diag(cm) / (cm.sum(axis=0) + 1e-6)
    recall = np.diag(cm) / (cm.sum(axis=1) + 1e-6)
    f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-6)

    metrics['F1-Score'] = f1_per_class
    metrics['Mean F1-Score'] = np.mean(f1_per_class)

    # Pixel Accuracy
    metrics['Pixel Accuracy'] = np.sum(intersection) / np.sum(cm)

    # Mean Pixel Accuracy
    class_wise_accuracy = intersection / (cm.sum(axis=1) + 1e-6)
    metrics['Mean Pixel Accuracy'] = np.mean(class_wise_accuracy)

    return metrics, cm

def plot_metrics(metrics, cm, class_labels, output_folder="plots"):
    """
    Gera gráficos para visualizar as métricas e a matriz de confusão e os salva.

    Args:
        metrics: Dicionário com as métricas calculadas.
        cm: Matriz de confusão.
        class_labels: Lista de nomes das classes.
        output_folder: Caminho da pasta para salvar os gráficos.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Plot IoU por classe
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(class_labels)), metrics['IoU'], tick_label=class_labels)
    plt.title("IoU por Classe")
    plt.ylabel("IoU")
    plt.xlabel("Classes")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    iou_path = os.path.join(output_folder, "iou_por_classe.png")
    plt.savefig(iou_path)
    plt.close()

    # Plot F1-Score por classe
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(class_labels)), metrics['F1-Score'], tick_label=class_labels)
    plt.title("F1-Score por Classe")
    plt.ylabel("F1-Score")
    plt.xlabel("Classes")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    f1_path = os.path.join(output_folder, "f1_score_por_classe.png")
    plt.savefig(f1_path)
    plt.close()

    # Plot Matriz de Confusão
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    # Adicionar valores na matriz
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Classe Verdadeira")
    plt.xlabel("Classe Predita")
    plt.tight_layout()
    cm_path = os.path.join(output_folder, "matriz_de_confusao.png")
    plt.savefig(cm_path)
    plt.close()

def evaluate_model_on_dataset(model, dataset):
    """
    Avalia o modelo em um dataset de teste e calcula as métricas adicionais.

    Args:
        model: Modelo U-Net treinado.
        dataset: Dataset de teste no formato tf.data.Dataset.

    Returns:
        metrics: Dicionário com as métricas calculadas.
        cm: Matriz de confusão.
    """
    y_true_list = []
    y_pred_list = []

    for images, masks in dataset:
        # Predição do modelo
        preds = model.predict(images)
        preds_argmax = np.argmax(preds, axis=-1)  # Obtém a classe predita para cada pixel

        # Coleta de rótulos verdadeiros e predições
        y_true_list.append(masks.numpy().squeeze())
        y_pred_list.append(preds_argmax)

    # Concatena todas as imagens do dataset
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    # Calcula métricas
    metrics, cm = calculate_metrics(y_true, y_pred, num_classes)
    return metrics, cm

# Exemplo de uso
if __name__ == "__main__":
    model_name = "model_batch_12_epochs_250_v3"
    # Carrega modelo e dataset de teste
    saved_model_path = f"trained_models/{model_name}.h5"
    model = tf.keras.models.load_model(saved_model_path)
    train_dataset, test_dataset = get_dataset()

    # Avalia o modelo
    metrics, cm = evaluate_model_on_dataset(model, train_dataset)

    # Exibe resultados
    print("\n=== Métricas de Avaliação ===")
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.tolist()}")
        else:
            print(f"{key}: {value:.4f}")

    print("\n=== Matriz de Confusão ===")
    print(cm)

    # Gera gráficos
    output_folder = os.path.join('exports', f"{model_name}_predictions", "evaluate") 
    plot_metrics(metrics, cm, class_labels, output_folder)
