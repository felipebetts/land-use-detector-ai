import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

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

    # Erro Quadrático Médio (MSE)
    metrics['MSE'] = mean_squared_error(y_true_flat, y_pred_flat)

    # Coeficiente de Correlação de Pearson (PCC)
    y_true_mean = np.mean(y_true_flat)
    y_pred_mean = np.mean(y_pred_flat)
    numerator = np.sum((y_true_flat - y_true_mean) * (y_pred_flat - y_pred_mean))
    denominator = np.sqrt(np.sum((y_true_flat - y_true_mean)**2) * np.sum((y_pred_flat - y_pred_mean)**2))
    metrics['PCC'] = numerator / (denominator + 1e-6)  # Evita divisão por zero

    # Matriz Kappa
    total = np.sum(cm)
    po = np.trace(cm) / total
    pe = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (total**2)
    metrics['Kappa'] = (po - pe) / (1 - pe + 1e-6)

    # Sensibilidade (Recall) Média
    metrics['Mean Recall'] = np.mean(recall)

    # Especificidade por classe
    tn = np.sum(cm) - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    fp = cm.sum(axis=0) - np.diag(cm)
    specificity_per_class = tn / (tn + fp + 1e-6)
    metrics['Specificity'] = specificity_per_class
    metrics['Mean Specificity'] = np.mean(specificity_per_class)

    # Total de Pixels por Classe
    total_pixels_per_class = cm.sum(axis=0)
    metrics['Total Pixels per Class'] = total_pixels_per_class

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

    # Tabelas
    metrics_df = pd.DataFrame({
        "Class": class_labels,
        "IoU": metrics['IoU'],
        "F1-Score": metrics['F1-Score'],
        "Specificity": metrics['Specificity'],
        "Mean Recall": [metrics['Mean Recall']] * len(class_labels),
        "Kappa": [metrics['Kappa']] * len(class_labels),
        "PCC": [metrics['PCC']] * len(class_labels),
        "Total Pixels": metrics['Total Pixels per Class']
    })
    metrics_df.to_csv(os.path.join(output_folder, "metrics_table.csv"), index=False)

    # Plotar tabela das métricas
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center')
    plt.title("Tabela de Métricas")
    plt.savefig(os.path.join(output_folder, "metrics_table_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

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
    cm_percent = cm / np.sum(cm, axis=1, keepdims=True) * 100

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão (%)")
    plt.colorbar(format='%.2f%%')
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    # Total Pixels por Classe
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(class_labels)), metrics['Total Pixels per Class'], tick_label=class_labels)
    plt.title("Total de Pixels por Classe")
    plt.ylabel("Total de Pixels")
    plt.xlabel("Classes")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    total_pixels_path = os.path.join(output_folder, "total_pixels_per_class.png")
    plt.savefig(total_pixels_path)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.pie(metrics['Total Pixels per Class'], labels=class_labels, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.title("Distribuição de Pixels por Classe")
    pie_chart_path = os.path.join(output_folder, "total_pixels_per_class_pie.png")
    plt.savefig(pie_chart_path)
    plt.close()



    # Adicionar valores na matriz como percentual
    thresh = cm_percent.max() / 2.
    for i, j in np.ndindex(cm_percent.shape):
        plt.text(j, i, f"{cm_percent[i, j]:.2f}%",
                 horizontalalignment="center",
                 color="white" if cm_percent[i, j] > thresh else "black")

    plt.ylabel("Classe Verdadeira")
    plt.xlabel("Classe Predita")
    plt.tight_layout()
    cm_path = os.path.join(output_folder, "matriz_de_confusao_percentual.png")
    plt.savefig(cm_path)
    plt.close()

    # Salvar a matriz de confusão como tabela
    cm_df = pd.DataFrame(cm_percent, index=class_labels, columns=class_labels)
    cm_df.to_csv(os.path.join(output_folder, "matriz_de_confusao_percentual.csv"))

def plot_pixels_and_accuracy(metrics, cm, class_labels, output_folder="plots"):
    """
    Plota um gráfico com dois eixos: quantidade de pixels por classe e porcentagem de acerto da matriz de confusão normalizada.

    Args:
        metrics: Dicionário com as métricas calculadas.
        cm: Matriz de confusão.
        class_labels: Lista de nomes das classes.
        output_folder: Caminho da pasta para salvar os gráficos.
    """
    os.makedirs(output_folder, exist_ok=True)

    total_pixels = metrics['Total Pixels per Class']
    accuracy_per_class = np.diag(cm) / (cm.sum(axis=1) + 1e-6) * 100
    f1_scores = metrics['F1-Score']
    iou_scores = metrics['IoU']

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Total de Pixels', color=color)
    ax1.bar(class_labels, total_pixels, color=color, alpha=0.6, label='Total de Pixels')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Porcentagem de Acerto (%)', color=color)
    ax2.plot(class_labels, accuracy_per_class, color=color, marker='o', label='Porcentagem de Acerto')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('Total de Pixels por Classe vs Porcentagem de Acerto')
    fig.tight_layout()
    plt.savefig(os.path.join(output_folder, "pixels_vs_accuracy.png"), dpi=300)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('F1-Score', color=color)
    ax1.bar(class_labels, f1_scores, color=color, alpha=0.6, label='F1-Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('IoU', color=color)
    ax2.plot(class_labels,total_pixels, color=color, marker='o', label='Total de Pixels')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('F1-Score vs Total de Pixels por Classe')
    fig.tight_layout()
    plt.savefig(os.path.join(output_folder, "pixels_vs_f1.png"), dpi=300)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('IoU', color=color)
    ax1.bar(class_labels, iou_scores, color=color, alpha=0.6, label='IoU')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('IoU', color=color)
    ax2.plot(class_labels,total_pixels, color=color, marker='o', label='Total de Pixels')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('IoU vs Total de Pixels por Classe')
    fig.tight_layout()
    plt.savefig(os.path.join(output_folder, "pixels_vs_iou.png"), dpi=300)
    plt.close()

    

def evaluate_model_on_dataset(model, dataset):
    """
    Avalia o modelo em um dataset de teste e calcula as métricas adicionais.

    Args:
        model: Modelo treinado.
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
    metrics, cm = evaluate_model_on_dataset(model, test_dataset)

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
    # output_folder = os.path.join('exports', f"{model_name}_predictions", "evaluate") 
    output_folder = os.path.join('exports', f"{model_name}_evaluations", "evaluate") 
    plot_metrics(metrics, cm, class_labels, output_folder)
    plot_pixels_and_accuracy(metrics, cm, class_labels, output_folder)
