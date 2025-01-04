import os
import math
import matplotlib.pyplot as plt

from data_loader import BATCH_SIZE
from model import get_compiled_model

EPOCHS = 20  # Ajuste conforme necessidade e recursos de hardware
N = 642

def train_model(train_dataset, test_dataset, trained_model_name, epochs=EPOCHS, predictions_folder="exports"):
    model = get_compiled_model()
    steps_per_epoch = math.ceil(N / BATCH_SIZE)
    
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
    )

    # Avaliação final no conjunto de validação
    val_loss, val_acc = model.evaluate(test_dataset)
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")

    output_path = os.path.join('trained_models', f"{trained_model_name}.h5")
    model.save(output_path)  # Formato HDF5

    # Gerar gráficos
    generate_training_plots(history, predictions_folder)

    # Exportar histórico para CSV
    export_training_history_to_csv(history, predictions_folder)

    return output_path

def generate_training_plots(history, output_folder):
    """Gera gráficos de perda e acurácia com base no histórico do treinamento."""
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    os.makedirs(output_folder, exist_ok=True)

    # Gráfico da Perda
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label='Perda de Treinamento')
    plt.plot(epochs, val_loss, label='Perda de Validação')
    plt.title('Evolução da Perda')
    plt.xlabel('Epochs')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid()
    loss_plot_path = os.path.join(output_folder, "grafico_perda.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfico da Acurácia
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy, label='Acurácia de Treinamento')
    plt.plot(epochs, val_accuracy, label='Acurácia de Validação')
    plt.title('Evolução da Acurácia')
    plt.xlabel('Epochs')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid()
    accuracy_plot_path = os.path.join(output_folder, "grafico_precisao.png")
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def export_training_history_to_csv(history, output_folder):
    """Exporta o histórico de treinamento para um arquivo CSV."""
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    os.makedirs(output_folder, exist_ok=True)

    csv_path = os.path.join(output_folder, "training_history.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss", "Validation Loss", "Accuracy", "Validation Accuracy"])
        for epoch, l, vl, acc, vacc in zip(epochs, loss, val_loss, accuracy, val_accuracy):
            writer.writerow([epoch, l, vl, acc, vacc])
    print(f"Histórico de treinamento salvo em: {csv_path}")