import os
import math
from data_loader import BATCH_SIZE
from model import get_compiled_model

EPOCHS = 20  # Ajuste conforme necessidade e recursos de hardware
N = 642

def train_model(train_dataset, test_dataset, trained_model_name, epochs=EPOCHS):
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
    return output_path
