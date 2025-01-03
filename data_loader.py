import os
import glob
import tensorflow as tf
import kagglehub
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2  # Fração do dataset para TEST
BATCH_SIZE = 8   # Ajuste conforme necessidade
COLOR_MAP = {
    (0,   255, 255): 0,  # Urban land
    (255, 255, 0  ): 1,  # Agriculture
    (255, 0,   255): 2,  # Rangeland
    (0,   255, 0  ): 3,  # Forest
    (0,   0,   255): 4,  # Water
    (255, 255, 255): 5,  # Barren
    (0,   0,   0  ): 6,  # Unknown
}


def dowload_dataset(destination_folder="datasets"):
    """
    Faz o download do dataset do Kaggle e salva na pasta especificada.
    
    :param destination_folder: Caminho para a pasta onde o dataset será salvo.
    :return: Caminho do dataset baixado.
    """
    os.makedirs(destination_folder, exist_ok=True)  # Cria a pasta, se necessário
    path = kagglehub.dataset_download(
        "balraj98/deepglobe-land-cover-classification-dataset", 
        path=destination_folder
    )
    print(f"Dataset baixado em: {path}")
    return path

def load_image_mask_pairs(base_path):
    sat_files = sorted(glob.glob(os.path.join(base_path, "*_sat.jpg")))

    pairs = []
    for sat_file in sat_files:
        mask_file = sat_file.replace("_sat.jpg", "_mask.png")
        
        if os.path.exists(mask_file):
            pairs.append((sat_file, mask_file))
        else:
            print(f"Aviso: A máscara para '{sat_file}' não foi encontrada em '{mask_file}'.")

    return pairs

def remap_mask_rgb(mask_rgb):
    """
    1) Binariza cada canal em <128 => 0, caso contrário => 255.
    2) Para cada pixel, converte (R,G,B) => índice de classe [0..6].
    """
    binarized = tf.where(mask_rgb < 128, 0, 255)  # [H, W, 3]

    h, w = tf.shape(binarized)[0], tf.shape(binarized)[1]
    mask_indices = tf.zeros([h, w], dtype=tf.uint8)

    for (r_val, g_val, b_val), class_idx in COLOR_MAP.items():
        cond = tf.logical_and(
            tf.equal(binarized[..., 0], r_val),
            tf.logical_and(
                tf.equal(binarized[..., 1], g_val),
                tf.equal(binarized[..., 2], b_val)
            )
        )
        # Ajuste: converter class_idx para uint8
        mask_indices = tf.where(cond, tf.cast(class_idx, tf.uint8), mask_indices)

    # Expande para [H, W, 1]
    return mask_indices[..., tf.newaxis]


def parse_image_mask(img_path, mask_path, target_size=(256, 256)):
    """
    Lê a imagem e a máscara do disco, redimensiona e normaliza para 0-1.
    """
    # Imagem (RGB) normalizada
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0

    # Máscara (RGB) -> binariza -> remapeia -> [H,W,1]
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.image.resize(mask, target_size, method='nearest')
    mask = remap_mask_rgb(mask)
    mask = tf.cast(mask, tf.uint8)  # final [H,W,1], cada pixel [0..6]

    return img, mask

def augment_image_mask(img, mask):
    # Exemplo simples: flip horizontal
    # Use um random para ter 50% de chance de flip
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    # Adicionar outras transforms, p.ex. random brightness, rotations...
    return img, mask

def build_tf_dataset(image_mask_list, batch_size=8, shuffle=True, buffer_size=100):
    """
    Constrói tf.data.Dataset a partir de uma lista de pares (img_path, mask_path).
    """
    def generator():
        for (img_path, mask_path) in image_mask_list:
            yield (img_path, mask_path)
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.string, tf.string),
        output_shapes=((), ())
    )
    # Mapeia para (img_tensor, mask_tensor)
    dataset = dataset.map(lambda x, y: parse_image_mask(x, y), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    # APLICA AUGMENTATION AQUI
    dataset = dataset.map(augment_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_dataset():
    path = dowload_dataset('datasets')
    path_train = f"{path}/train"
    image_mask_pairs = load_image_mask_pairs(path_train)

    if not image_mask_pairs:
        raise ValueError("Nenhum par (imagem, máscara) foi encontrado. "
                         "Verifique o diretório e o padrão dos arquivos.")
    
    # Usa train_test_split para separar em treino e teste
    train_pairs, test_pairs = train_test_split(
        image_mask_pairs,
        test_size=TEST_SIZE,
        random_state=42
    )

    print(f"Total de pares encontrados: {len(image_mask_pairs)}")
    print(f"Quantidade de pares em treino: {len(train_pairs)}")
    print(f"Quantidade de pares em teste: {len(test_pairs)}")


    train_dataset = build_tf_dataset(train_pairs, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = build_tf_dataset(test_pairs, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataset, test_dataset


if __name__ == '__main__':
    get_dataset()