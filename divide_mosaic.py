from PIL import Image
import os

from constants import TILE_SIZE


def divide_image_with_padding(image_path, output_folder, tile_size=TILE_SIZE):
    # Carregar a imagem
    img = Image.open(image_path)
    width, height = img.size
    
    # Calcular o número de tiles necessários em cada dimensão
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size

    # Criar pasta de saída se não existir
    os.makedirs(output_folder, exist_ok=True)

    tiles_paths = []

    # Iterar sobre cada tile
    for i in range(tiles_y):
        for j in range(tiles_x):
            # Determinar as coordenadas do tile atual
            left = j * tile_size
            upper = i * tile_size
            right = min(left + tile_size, width)
            lower = min(upper + tile_size, height)
            
            # Extrair o tile
            tile = img.crop((left, upper, right, lower))
            
            # Adicionar preenchimento, se necessário
            padded_tile = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
            padded_tile.paste(tile, (0, 0))

            # Salvar o tile
            tile_name = f"tile_{i}_{j}.png"
            tile_path = os.path.join(output_folder, tile_name)
            padded_tile.save(tile_path)
            tiles_paths.append(tile_path)
            print(f"Tile salvo: {tile_path}")

    return tiles_paths


def recompose_image(tiles_folder, output_path, original_size, tile_size=TILE_SIZE):
    """
    Recompoe os tiles em uma única imagem com o tamanho original.

    Args:
        tiles_folder (str): Pasta onde os tiles estão armazenados.
        output_path (str): Caminho para salvar a imagem recomposta.
        original_size (tuple): Tamanho original da imagem (largura, altura).
        tile_size (int): Tamanho dos tiles (padrão: TILE_SIZExTILE_SIZE).
    """
    original_width, original_height = original_size
    tiles_x = (original_width + tile_size - 1) // tile_size
    tiles_y = (original_height + tile_size - 1) // tile_size

    # Criar imagem vazia com o tamanho original
    recomposed_image = Image.new("RGB", (original_width, original_height))

    # Carregar os tiles e recompor a imagem
    for i in range(tiles_y):
        for j in range(tiles_x):
            tile_name = f"tile_{i}_{j}.png"
            tile_path = os.path.join(tiles_folder, tile_name)

            if os.path.exists(tile_path):
                tile = Image.open(tile_path)

                # Determinar as coordenadas de inserção
                left = j * tile_size
                upper = i * tile_size

                # Inserir o tile na imagem recomposta
                recomposed_image.paste(tile, (left, upper))

                print(f"Carregando {tile_name} com dimensões {tile.size}")
            else:
                print(f"Tile não encontrado: {tile_name}")

    # Salvar a imagem recomposta
    recomposed_image.save(output_path)
    print(f"Imagem recomposta salva em: {output_path}") 
    return output_path     


def main():
    # Exemplo de uso
    divide_image_with_padding(
        image_path="exports/mosaico_final.png", 
        output_folder="mosaico_tiles", 
        tile_size=TILE_SIZE
    )

if __name__ == '__main__':
    main()