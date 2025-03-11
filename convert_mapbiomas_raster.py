import rasterio
import numpy as np
import matplotlib.pyplot as plt

# # Definir as cores diretamente em RGB para cada valor (0-33)
# legend_rgb = {
#     0: (0, 0, 0),         # Preto (ou NoData se necessário)
#     1: (18, 153, 18),     # Floresta
#     3: (0, 100, 0),       # Formações Naturais Não Florestais
#     4: (0, 255, 0),       # Mangue
#     5: (104, 117, 55),    # Área Úmida Natural
#     9: (255, 255, 178),   # Pastagem
#     11: (255, 217, 102),  # Agricultura
#     12: (233, 116, 237),  # Agricultura Irrigada
#     15: (213, 166, 189),  # Floresta Plantada
#     19: (255, 0, 0),      # Área Urbana
#     21: (0, 0, 255),      # Corpo d'Água
#     23: (153, 153, 153),  # Não Observado
#     25: (213, 213, 229),  # Outras Formações Não Vegetadas
#     29: (221, 201, 180),  # Mineração
#     33: (246, 225, 225),  # Aquicultura
# }

# def convert_grayscale_to_rgb(input_raster_path, output_raster_path, legend_rgb):
#     """
#     Converte um raster grayscale em um raster RGB, onde cada valor do grayscale é mapeado para uma cor RGB.

#     Parâmetros:
#         input_raster_path (str): Caminho para o raster de entrada.
#         output_raster_path (str): Caminho para salvar o raster RGB de saída.
#         legend_rgb (dict): Dicionário de mapeamento de valores para cores RGB.
#     """
#     with rasterio.open(input_raster_path) as src:
#         # Ler a primeira banda (grayscale)
#         grayscale_data = src.read(1)
        
#         # Preparar metadados para o raster de saída
#         meta = src.meta
#         meta.update({
#             "count": 3,  # Três bandas (R, G, B)
#             "dtype": "uint8"  # Os valores RGB estão na faixa de 0-255
#         })

#         # Criar arrays para as 3 bandas RGB
#         red = np.zeros_like(grayscale_data, dtype=np.uint8)
#         green = np.zeros_like(grayscale_data, dtype=np.uint8)
#         blue = np.zeros_like(grayscale_data, dtype=np.uint8)

#         # Mapear valores do grayscale para RGB
#         for value, (r, g, b) in legend_rgb.items():
#             mask = grayscale_data == value
#             red[mask] = r
#             green[mask] = g
#             blue[mask] = b

#         # Salvar o raster RGB
#         with rasterio.open(output_raster_path, "w", **meta) as dst:
#             dst.write(red, 1)    # Banda R
#             dst.write(green, 2)  # Banda G
#             dst.write(blue, 3)   # Banda B

#     print(f"Raster RGB salvo em: {output_raster_path}")



# # Caminhos do raster de entrada e saída
# input_raster_path = "cropped_raster.tif"  # Substitua pelo raster enviado
# output_raster_path = "fmp_mapbiomas_rgb.tif"

# # Converter o raster
# convert_grayscale_to_rgb(input_raster_path, output_raster_path, legend_rgb)



def convert_raster_to_png(input_raster_path, output_png_path):
    """
    Converte um raster GeoTIFF para PNG sem alterar os valores.

    Parâmetros:
        input_raster_path (str): Caminho para o raster de entrada (GeoTIFF).
        output_png_path (str): Caminho para salvar a imagem PNG de saída.
    """
    with rasterio.open(input_raster_path) as src:
        # Ler a primeira banda do raster (grayscale)
        grayscale_data = src.read(1)
        
        # Plotar e salvar como PNG
        plt.figure(figsize=(10, 10))
        plt.imshow(grayscale_data, cmap="gray")
        plt.axis('off')  # Remover os eixos
        plt.savefig(output_png_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    print(f"Imagem PNG salva em: {output_png_path}")


# Caminhos do raster de entrada e do PNG de saída
input_raster_path = "fmp_mapbiomas.tif"  # Substitua pelo raster enviado
output_png_path = "fmp_mapbiomas.png"  # Caminho para salvar o PNG

# Converter o raster para PNG
convert_raster_to_png(input_raster_path, output_png_path)