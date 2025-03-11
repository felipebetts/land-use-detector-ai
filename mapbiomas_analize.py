import rasterio
import numpy as np
from collections import Counter

def count_pixels_by_value(raster_path, value_range=(0, 33)):
    """
    Conta a quantidade de pixels para cada valor dentro de uma faixa no raster.

    Parâmetros:
        raster_path (str): Caminho para o arquivo raster.
        value_range (tuple): Faixa de valores para considerar (valor mínimo, valor máximo).

    Retorna:
        dict: Um dicionário com os valores e suas respectivas quantidades de pixels.
    """
    with rasterio.open(raster_path) as src:
        # Ler os dados do raster (primeira banda)
        raster_data = src.read(1)

        # Converter para um array unidimensional (ignorar pixels nodata)
        raster_data = raster_data[np.isfinite(raster_data)]

        # Filtrar valores dentro da faixa especificada
        raster_data = raster_data[(raster_data >= value_range[0]) & (raster_data <= value_range[1])]

        # Contar a quantidade de pixels para cada valor
        pixel_counts = Counter(raster_data)

    return dict(pixel_counts)

# Caminho para o raster grayscale
raster_path = "fmp_mapbiomas.tif"  # Substitua pelo caminho real do arquivo

# Faixa de valores (0 a 33)
value_range = (0, 33)

# Contar os pixels por valor
pixel_counts = count_pixels_by_value(raster_path, value_range=value_range)

# Exibir os resultados
print("Contagem de Pixels por Valor:")
for value, count in sorted(pixel_counts.items()):
    print(f"Valor {int(value)}: {count} pixels")
