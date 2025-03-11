import rasterio
from rasterio.windows import from_bounds
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def get_bbox_from_shapefile(shapefile_path):
    """
    Obtém a bounding box (bbox) a partir de um shapefile e reprojeta para EPSG:4326 se necessário.
    
    Parâmetros:
        shapefile_path (str): Caminho para o arquivo shapefile (.shp).
    
    Retorna:
        tuple: Bounding box no formato (lon_min, lat_min, lon_max, lat_max).
    """
    gdf = gpd.read_file(shapefile_path)

    # Reprojetar para EPSG:4326 se necessário
    if gdf.crs.to_string() != "EPSG:4326":
        print("Reprojetando o shapefile para EPSG:4326...")
        gdf = gdf.to_crs("EPSG:4326")
    
    bbox = gdf.total_bounds  # Formato: [lon_min, lat_min, lon_max, lat_max]
    print(f"BBox extraída do shapefile: {bbox}")
    return tuple(bbox)


def crop_raster_with_bbox(raster_path, bbox, output_path="cropped_raster.tif"):
    """
    Realiza o corte de um raster com base em uma bounding box e salva o resultado.

    Parâmetros:
        raster_path (str): Caminho para o arquivo raster GeoTIFF.
        bbox (tuple): Bounding box no formato (lon_min, lat_min, lon_max, lat_max).
        output_path (str): Caminho para salvar o raster cortado.
    
    Retorna:
        np.ndarray: Dados do raster cortado.
        dict: Metadados do raster cortado.
    """
    with rasterio.open(raster_path) as src:
        # Obter os limites do corte
        window = from_bounds(*bbox, transform=src.transform)

        # Ler os dados cortados
        cropped_data = src.read(1, window=window)

        # Atualizar os metadados para o raster cortado
        out_meta = src.meta.copy()
        out_meta.update({
            "height": window.height,
            "width": window.width,
            "transform": src.window_transform(window)
        })

        # Salvar o raster cortado
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(cropped_data, 1)

    print(f"Raster cortado salvo em: {output_path}")
    return cropped_data, out_meta


# Legenda com todas as classes do MapBiomas (classe: cor)
legend_colors = {
    1: "#129912",   # Floresta
    3: "#006400",   # Formações Naturais Não Florestais
    4: "#00ff00",   # Mangue
    5: "#687537",   # Área Úmida Natural
    9: "#ffffb2",   # Pastagem
    11: "#ffd966",  # Agricultura
    12: "#e974ed",  # Agricultura Irrigada
    15: "#d5a6bd",  # Floresta Plantada
    19: "#ff0000",  # Área Urbana
    21: "#0000ff",  # Corpo d'Água
    23: "#999999",  # Não Observado
    25: "#d5d5e5",  # Outras Formações Não Vegetadas
    29: "#ddc9b4",  # Mineração
    33: "#f6e1e1",  # Aquicultura
}

# Classes e cores
classes = list(legend_colors.keys())
colors = [legend_colors[cls] for cls in classes]

# Criar colormap para o raster
cmap = ListedColormap(colors)
norm = BoundaryNorm(classes + [max(classes) + 1], cmap.N)

# Caminhos de entrada e saída
shapefile_path = "assets/fmp_shapes/FMP_poligonos_wgs84_utm23s_1.shp"  # Substitua pelo caminho do shapefile
raster_path = "assets/brasil_coverage_2023.tif"  # Substitua pelo caminho do GeoTIFF
output_cropped_raster = "cropped_raster.tif"

# Obter a bounding box a partir do shapefile
bbox = get_bbox_from_shapefile(shapefile_path)

# Realizar o corte do raster com a bounding box
cropped_data, cropped_meta = crop_raster_with_bbox(raster_path, bbox, output_path=output_cropped_raster)

# Plotar o raster cortado com a legenda de cores
plt.figure(figsize=(12, 10))
mappable = plt.imshow(cropped_data, cmap=cmap, norm=norm)
plt.colorbar(
    mappable,  # Associar o mapeamento de cores
    ticks=classes,
    label="Classes de Uso e Cobertura do Solo",
    orientation="vertical"
)
plt.title("Mapa de Uso e Cobertura do Solo - MapBiomas (Cortado)", fontsize=16)
plt.axis("off")
plt.tight_layout()

# Salvar a imagem colorida com a legenda
plt.savefig("mapbiomas_colored_cropped_map.png", dpi=300)
plt.show()


# Caminho para o shapefile

# Caminho para o raster GeoTIFF

# Caminho para salvar o raster cortado