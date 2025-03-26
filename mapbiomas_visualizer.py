import rasterio
import os
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


def main():
    # Criar diretório para salvar os arquivos
    output_dir = os.path.join('exports', 'map_biomas')
    os.makedirs(output_dir, exist_ok=True)

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

    # Caminhos de entrada
    shapefile_path = "assets/fmp_shapes/FMP_poligonos_wgs84_utm23s_1.shp"
    raster_path = "assets/brasil_coverage_2023.tif"

    # Obter a bounding box do shapefile
    bbox = get_bbox_from_shapefile(shapefile_path)

    # Caminhos de saída
    cropped_tif = os.path.join(output_dir, "mapbiomas_cropped.tif")
    colored_png = os.path.join(output_dir, "mapbiomas_colored.png")
    plot_png = os.path.join(output_dir, "mapbiomas_plot.png")

    # Realizar o corte do raster com a bounding box
    cropped_data, cropped_meta = crop_raster_with_bbox(raster_path, bbox, output_path=cropped_tif)

    # Converter TIF para PNG colorido
    with rasterio.open(cropped_tif) as src:
        data = src.read(1)
        # Normalizar dados para 0-255
        data_norm = ((data - data.min()) * (255.0 / (data.max() - data.min()))).astype(np.uint8)
        # Criar imagem RGB
        img = Image.fromarray(data_norm)
        img.save(colored_png)

    # Plotar o raster com legenda
    plt.figure(figsize=(12, 10))
    mappable = plt.imshow(cropped_data, cmap=cmap, norm=norm)
    
    # Adicionar colorbar com legenda
    cbar = plt.colorbar(
        mappable,
        ticks=classes,
        label="Classes de Uso e Cobertura do Solo",
        orientation="vertical"
    )
    
    # Adicionar descrições das classes na colorbar
    class_descriptions = {
        1: "Floresta",
        3: "Formações Naturais Não Florestais", 
        4: "Mangue",
        5: "Área Úmida Natural",
        9: "Pastagem",
        11: "Agricultura",
        12: "Agricultura Irrigada", 
        15: "Floresta Plantada",
        19: "Área Urbana",
        21: "Corpo d'Água",
        23: "Não Observado",
        25: "Outras Formações Não Vegetadas",
        29: "Mineração",
        33: "Aquicultura"
    }
    cbar.ax.set_yticklabels([f"{k} - {class_descriptions[k]}" for k in classes])

    plt.title("Mapa de Uso e Cobertura do Solo - MapBiomas", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Arquivos gerados em {output_dir}:")
    print(f"- Raster cortado: {os.path.basename(cropped_tif)}")
    print(f"- Imagem colorida: {os.path.basename(colored_png)}")
    print(f"- Plot com legenda: {os.path.basename(plot_png)}")

if __name__ == "__main__":
    main()