import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import unary_union
from shapely.validation import make_valid
import numpy as np
import matplotlib.pyplot as plt

def fix_geometries(gdf):
    """
    Corrige geometrias inválidas em um GeoDataFrame.
    Utiliza 'make_valid' para garantir que todas as geometrias sejam válidas.
    """
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
    gdf = gdf[~gdf['geometry'].is_empty]  # Remove geometrias vazias
    return gdf

def crop_raster_with_shapefile(raster_path, shapefile_path, output_path):
    print("Carregando shapefile...")
    shapefile = gpd.read_file(shapefile_path)

    if shapefile.empty:
        raise ValueError("O shapefile não contém nenhuma feição válida.")

    print("Verificando e corrigindo geometrias...")
    shapefile = fix_geometries(shapefile)

    print("Convertendo CRS para EPSG:4326...")
    shapefile = shapefile.to_crs(epsg=4326)

    with rasterio.open(raster_path) as src:
        print("Raster CRS:", src.crs)

        # Certifique-se de que o shapefile esteja no mesmo CRS do raster
        print("Convertendo shapefile para o CRS do raster...")
        shapefile = shapefile.to_crs(src.crs)

        # Extrair as geometrias válidas
        shapes = [mapping(geom) for geom in shapefile.geometry if not geom.is_empty]

        if not shapes:
            raise ValueError("Nenhuma geometria válida foi encontrada no shapefile.")

        print("Aplicando recorte...")
        out_image, out_transform = mask(src, shapes, crop=True, filled=True)

        # Reescalar os valores para o intervalo [0, 255]
        out_image_rescaled = (out_image - np.min(out_image)) / (np.max(out_image) - np.min(out_image))
        out_image_rescaled = (out_image_rescaled * 255).astype(np.uint8)

        # Log dos valores no raster recortado
        print("Valores do raster recortado (após reescalado):")
        print("Min:", np.min(out_image_rescaled))
        print("Max:", np.max(out_image_rescaled))
        print("Valores únicos:", np.unique(out_image_rescaled))

        if np.all(out_image_rescaled == 0):
            raise ValueError("O recorte resultou em um raster vazio. Verifique as geometrias ou a função de recorte.")

        # Atualizar metadados do raster de saída
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "dtype": "uint8",  # Atualizar o tipo de dado para uint8
            "height": out_image_rescaled.shape[1],
            "width": out_image_rescaled.shape[2],
            "transform": out_transform,
            "nodata": 0,  # Define explicitamente nodata como 0
            "crs": "EPSG:4326"
        })

    print("Salvando o raster recortado...")
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image_rescaled)

    print(f"Recorte concluído com sucesso. Arquivo salvo em: {output_path}")

def visualize_raster(raster_path):
    with rasterio.open(raster_path) as src:
        plt.figure(figsize=(10, 10))
        show(src, title="Visualização do Raster")
        plt.show()

def main():
    # Parâmetros
    raster_path = "exports/model_30_epochs_mosaic.tif" # GeoTIFF já unificado
    shapefile_path = 'assets/fmp_shapes/FMP_poligonos_wgs84_utm23s_1.shp' # shapefile de recorte
    output_path = "exports/model_30_epochs_cropped_mask.tif"
    
    # Realizar recorte
    crop_raster_with_shapefile(raster_path, shapefile_path, output_path)
    
    # Visualizar raster resultante
    visualize_raster(output_path)

if __name__ == "__main__":
    main()
