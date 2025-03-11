import ee
import geopandas as gpd
import requests
from gee import initialize_earth_engine


def get_bbox_from_shapefile(shapefile_path):
    """
    Obtém a bounding box (bbox) a partir de um shapefile.
    Reprojeta o shapefile para EPSG:4326, se necessário.
    
    Parâmetros:
        shapefile_path (str): Caminho para o arquivo shapefile (.shp).
    
    Retorna:
        list: Coordenadas da bounding box no formato [lon_min, lat_min, lon_max, lat_max].
    """
    # Carregar o shapefile usando GeoPandas
    gdf = gpd.read_file(shapefile_path)
    
    # Reprojetar para EPSG:4326 (WGS 84) caso necessário
    if gdf.crs.to_string() != "EPSG:4326":
        print("Reprojetando o shapefile para EPSG:4326...")
        gdf = gdf.to_crs("EPSG:4326")
    
    # Obter a bounding box (lon_min, lat_min, lon_max, lat_max)
    bbox = gdf.total_bounds
    print(f"Bounding Box extraída: {bbox}")
    return [bbox[0], bbox[1], bbox[2], bbox[3]]

def download_lulc_map(bbox, output_file):
    """
    Consome o mapa de uso e cobertura do solo (LULC) do MapBiomas, recorta pela bounding box e salva localmente.
    
    Parâmetros:
        bbox (list): Coordenadas da bounding box no formato [lon_min, lat_min, lon_max, lat_max].
        output_file (str): Caminho completo para salvar o raster localmente (incluindo extensão .tif).
    """
    # Definir a coleção do MapBiomas (LULC - Land Use and Land Cover)
    mapbiomas_collection = ee.Image("projects/mapbiomas-workspace/public/collection8/mapbiomas_collection80_integration_v1")
    
    # Criar uma região de interesse (ROI) a partir da bounding box
    roi = ee.Geometry.BBox(*bbox)
    
    # Cortar os dados de uso e cobertura do solo pela ROI
    lulc_clipped = mapbiomas_collection.clip(roi)
    
    # Gerar URL de download
    url = lulc_clipped.getDownloadURL({
        'scale': 30,  # Resolução em metros
        'region': roi.getInfo()['coordinates'],  # Região da bounding box
        'crs': 'EPSG:4326',  # Sistema de referência
        'fileFormat': 'GeoTIFF'  # Formato do arquivo
    })

    print(f"URL de download gerado: {url}")
    
    # Fazer o download do arquivo raster
    print("Iniciando o download do mapa LULC...")
    response = requests.get(url, stream=True)

    # Salvar o arquivo localmente
    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Download concluído! Arquivo salvo em: {output_file}")

if __name__ == "__main__":
    # Inicializar o Google Earth Engine
    credentials='assets/lulc-piabanha-credentials.json'
    initialize_earth_engine(credentials)

    # Caminho para o shapefile
    shapefile_path = "assets/fmp_shapes/FMP_poligonos_wgs84_utm23s_1.shp" 

    # Caminho de saída para o arquivo raster
    output_file = "mapbiomas_lulc.tif"  # Substitua pelo nome desejado

    # Obter a bounding box do shapefile
    bbox = get_bbox_from_shapefile(shapefile_path)

    # Fazer o download do mapa LULC recortado pela bounding box
    download_lulc_map(bbox, output_file)