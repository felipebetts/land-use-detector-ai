import os
import ee
import geemap
import geopandas as gpd
from shapely.geometry import box

def initialize_earth_engine(token_json_path=None):
    """
    Inicializa o Earth Engine com base em credenciais locais ou via login interativo.
    - token_json_path: caminho para o JSON com credenciais, caso possua.
      Se None, solicitará login no browser.
    """
    if token_json_path and os.path.exists(token_json_path):
        credentials = ee.ServiceAccountCredentials(None, token_json_path)
        ee.Initialize(credentials)
    else:
        try:
            ee.Initialize()
        except:
            ee.Authenticate()
            ee.Initialize()

def read_shapefile(shapefile_path):
    """
    Lê o shapefile utilizando geopandas e retorna:
    - gdf: GeoDataFrame
    """
    gdf = gpd.read_file(shapefile_path)
    # Converte para WGS84 se necessário
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

def subdivide_bbox(gdf, nx=2, ny=2):
    """
    Dado um GeoDataFrame (em EPSG:4326), obtém o bounding box (minx, miny, maxx, maxy)
    e o subdivide em nx x ny colunas/linhas, retornando uma lista de polígonos do shapely.
    
    - nx, ny: número de divisões no eixo x e y.
    - Retorna: lista [polygon1, polygon2, ...] (shapely polygons).
    """
    # Pega o bounding box total do gdf
    minx, miny, maxx, maxy = gdf.total_bounds  # (minx, miny, maxx, maxy)

    dx = (maxx - minx) / nx  # Largura de cada tile
    dy = (maxy - miny) / ny  # Altura de cada tile
    
    tiles = []
    for i in range(nx):
        for j in range(ny):
            x1 = minx + i * dx
            x2 = minx + (i + 1) * dx
            y1 = miny + j * dy
            y2 = miny + (j + 1) * dy
            poly = box(x1, y1, x2, y2)  # Shapely polygon
            tiles.append(poly)
    return tiles

def filter_satellite_collection(start_date, end_date, geometry, satellite="SENTINEL_2"):
    """
    Filtra a coleção de imagens de acordo com datas, geometria e tipo de satélite.
    Usamos a coleção harmonizada do Sentinel-2 para evitar depreciações.
    """
    if satellite.upper() == "SENTINEL_2":
        # Sentinel-2 SR Harmonized
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        )
    elif satellite.upper() == "LANDSAT_8":
        collection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
        )
    elif satellite.upper() == "LANDSAT_9":
        collection = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
        )
    else:
        raise ValueError("Coleção de satélite não suportada ou não reconhecida.")
    return collection

def get_composite(image_collection):
    """
    Cria um 'composite' a partir da coleção (ex.: mediana).
    """
    composite = image_collection.median()
    return composite

def export_image_local(image, region, description, out_dir="exports", scale=10, crs="EPSG:4326"):
    """
    Exporta a imagem do Earth Engine localmente usando geemap.
    
    - image: ee.Image a ser exportada
    - region: ee.Geometry (para recorte e 'region' no download)
    - description: nome base do arquivo de saída (sem extensão)
    - out_dir: pasta local onde o arquivo será salvo
    - scale: resolução em metros (10 para Sentinel-2, 30 para Landsat, etc.)
    - crs: projeção desejada
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_file_path = os.path.join(out_dir, f"{description}.tif")
    
    try:
        region_coords = region.getInfo()['coordinates']  # Necessário para geemap
        geemap.ee_export_image(
            ee_object=image.clip(region),
            filename=out_file_path,
            scale=scale,
            region=region_coords,
            file_per_band=False,
            crs=crs
        )
        print(f"[*] Imagem exportada localmente em: {out_file_path}")
    except: 
        print("Erro ao exportar imagem")

def fetch_gee_data_with_shp(credentials='assets/lulc-piabanha-credentials.json', shapefile_path="assets/fmp_shapes/FMP_poligonos_wgs84_utm23s_1.shp", output_folder='exports'):
    # 1. Inicializar Earth Engine
    initialize_earth_engine(credentials)

    # 2. Parâmetros
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    satellite = "SENTINEL_2"
    scale = 10  # Mantemos 10m para não perder resolução
    nx, ny = 6, 6  # A quantidade de divisões na horizontal e vertical (ajuste conforme necessário)
    

    # 3. Ler shapefile
    gdf = read_shapefile(shapefile_path)
    # Observação: se o shapefile tiver inúmeras feições separadas,
    # a bounding box abaixo engloba tudo.
    
    # 4. Subdividir bounding box em NxN
    tiles = subdivide_bbox(gdf, nx=nx, ny=ny)
    
    # 5. Para cada tile, converter para ee.Geometry e fazer a exportação
    #    (caso o tile retorne vazio, a coleção pode ficar vazia, cuidado)
    for idx, tile_poly in enumerate(tiles):
        # Convertemos o shapely polygon para um ee.Geometry
        tile_coords = [[[x, y] for (x, y) in tile_poly.exterior.coords]]
        tile_geom = ee.Geometry.Polygon(coords=tile_coords)
        
        # Filtrar a coleção apenas nessa subárea
        collection = filter_satellite_collection(start_date, end_date, tile_geom, satellite)
        
        # Se a coleção estiver vazia, pode acontecer erro ou imagem preta. Podemos testar:
        # Ex: se collection.size() == 0, pular
        if collection.size().getInfo() == 0:
            print(f"[!] Nenhuma imagem encontrada para tile {idx}. Pulando.")
            continue
        
        # Criar composite (mediana)
        composite = get_composite(collection)
        
        # Descrição (nome do arquivo)
        filename = f"Composto_{satellite}_Piabanha_{start_date}_to_{end_date}_tile_{idx}"
        
        # Exportar local
        export_image_local(
            image=composite,
            region=tile_geom,
            description=filename,
            out_dir=output_folder,
            scale=scale,
            crs="EPSG:4326"
        )

if __name__ == "__main__":
    fetch_gee_data_with_shp()

