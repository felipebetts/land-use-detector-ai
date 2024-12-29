import glob
import os
import rasterio
from rasterio.merge import merge
from rasterio.plot import show

def generate_tif_mosaic(input_folder, output_path):
    """
    Faz o mosaico de todos os arquivos .tif de 'input_folder'
    e salva como um único arquivo em 'output_path'.
    """
    # Lista todos os .tif do diretório
    search_pattern = os.path.join(input_folder, "*.tif")
    tif_files = glob.glob(search_pattern)
    
    # Lista de datasets rasterio
    src_files_to_mosaic = []
    for fp in tif_files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    # Unifica tudo
    mosaic, out_trans = merge(src_files_to_mosaic)

    # Copiamos o perfil do primeiro arquivo para manter metadados (ex.: CRS, dtype, etc.)
    out_meta = src_files_to_mosaic[0].meta.copy()
    # Ajusta a transform para a do mosaico
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    # Salva em disco
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"Mosaico gerado em: {output_path}")


if __name__ == "__main__":
    # Ajuste para sua pasta e nome de saída
    input_folder = "exports"  # onde estão os tiles
    output_file = "exports/mosaico_final.tif"

    generate_tif_mosaic(input_folder, output_file)
