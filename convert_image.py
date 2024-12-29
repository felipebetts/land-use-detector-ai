import rasterio
import numpy as np
from PIL import Image

def geotiff_to_png(input_tif, output_png, bands=[1, 2, 3], stretch=True):
    """
    Converte um GeoTIFF para PNG.

    - input_tif: caminho para o arquivo GeoTIFF de entrada
    - output_png: caminho para salvar o arquivo PNG
    - bands: lista de bandas para usar (default: [1, 2, 3] para RGB)
    - stretch: se True, aplica um stretch linear para melhorar a visualização
    """
    with rasterio.open(input_tif) as src:
        # Ler as bandas especificadas
        img = src.read(bands)
        
        # Transpor para (altura, largura, bandas)
        img = np.transpose(img, (1, 2, 0))
        
        if stretch:
            # Aplicar stretch linear (percentil 2 a 98) para cada banda
            img_stretched = np.empty_like(img, dtype=np.float32)
            for i in range(img.shape[2]):
                band = img[:, :, i]
                p2 = np.percentile(band, 2)
                p98 = np.percentile(band, 98)
                # Evitar divisão por zero
                if p98 - p2 == 0:
                    p98 = p2 + 1
                band_stretched = np.clip((band - p2) * 255.0 / (p98 - p2), 0, 255)
                img_stretched[:, :, i] = band_stretched
            img = img_stretched
        else:
            # Normalizar para 0-255 com base no mínimo e máximo
            img_min = img.min(axis=(0,1), keepdims=True)
            img_max = img.max(axis=(0,1), keepdims=True)
            img = (img - img_min) / (img_max - img_min) * 255.0
            img = np.clip(img, 0, 255)
        
        # Converter para uint8
        img = img.astype(np.uint8)
        
        # Salvar com PIL
        image = Image.fromarray(img)
        image.save(output_png)
        print(f"PNG salvo em: {output_png}")

def png_to_geotiff(png_path, output_path, original_tif_path, bands=[1, 2, 3]):


    with rasterio.open(original_tif_path) as src:
        profile = src.profile
        # Garantir que os índices das bandas sejam válidos
        if max(bands) > src.count or min(bands) < 1:
            raise ValueError(f"As bandas especificadas {bands} estão fora do intervalo disponível no TIFF original.")
    
        # Ler as bandas especificadas para georreferenciamento (não são usadas diretamente no RGB)
        geotransform = src.transform
        crs = src.crs

    # Atualizar o perfil para RGB (3 bandas)
    profile.update(
        dtype=rasterio.uint8,  # Tipo uint8 para RGB
        count=3,               # Três bandas (R, G, B)
        driver='GTiff',        # Formato TIFF
        photometric="RGB"      # Indicador de cores RGB
    )

    mask_png = Image.open(png_path)
    mask_array = np.array(mask_png)

    # Verificar se a imagem PNG é RGB
    if mask_array.ndim != 3 or mask_array.shape[2] != 3:
        raise ValueError("A imagem PNG deve ser RGB com 3 bandas.")
    
    # Mapear os canais do PNG para as bandas especificadas
    reordered_array = np.transpose(mask_array, (2, 0, 1))
    # for idx, band in enumerate(bands):
    #     reordered_array[idx] = mask_array[idx]

     # Salvar o novo TIFF com as informações geoespaciais e as bandas RGB
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(reordered_array)  # Escrever todas as bandas (R, G, B)
        dst.transform = geotransform  # Adicionar transformações do TIFF original
        dst.crs = crs                 # Adicionar CRS do TIFF original
        

if __name__ == "__main__":
    # Configuração de entrada e saída
    input_tif = "exports/FMP_v3.tif"  # Caminho para o mosaico GeoTIFF
    output_png = "exports/FMP_v3.png"  # Caminho para salvar o PNG

    # Converter o mosaico para PNG
    # geotiff_to_png(input_tif, output_png, bands=[4, 3, 2], stretch=True)
    png_to_geotiff(png_path='mosaico_final-predictions-2/output.png', output_path='exports/mosaico_final_mask.tif', original_tif_path="exports/mosaico_final.tif", bands=[4, 3, 2])
