import os
from tensorflow.keras.models import load_model

from data_loader import get_dataset
from train import train_model
from gee import fetch_gee_data_with_shp
from mosaic import generate_tif_mosaic
from convert_image import geotiff_to_png, png_to_geotiff
from predict import predict
from crop_raster import crop_raster_with_shapefile
from results import get_results

def get_trained_model(model_name):
    # train_dataset, test_dataset = get_dataset()
    # trained_model_path = train_model(train_dataset, test_dataset, model_name, epochs=30)
    trained_model_path = os.path.join('trained_models', f"{model_name}.h5")
    trained_model = load_model(trained_model_path)
    return trained_model

def get_processed_gee_data(model_name, predictions_folder):
    gee_output_folder = os.path.join('exports', 'gee_output')
    # fetch_gee_data_with_shp(output_folder=gee_output_folder)
    mosaic_path_tif = os.path.join('exports', predictions_folder ,f"{model_name}.tif")
    generate_tif_mosaic(input_folder=gee_output_folder, output_path=mosaic_path_tif)
    mosaic_path_png = os.path.join('exports', predictions_folder ,f"{model_name}.png")
    geotiff_to_png(mosaic_path_tif, mosaic_path_png, bands=[4, 3, 2])

    return mosaic_path_png, mosaic_path_tif


def main():
    # define constants
    model_name = 'meu_modelo'
    predictions_folder = f"{model_name}_predictions"

    # cria o diretorio caso nao exista
    predictions_folder_path = os.path.join('exports', predictions_folder)
    os.makedirs(predictions_folder_path, exist_ok=True)

    # get dataset and train model
    trained_model = get_trained_model(model_name)

    # # get and process gee data
    mosaic_path_png, mosaic_path_tif = get_processed_gee_data(model_name, predictions_folder)

    # predict
    predicted_mask_png = predict(input_path=mosaic_path_png, model=trained_model)
    predicted_mask_tif_path = os.path.join('exports', predictions_folder ,f"{model_name}_mask.tif")
    png_to_geotiff(predicted_mask_png, predicted_mask_tif_path, mosaic_path_tif)
    shapefile_path = os.path.join('assets', 'fmp_shapes', 'FMP_poligonos_wgs84_utm23s_1.shp') 
    predicted_mask_cropped_tif = os.path.join('exports', predictions_folder, f"{model_name}_cropped.tif")
    crop_raster_with_shapefile(predicted_mask_tif_path, shapefile_path, predicted_mask_cropped_tif)
    predicted_mask_cropped_png = os.path.join('exports', predictions_folder, f"{model_name}_cropped.png")
    geotiff_to_png(predicted_mask_cropped_tif, predicted_mask_cropped_png)

    # analysis
    get_results(model_name)


if __name__ == '__main__':
    main()