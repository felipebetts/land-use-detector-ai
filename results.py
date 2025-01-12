import os
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_charts(distribution, output_folder="charts"):
    """
    Gera gráficos relevantes a partir da distribuição das classes.
    
    Args:
        distribution (dict): Dicionário com a porcentagem de cada classe.
        output_folder (str): Pasta onde os gráficos serão salvos.
    """
    # Garantir que a pasta de saída exista
    os.makedirs(output_folder, exist_ok=True)
    
    # Classes e porcentagens
    labels = list(distribution.keys())
    percentages = list(distribution.values())

    # Gráfico de Pizza
    plt.figure(figsize=(8, 8))
    plt.pie(
        percentages, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=90, 
        wedgeprops={'edgecolor': 'black'}
    )
    plt.title("Distribuição das Classes (Gráfico de Pizza)")
    pie_chart_path = os.path.join(output_folder, "class_distribution_pie_chart.png")
    plt.savefig(pie_chart_path)
    plt.close()
    print(f"Gráfico de pizza salvo em: {pie_chart_path}")
    
    # Gráfico de Barras
    plt.figure(figsize=(10, 6))
    plt.bar(labels, percentages, color='skyblue', edgecolor='black')
    plt.xlabel("Classes")
    plt.ylabel("Porcentagem (%)")
    plt.title("Distribuição das Classes (Gráfico de Barras)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    bar_chart_path = os.path.join(output_folder, "class_distribution_bar_chart.png")
    plt.savefig(bar_chart_path)
    plt.close()
    print(f"Gráfico de barras salvo em: {bar_chart_path}")

def get_color_distribution(mask_image_path, ignore_color=(0, 0, 0)):
    # Abrir imagem e converter para numpy array
    mask_image = Image.open(mask_image_path)
    mask_array = np.array(mask_image)

    # Inicializar dicionário para contagem de pixels por cor
    color_distribution = {}

    # Obter as cores únicas presentes na máscara
    unique_colors, counts = np.unique(mask_array.reshape(-1, 3), axis=0, return_counts=True)

    for color, count in zip(unique_colors, counts):
        color_tuple = tuple(color)
        if color_tuple == ignore_color:
            continue  # Ignorar a cor especificada
        color_distribution[color_tuple] = count

    # Print do dicionário com as contagens
    print(f"Distribuição de cores na máscara: {color_distribution}")

    return color_distribution

def calculate_area_from_color_distribution(color_distribution, class_labels, class_colors_rgb, pixel_area_m2=100):
    """
    Calcula a área em hectares para cada classe com base na distribuição de cores.

    Args:
        color_distribution (dict): Distribuição de cores na máscara.
        class_labels (list): Lista de rótulos das classes.
        class_colors_rgb (list): Lista de cores RGB correspondentes às classes.
        pixel_area_m2 (int): Área representada por cada pixel em m² (default: 100 m²).

    Returns:
        dict: Dicionário com as áreas em hectares para cada classe.
    """
    class_areas = {label: 0 for label in class_labels}

    for color, count in color_distribution.items():
        if color in class_colors_rgb:
            idx = class_colors_rgb.index(color)
            class_label = class_labels[idx]
            class_areas[class_label] = (count * pixel_area_m2) / 10_000  # Convertendo para hectares

    return class_areas

def analyze_mask_distribution(mask_image_path, class_labels, class_colors_rgb, ignore_color=(0, 0, 0)):
    """
    Realiza uma análise estatística da distribuição das classes em uma imagem de máscara.
    
    Args:
        mask_image_path (str): Caminho para a imagem da máscara (PNG).
        class_labels (list): Lista com os rótulos das classes (em ordem dos índices).
        class_colors_rgb (list): Lista com as cores RGB correspondentes às classes.
        ignore_color (tuple): Cor a ser ignorada na análise (default: preto - (0, 0, 0)).
    
    Returns:
        dict: Dicionário contendo as porcentagens de cada classe.
    """
    # Abrir imagem e converter para numpy array
    mask_image = Image.open(mask_image_path)
    mask_array = np.array(mask_image)

    # Verificar se é RGB
    if mask_array.ndim != 3 or mask_array.shape[2] != 3:
        raise ValueError("A imagem de máscara deve estar no formato RGB.")

    # Converter o ignore_color para um array booleano
    ignore_mask = np.all(mask_array == ignore_color, axis=-1)
    
    # Contar os pixels ignorados e totais
    total_pixels = mask_array.shape[0] * mask_array.shape[1]
    ignored_pixels = np.sum(ignore_mask)
    relevant_pixels = total_pixels - ignored_pixels

    # Logs
    print(f"Total de pixels na imagem: {total_pixels}")
    print(f"Total de pixels não pretos (relevantes): {relevant_pixels}")
    print(f"Total de pixels pretos: {total_pixels - relevant_pixels}")

    # Criar um dicionário para armazenar os resultados
    class_distribution = {label: 0 for label in class_labels}

    # Analisar cada classe
    for idx, label in enumerate(class_labels):
        # Converter a cor da classe para formato booleano
        class_color = np.array(class_colors_rgb[idx])
        class_mask = np.all(mask_array == class_color, axis=-1)

        # Contar os pixels da classe (ignorando os pretos)
        class_count = np.sum(class_mask & ~ignore_mask)
        class_distribution[label] = (class_count / relevant_pixels) * 100

    return class_distribution

def scientific_statistical_analysis(distribution, output_folder="charts"):
    """
    Realiza uma análise estatística científica detalhada com base na distribuição das classes.
    
    Args:
        distribution (dict): Dicionário com a porcentagem de cada classe.
        output_folder (str): Pasta onde os resultados e gráficos serão salvos.
    """
    # Garantir que a pasta de saída exista
    os.makedirs(output_folder, exist_ok=True)
    
    # Converter a distribuição em um DataFrame para facilitar a análise
    data = pd.DataFrame.from_dict(distribution, orient='index', columns=['Porcentagem'])
    data.index.name = 'Classe'
    data.reset_index(inplace=True)
    
    # Estatísticas descritivas
    stats = data['Porcentagem'].describe()
    stats_path = os.path.join(output_folder, "estatisticas_descritivas.csv")
    stats.to_csv(stats_path)
    print(f"Estatísticas descritivas salvas em: {stats_path}")
    print(stats)
    
    # Gráficos avançados para análise
    # Gráfico de densidade (KDE)
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data['Porcentagem'], fill=True, color="skyblue", linewidth=2)
    plt.title("Distribuição de Densidade das Porcentagens")
    plt.xlabel("Porcentagem (%)")
    plt.ylabel("Densidade")
    kde_chart_path = os.path.join(output_folder, "class_distribution_kde_chart.png")
    plt.savefig(kde_chart_path)
    plt.close()
    print(f"Gráfico de densidade salvo em: {kde_chart_path}")
    
    # Boxplot para visualização de dispersão
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Porcentagem', data=data, color="lightgreen")
    plt.title("Boxplot da Distribuição das Classes")
    plt.xlabel("Porcentagem (%)")
    boxplot_path = os.path.join(output_folder, "class_distribution_boxplot.png")
    plt.savefig(boxplot_path)
    plt.close()
    print(f"Boxplot salvo em: {boxplot_path}")
    
    # Medidas adicionais
    iqr = stats['75%'] - stats['25%']  # Intervalo interquartil
    skewness = data['Porcentagem'].skew()  # Assimetria
    kurtosis = data['Porcentagem'].kurt()  # Curtose
    
    # Salvar medidas adicionais em um arquivo
    additional_measures = pd.DataFrame({
        "Métrica": ["IQR (Intervalo Interquartil)", "Assimetria (Skewness)", "Curtose (Kurtosis)"],
        "Valor": [iqr, skewness, kurtosis]
    })
    additional_measures_path = os.path.join(output_folder, "medidas_adicionais.csv")
    additional_measures.to_csv(additional_measures_path, index=False)
    print(f"Medidas adicionais salvas em: {additional_measures_path}")
    print(additional_measures)

def save_distribution_to_csv(distribution, output_folder):
    """
    Salva a distribuição de classes em um arquivo CSV.
    
    Args:
        distribution (dict): Distribuição de classes com rótulos e porcentagens.
        output_path (str): Caminho onde o arquivo CSV será salvo.
    """
    output_path = os.path.join(output_folder, 'distribution.csv')
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Classe", "Porcentagem"])
        for class_name, percentage in distribution.items():
            writer.writerow([class_name, percentage])
    print(f"Distribuição salva em: {output_path}")

def save_areas_to_csv(areas, output_folder):
    """
    Salva a distribuição de classes em um arquivo CSV.
    
    Args:
        distribution (dict): Distribuição de classes com rótulos e porcentagens.
        output_path (str): Caminho onde o arquivo CSV será salvo.
    """
    output_path = os.path.join(output_folder, 'areas.csv')
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Classe", "Área (ha)"])
        for class_name, percentage in areas.items():
            writer.writerow([class_name, percentage])
    print(f"Distribuição salva em: {output_path}")

def plot_distribution_table(distribution, output_path):
    """
    Plota uma tabela com a distribuição das classes e salva como imagem.
    
    Args:
        distribution (dict): Distribuição de classes com rótulos e porcentagens.
        output_path (str): Caminho para salvar o gráfico da tabela.
    """
    # Preparar dados para a tabela
    classes = list(distribution.keys())
    percentages = list(distribution.values())
    
    # Dados no formato para tabela
    table_data = [[cls, f"{perc:.2f}%"] for cls, perc in zip(classes, percentages)]
    
    # Criar figura e eixo
    fig, ax = plt.subplots(figsize=(8, len(classes) * 0.5))
    ax.axis('tight')
    ax.axis('off')

    # Adicionar tabela ao gráfico
    table = plt.table(cellText=table_data,
                      colLabels=["Classe", "Porcentagem"],
                      loc='center',
                      cellLoc='center',
                      colLoc='center')
    
    # Formatar tabela
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(table_data[0]))))
    
    # Salvar gráfico
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Tabela salva em: {output_path}")

def get_results(model_name):
    # Definir classes e cores correspondentes
    class_labels = ["Urbano", "Agricultura", "Pastagem", "Floresta", "Água", "Descampado", "Desconhecido"]
    class_colors_rgb = [
        (0, 255, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)
    ]
    
    # Caminho para a máscara
    predictions_folder = f"{model_name}_predictions"
    mask_image_path = os.path.join('exports', predictions_folder, f"{model_name}_cropped.png")
    
    # Realizar a análise de distribuição
    distribution = analyze_mask_distribution(mask_image_path, class_labels, class_colors_rgb)
    
    # Exibir distribuicao
    print(distribution)
    color_distribution = get_color_distribution(mask_image_path)
    areas = calculate_area_from_color_distribution(color_distribution, class_labels, class_colors_rgb)
    print('areas:', areas)

    # Gerar tabelas
    tables_folder = os.path.join('exports', predictions_folder, 'tables')
    os.makedirs(tables_folder, exist_ok=True)
    table_path = os.path.join(tables_folder, 'distribution.png')
    plot_distribution_table(distribution, table_path)
    save_distribution_to_csv(distribution, tables_folder)
    save_areas_to_csv(areas, tables_folder)

    
    # Gerar graficos
    charts_folder = os.path.join('exports', predictions_folder, 'charts')
    generate_charts(distribution, charts_folder)

    # Analise estatistica
    statistical_analysis_folder = os.path.join('exports', predictions_folder, 'statistical_analysis')
    scientific_statistical_analysis(distribution, statistical_analysis_folder)

def main():
    model_name = 'model_batch_12_epochs_250_v3'
    # model_name = 'model_30_epochs'
    get_results(model_name)
    

if __name__ == "__main__":
    main()
