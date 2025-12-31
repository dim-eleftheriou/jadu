import json
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def plot_umap(X_umap, y, embedding_model_name):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        hue=y,
        alpha=0.7
    )
    plt.title(f'UMAP Projection of Text Embeddings using {embedding_model_name}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Binary qaTransformedScore')
    plt.show()