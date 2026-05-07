import os
os.environ["USE_TF"] = "0"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)

def main():
    df = pd.read_csv('data/climate_articles.csv')
    
    en_texts = df[df['language'] == 'en'].head(10)
    ar_texts = df[df['language'] == 'ar'].head(10)
    selected = pd.concat([en_texts, ar_texts])
    
    model_name = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    encoded_input = tokenizer(
        selected['text'].tolist(),
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    similarity_matrix = embeddings @ embeddings.T
    similarity_matrix = similarity_matrix.numpy()
    
    labels = [f"{lang}-{id}: {text[:40]}..." for lang, id, text in zip(selected['language'], selected['id'], selected['text'])]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap='viridis'
    )
    plt.title('Cross-Lingual Embedding Similarity Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('cross_lingual_similarity_heatmap.png')
    print("Heatmap saved as cross_lingual_similarity_heatmap.png")
    
    print("\nSimilarity Matrix:")
    print(similarity_matrix)

if __name__ == "__main__":
    main()
