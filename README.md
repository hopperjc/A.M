# Projeto AM 2025 — Agrupamento e Classificação no Dataset SPECTF

Este repositório contém a implementação completa da **Questão 1**, **Questão 2**, **Projeto 1** e **Projeto 2** da cadeira de Aprendizado de Máquina 2025.

---

## 🧪 Experimento (Questão 1)

### ✅ Objetivo
Aplicar o algoritmo **VKCM-K** com kernel Gaussiano para identificar agrupamentos no dataset SPECTF. A performance é avaliada com **índice de Silhouette** e **Adjusted Rand Index (ARI)**.

### 📌 Etapas

1. **Pré-processamento dos dados**
   - Normalização
   - Balanceamento com SMOTE
   - Divisão em treino (80%), validação (10%) e teste (10%)

2. **Implementação do VKCM-K**
   - Kernel Gaussiano com pesos de relevância para atributos
   - Reatribuição iterativa de rótulos com atualização de centróides e pesos

3. **Avaliação**
   - 50 execuções para cada \( K \in \{2, 3, 4, 5\} \)
   - Escolha do melhor K com maior Silhouette médio
   - Avaliação da melhor partição com ARI

4. **Visualizações**
   - Silhouette médio por K
   - Pesos de relevância das features
   - Boxplot das execuções
   - Comparação com KMeans
   - Projeção PCA dos clusters

---

## 📦 Requisitos

Instale os pacotes com:

```bash
pip install -r requirements.txt