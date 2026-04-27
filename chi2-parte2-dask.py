import time
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.stats import chi2_contingency


def print_elapsed(label="", start_time=None, last_time=None):
    current_time = time.time()
    elapsed = current_time - last_time
    total_elapsed = current_time - start_time
    print(f"⏱️  {label} | Tempo decorrido: {elapsed:.2f}s | Total: {total_elapsed:.2f}s")
    return current_time


def main():
    # Inicializar cronômetro
    start_time = time.time()
    last_time = start_time
    
    # ==============================================================================
    # PASSO 1: CONFIGURAÇÃO DO DASK PARA BIG DATA COM 4 WORKERS
    # ==============================================================================

    # Criar LocalCluster com 4 workers, cada um com 1GB de memória
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        memory_limit='2GB',
        silence_logs=True
    )
    client = Client(cluster)
    
    # Otimizações de memória para Dask
    dask.config.set({
        'dataframe.shuffle.method': 'tasks',  # Shuffle eficiente para groupby
        'array.chunk-size': '128MB',           # Tamanho padrão de chunks
    })

    # 1. Carregando o dataset com blocksize MUITO PEQUENO
    path_csv = './processados/dados_quiquadrado_reduzido.csv'

    # Blocksize pequeno = menos memória por chunk, Dask faz tree-reduction
    # Dask soma chunks em paralelo e depois agrega resultados pequenos
    df = dd.read_csv(path_csv, blocksize='5MB')

    k = 30 # Número de features que queremos selecionar
    coluna_alvo = 'Alvo_Evadido'
    features = [c for c in df.columns if c != coluna_alvo]

    # Diagnostics: mostrar que Dask está realmente particionado
    print(f"\n📊 CONFIGURAÇÃO DASK:")
    print(f"  ✓ Número de partições: {df.npartitions}")
    print(f"  ✓ Tamanho estimado total: {df.memory_usage(deep=True).sum().compute() / 1e9:.2f} GB")
    print(f"  ✓ Número de features: {len(features)}")
    print(f"  ✓ Scheduler: threads (paralelismo real)")
    print()

    # =========================================================
    # FASE 1: PROCESSAMENTO DISTRIBUÍDO COM DASK
    # O Dask faz tree-reduction: soma chunks em paralelo, depois agrega
    # Resultado final é pequeno e cabe na memória
    # =========================================================

    # O: Valores Observados (Soma de cada feature para cada classe do target)
    # Dask agrupa por coluna_alvo e soma - isso é otimizado internamente
    last_time = print_elapsed('calculando O (tree-reduction em paralelo)', start_time, last_time)
    O = df.groupby(coluna_alvo)[features].sum().compute()

    # Contagem de amostras por classe (também otimizada em Dask)
    class_counts = df[coluna_alvo].value_counts().compute()
    last_time = print_elapsed('calculando contagem de classes', start_time, last_time)

    # =========================================================
    # FASE 2: CÁLCULO MATEMÁTICO RÁPIDO NA MEMÓRIA (PANDAS/NUMPY)
    # =========================================================

    # Total de amostras
    n_amostras = class_counts.sum()

    # Probabilidade de cada classe ocorrer
    class_probs = class_counts / n_amostras

    # Soma global de cada feature (podemos obter somando a matriz O na vertical)
    feature_sums = O.sum(axis=0)
    
    # Garantir que tudo é numérico (converter para float)
    class_probs = class_probs.astype(float).values
    feature_sums = feature_sums.astype(float).values

    # E: Valores Esperados (Probabilidade da classe * Soma total da feature)
    # Usamos np.outer para cruzar rapidamente a probabilidade com a soma
    E = pd.DataFrame(
        np.outer(class_probs, feature_sums),
        index=O.index,
        columns=features
    )

    # Aplicando a fórmula do Qui-Quadrado: soma de (O - E)² / E
    chi2_stat = ((O - E) ** 2 / E).sum(axis=0)
    last_time = print_elapsed('calculando estatística do Qui-Quadrado', start_time, last_time)

    # =========================================================
    # FASE 3: SELEÇÃO E APLICAÇÃO
    # =========================================================

    # Pegando os nomes das K melhores features (maiores valores de Qui-Quadrado)
    melhores_features = chi2_stat.nlargest(k).index.tolist()

    last_time = print_elapsed(f"As {k} melhores features pelo Qui-Quadrado são:", start_time, last_time)
    print(melhores_features)

    # Criando o dataframe final preguiçoso apenas com as colunas selecionadas
    df_selecionado = df[melhores_features + [coluna_alvo]]

    # =========================================================
    # FASE 4: EXIBINDO RESULTADOS
    # =========================================================

    # Score do Qui-Quadrado (estatística total)
    chi2_score = chi2_stat.sum()

    # Graus de liberdade: (número de features - 1) * (número de classes - 1)
    n_features = len(features)
    n_classes = len(class_counts)
    df_chi2 = (n_features - 1) * (n_classes - 1)

    # P-value: probabilidade de observar uma estatística tão extrema ou mais
    p_value = chi2.sf(chi2_score, df_chi2)

    print("\n" + "="*60)
    print("📊 RESULTADOS DO TESTE QUI-QUADRADO POR FEATURE")
    print("="*60)
    
    # Imprimir score e p-value para cada feature (ordenado por score)
    chi2_sorted = chi2_stat.sort_values(ascending=False)
    for feature, score in chi2_sorted.items():
        # P-value individual: graus de liberdade = (n_classes - 1)
        df_individual = n_classes - 1
        p_value_individual = chi2.sf(score, df_individual)
        print(f"{feature:40s} | χ²: {score:12.4f} | p-value: {p_value_individual:.2e}")
    
    print("\n" + "="*60)
    print("📊 RESULTADOS AGREGADOS")
    print("="*60)
    print(f"Score Total (χ²): {chi2_score:.4f}")
    print(f"P-value Total: {p_value:.2e}")
    print(f"Graus de liberdade: {df_chi2}")
    print("="*60)
    
    # Fechar o cluster Dask
    # client.close()
    # cluster.close()

    # def cramers_v(x, y):
    #     """
    #     Calcula o V de Cramer entre duas variáveis categóricas.
    #     """
    #     contingency_table = pd.crosstab(x, y)
    #     chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    #     n = contingency_table.sum().sum()
    #     phi2 = chi2 / n
    #     r, k = contingency_table.shape
    #     # Ajuste para evitar divisão por zero se r ou k forem 1
    #     phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    #     rcorr = r - ((r-1)**2)/(n-1)
    #     kcorr = k - ((k-1)**2)/(n-1)
    #     return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))) if min((kcorr-1), (rcorr-1)) != 0 else np.nan

    # print("Calculando o V de Cramer para as features selecionadas...")

    # cramer_v_results = []
    # # Iterar apenas sobre as features que foram realmente selecionadas e são válidas (não NaN)
    # # Excluindo 'Numero de registros' e 'Matricula Atendida' que resultaram em NaN no chi2
    # valid_selected_features = [f for f in chi2_sorted if f not in ['Numero de registros', 'Matricula Atendida']]

    # for feature in valid_selected_features:
    #     # Certifique-se de que as colunas são tratadas como categóricas para pd.crosstab
    #     v_value = cramers_v(df[feature].astype('category'), df['Alvo_Evadido'].astype('category'))
    #     cramer_v_results.append({'Feature': feature, 'Cramers_V': v_value})

    # cramer_v_df = pd.DataFrame(cramer_v_results)
    # cramer_v_df = cramer_v_df.sort_values(by='Cramers_V', ascending=False)

    # print("\nResultados do V de Cramer (Maiores valores indicam maior associação):\n")
    # print(cramer_v_df)

if __name__ == '__main__':
    main()