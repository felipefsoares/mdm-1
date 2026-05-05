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
    # PASSO 1: CONFIGURAÇÃO DO DASK PARA BIG DATA COM TODOS OS 12 CORES
    # ==============================================================================

    # Configuração de cluster de CPU (usando todos os cores disponíveis)
    cluster = LocalCluster(
        n_workers=2,          # Ajustado para os 12 cores informados anteriormente
        threads_per_worker=1,
        memory_limit='3GB',    # Limite conservador por worker para evitar OOM no PC
        silence_logs=True
    )
    client = Client(cluster)
    
    # Otimizações de memória para Dask
    dask.config.set({
        'dataframe.shuffle.method': 'tasks',  # Shuffle eficiente para groupby
    })

    # 1. Carregando o dataset
    path_pattern = './processados/dask/microdados_matriculas_2019_quiquadrado_*.csv'
    
    # No Dask DataFrame (CPU), podemos usar blocksize para ler em paralelo se os arquivos forem grandes
    df = dd.read_csv(path_pattern, sep=',')
    
    # O persist() pode ser usado se houver RAM suficiente, mas deixamos 
    # lazy para economia de memória inicial.

    k = 30 # Número de features que queremos selecionar
    coluna_alvo = 'Alvo_Evadido'
    
    # Aplicar coerção para float32 para economizar memória e acelerar somas.
    # Como TUDO é numérico, mandamos um cast unificado para tudo e sobrepomos o Alvo.
    df = df.astype('float32')
    df[coluna_alvo] = df[coluna_alvo].astype('int32')
    
    # Scaling (Min-Max) das colunas Renda Familiar e Fonte de Financiamento
    colunas_para_escalar = ['Renda Familiar', 'Fonte de Financiamento']
    for col in colunas_para_escalar:
        if col in df.columns:
            c_min = df[col].min()
            c_max = df[col].max()
            df[col] = (df[col] - c_min) / (c_max - c_min + 1e-12)

    # O array de features numéricas garantidas processadas
    features = [str(c) for c in df.columns if str(c) != coluna_alvo]

    # Diagnostics: mostrar que Dask está realmente particionado
    print(f"\n📊 CONFIGURAÇÃO DASK:")
    print(f"  ✓ Número de partições: {df.npartitions}")
    #print(f"  ✓ Tamanho estimado total: {df.memory_usage(deep=True).sum().compute() / 1e9:.2f} GB")
    print(f"  ✓ Número de features: {len(features)}")
    print(f"  ✓ Scheduler: processes (12 cores da CPU)")
    print()

    # =========================================================
    # FASE 1: PROCESSAMENTO DISTRIBUÍDO COM DASK
    # O Dask faz tree-reduction: soma chunks em paralelo, depois agrega
    # Criamos os ponteiros e calculamos tudo em ALTA PERFORMANCE (única ida ao disco)
    # =========================================================

    last_time = print_elapsed('iniciando cálculos lazy', start_time, last_time)
    
    # Criar promessas de cálculo (lazily evaluated)
    O_lazy = df.groupby(coluna_alvo)[features].sum()
    class_counts_lazy = df[coluna_alvo].value_counts()
    
    # Avaliar as duas simultaneamente
    O, class_counts = dask.compute(O_lazy, class_counts_lazy)
    
    # Garantir mesma ordem de classes antes da extração com .values
    O = O.sort_index()
    class_counts = class_counts.sort_index()
    
    # Já estão em formato Pandas/Series após o compute no Dask padrão
    
    last_time = print_elapsed('cálculos (O e class_counts) concluídos em um passe único do Dask', start_time, last_time)

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

    # Aplicando a fórmula do Qui-Quadrado COMPLETA (incluindo o caso onde a feature é 0)
    # Isso é necessário para o cálculo correto do V de Cramer em tabelas 2xN
    O_neg = class_counts.values[:, np.newaxis] - O
    E_neg = class_counts.values[:, np.newaxis] - E
    
    # Evitar divisão por zero (ocorre se uma feature é constante)
    epsilon = 1e-12
    chi2_pos = ((O - E) ** 2 / (E + epsilon)).sum(axis=0)
    chi2_neg = ((O_neg - E_neg) ** 2 / (E_neg + epsilon)).sum(axis=0)
    chi2_stat = chi2_pos + chi2_neg
    
    # Cálculo do V de Cramer: sqrt(chi2 / (n * min(k-1, r-1)))
    # Como as features são tratadas como binárias aqui (1 ou 0), k=2, então min(1, r-1) = 1
    cramers_v = np.sqrt(chi2_stat / n_amostras)
    
    last_time = print_elapsed('calculando estatísticas do Qui-Quadrado e V de Cramer', start_time, last_time)

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
    # Preparar resultados para exibição e exportação
    results_list = []
    chi2_sorted = chi2_stat.sort_values(ascending=False)
    
    # Imprimir score, p-value e V de Cramer para cada feature (ordenado por score)
    for feature in chi2_sorted.index:
        score = float(chi2_stat[feature])
        v_score = float(cramers_v[feature])
        # P-value individual: graus de liberdade = (n_classes - 1)
        df_individual = n_classes - 1
        p_value_individual = float(chi2.sf(score, df_individual))
        
        print(f"{feature:40s} | χ²: {score:12.4f} | V: {v_score:.4f} | p-value: {p_value_individual:.2e}")
        
        # Adicionar à lista para o CSV
        results_list.append({
            'feature': feature,
            'chi2_score': score,
            'cramers_v': v_score,
            'p_value': p_value_individual
        })

    # Gravar resultados em CSV
    df_results = pd.DataFrame(results_list)
    output_csv = 'resultados_qui_quadrado.csv'
    df_results.to_csv(output_csv, index=False)
    print(f"\n✅ Resultados exportados para: {output_csv}")
    print("\n" + "="*60)
    print("📊 RESULTADOS AGREGADOS")
    print("="*60)
    print(f"Score Total (χ²): {chi2_score:.4f}")
    print(f"P-value Total: {p_value:.2e}")
    print(f"Graus de liberdade: {df_chi2}")
    print("="*60)
    
    # Fechar o cluster Dask
    client.close()
    cluster.close()

if __name__ == '__main__':
    main()