import time
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.stats import chi2_contingency
import glob
from dask_ml.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt


def print_elapsed(label="", start_time=None, last_time=None):
    current_time = time.time()
    elapsed = current_time - last_time
    total_elapsed = current_time - start_time
    print(f"⏱️  {label} | Tempo decorrido: {elapsed:.2f}s | Total: {total_elapsed:.2f}s")
    return current_time


@dask.delayed
def load_and_align(path, all_columns):
    """Lê um CSV e garante que ele possua todas as colunas do conjunto global."""
    df = pd.read_csv(path)
    # Adiciona colunas faltantes com 0 e remove colunas extras (se houver)
    df = df.reindex(columns=all_columns, fill_value=0)
    
    # Garante que os tipos sejam exatamente iguais ao 'meta'
    df = df.astype('float32')
    if 'Alvo_Evadido' in df.columns:
        df['Alvo_Evadido'] = df['Alvo_Evadido'].astype('int32')
        
    return df


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
        memory_limit='6GB',    # Limite conservador por worker para evitar OOM no PC
        silence_logs=True
    )
    client = Client(cluster)
    
    # Otimizações de memória para Dask
    dask.config.set({
        'dataframe.shuffle.method': 'tasks',  # Shuffle eficiente para groupby
    })

    # 1. Carregando o dataset com alinhamento de colunas
    print("🔍 Analisando cabeçalhos de todos os arquivos para alinhar colunas...")
    path_glob = './processados/dask/microdados_matriculas_*_quiquadrado_*.csv'
    files = glob.glob(path_glob)
    
    if not files:
        print(f"❌ Nenhum arquivo encontrado em: {path_glob}")
        return

    # Criar a união de todas as colunas existentes em todos os arquivos
    all_cols = set()
    for f in files:
        header = pd.read_csv(f, nrows=0)
        all_cols.update(header.columns)
    
    all_cols = sorted(list(all_cols))
    print(f"✨ Total de colunas únicas identificadas: {len(all_cols)}")

    # Criar lista de objetos delayed para carregamento alinhado
    lazy_parts = [load_and_align(f, all_cols) for f in files]
    
    # Definir metadados para o Dask (importante para performance e tipos)
    meta = pd.DataFrame(columns=all_cols).astype('float32')
    coluna_alvo = 'Alvo_Evadido'
    if coluna_alvo in meta.columns:
        meta[coluna_alvo] = meta[coluna_alvo].astype('int32')

    df = dd.from_delayed(lazy_parts, meta=meta)
    
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
    
    # O Dask-ML prefere Dask Arrays com comprimentos conhecidos
    X = df[features].to_dask_array(lengths=True)
    y = df[coluna_alvo].to_dask_array(lengths=True)

    print("🚀 Treinando modelo de Regressão Logística...")
    lr = LogisticRegression(max_iter=100)
    lr.fit(X, y)
    
    # =========================================================
    # FASE 2: INTERPRETABILIDADE COM SHAP
    # =========================================================
    
    # 4. Aplicar o SHAP
    # O SHAP exige dados reais (em memória). Pegamos uma amostra significante.
    print("🧪 Coletando amostra para o SHAP...")
    X_sample = df[features].sample(frac=0.1).compute() # Pega 10% ou limite por n se for muito grande
    if len(X_sample) > 5000:
        X_sample = X_sample.sample(5000)

    # Para modelos lineares, o LinearExplainer é extremamente rápido
    explainer = shap.LinearExplainer(lr, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # 5. Visualização e Extração das Importâncias
    print("🖼️  Gerando gráfico de importância (shap_summary.png)...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)

    
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()

    # 6. Criar um ranking das colunas mais importantes
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values(by='importance', ascending=False)

    print("\nRanking das Colunas Mais Importantes:")
    print(feature_importance)

    

    # Gravar resultados em CSV
    output_csv = 'resultados_regressao_logistica.csv'
    feature_importance.to_csv(output_csv, index=False)
    print(f"\n✅ Resultados exportados para: {output_csv}")

    # Fechar o cluster Dask
    client.close()
    cluster.close()

if __name__ == '__main__':
    main()