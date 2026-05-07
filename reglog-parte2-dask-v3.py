import time
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import dask
from dask.distributed import Client, LocalCluster

# ==============================================================================
# ESTRATÉGIA: FEDERATED AVERAGING PARALELO
#
# Em vez de partial_fit() sequencial, cada worker do Dask:
#   1. Carrega 1 arquivo CSV (nunca mais de 1 por vez por worker)
#   2. Treina um SGDClassifier independente naquele arquivo
#   3. Retorna apenas os coeficientes (vetor pequeno)
#
# No final, os coeficientes de todos os modelos são AVERAGEADOS.
# Para modelos lineares (Regressão Logística), a média de coeficientes
# treinados em subconjuntos disjuntos converge para a solução global.
# Isso é equivalente ao "Federated Averaging" (FedAvg) de McMahan et al.
# ==============================================================================


def print_elapsed(label="", start_time=None, last_time=None):
    current_time = time.time()
    elapsed = current_time - last_time
    total_elapsed = current_time - start_time
    print(f"⏱️  {label} | Tempo decorrido: {elapsed:.2f}s | Total: {total_elapsed:.2f}s")
    return current_time


@dask.delayed
def compute_minmax_file(path, cols_to_scale):
    """Calcula min/max LOCAL de um único arquivo para as colunas de escala."""
    available = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in cols_to_scale if c in available]
    result = {c: (float('inf'), float('-inf')) for c in cols_to_scale}
    if not cols:
        return result
    chunk = pd.read_csv(path, usecols=cols)
    for c in cols:
        result[c] = (float(chunk[c].min()), float(chunk[c].max()))
    return result


@dask.delayed
def train_on_file(path, all_cols, features, coluna_alvo,
                  scale_min, scale_max, colunas_para_escalar, classes,
                  n_epochs=5):
    """
    Treina um SGDClassifier em UM único arquivo.
    Retorna os coeficientes (shape: (1, n_features)) ou None se vazio.
    """
    df = pd.read_csv(path)
    df = df.reindex(columns=all_cols, fill_value=0)
    df = df.astype('float32')
    if coluna_alvo in df.columns:
        df[coluna_alvo] = df[coluna_alvo].astype('int32')

    # Scaling Min-Max com os valores globais já computados
    for col in colunas_para_escalar:
        if col in df.columns:
            rng = scale_max[col] - scale_min[col] + 1e-12
            df[col] = (df[col] - scale_min[col]) / rng

    X = df[features].values.astype('float32')
    y = df[coluna_alvo].values.astype('int32')

    if len(X) == 0:
        return None  # arquivo vazio — ignorado no averaging

    lr = SGDClassifier(
        loss='log_loss',  # equivalente à Regressão Logística
        max_iter=n_epochs,
        tol=None,
        random_state=42,
    )
    lr.fit(X, y)
    return lr.coef_  # (1, n_features) — retorna apenas os pesos, não os dados


def main():
    start_time = time.time()
    last_time = start_time

    # ==============================================================================
    # PASSO 1: CONFIGURAÇÃO DO CLUSTER DASK
    # ==============================================================================
    cluster = LocalCluster(
        n_workers=4,           # ajuste conforme seus cores disponíveis
        threads_per_worker=2,
        memory_limit='8GB',    # por worker — nunca carrega mais de 1 CSV por vez
        silence_logs=True,
    )
    client = Client(cluster)
    print(f"🖥️  Dashboard Dask: {client.dashboard_link}")

    # ==============================================================================
    # PASSO 2: DESCOBERTA DE COLUNAS (leitura de cabeçalhos — memória mínima)
    # ==============================================================================
    print("\n🔍 Analisando cabeçalhos de todos os arquivos para alinhar colunas...")
    path_glob = './processados/dask/microdados_matriculas_*_quiquadrado_*.csv'
    files = sorted(glob.glob(path_glob))

    if not files:
        print(f"❌ Nenhum arquivo encontrado em: {path_glob}")
        client.close(); cluster.close()
        return

    all_cols = set()
    for f in files:
        all_cols.update(pd.read_csv(f, nrows=0).columns)

    all_cols = sorted(list(all_cols))
    coluna_alvo = 'Alvo_Evadido'
    features = [c for c in all_cols if c != coluna_alvo]

    print(f"✨ Total de colunas únicas: {len(all_cols)}")
    print(f"  ✓ Features: {len(features)} | Alvo: {coluna_alvo}")
    print(f"  ✓ Arquivos a processar: {len(files)}")
    last_time = print_elapsed('descoberta de colunas', start_time, last_time)

    # ==============================================================================
    # PASSO 3: MIN/MAX PARALELO (Dask distribui o cálculo entre os workers)
    # ==============================================================================
    colunas_para_escalar = [c for c in ['Renda Familiar', 'Fonte de Financiamento']
                            if c in all_cols]
    scale_min = {c: float('inf')  for c in colunas_para_escalar}
    scale_max = {c: float('-inf') for c in colunas_para_escalar}

    if colunas_para_escalar:
        print(f"\n📐 Calculando min/max global em paralelo: {colunas_para_escalar}")
        # Cria tarefas delayed para todos os arquivos e computa em paralelo
        minmax_tasks = [compute_minmax_file(f, colunas_para_escalar) for f in files]
        minmax_results = dask.compute(*minmax_tasks)  # executa em paralelo

        # Redução: agrega os min/max locais para obter os globais
        for local in minmax_results:
            for col in colunas_para_escalar:
                lo, hi = local[col]
                scale_min[col] = min(scale_min[col], lo)
                scale_max[col] = max(scale_max[col], hi)

        for col in colunas_para_escalar:
            print(f"  {col}: min={scale_min[col]:.4f}, max={scale_max[col]:.4f}")

    last_time = print_elapsed('min/max paralelo concluído', start_time, last_time)

    # ==============================================================================
    # PASSO 4: TREINO PARALELO (cada worker treina em 1 arquivo simultaneamente)
    # Federated Averaging: média dos coeficientes de N modelos independentes
    # ==============================================================================
    print(f"\n🚀 Iniciando treino paralelo em {len(files)} arquivos "
          f"({len(cluster.workers)} workers simultâneos)...")

    train_tasks = [
        train_on_file(
            f, all_cols, features, coluna_alvo,
            scale_min, scale_max, colunas_para_escalar,
            classes=np.array([0, 1]),
            n_epochs=5,
        )
        for f in files
    ]
    # dask.compute() submete todas as tarefas; o scheduler distribui pelos workers
    coef_list = dask.compute(*train_tasks)

    # Filtra arquivos vazios (None) e faz a média dos coeficientes
    valid_coefs = [c for c in coef_list if c is not None]
    if not valid_coefs:
        print("❌ Nenhum arquivo produziu um modelo válido.")
        client.close(); cluster.close()
        return

    n_skipped = len(coef_list) - len(valid_coefs)
    if n_skipped:
        print(f"  ⚠️  {n_skipped} arquivo(s) vazio(s) ignorado(s).")

    # Federated Averaging: média simples dos vetores de coeficientes
    avg_coef = np.mean(np.vstack(valid_coefs), axis=0)  # (n_features,)
    print(f"  ✓ Modelos averageados: {len(valid_coefs)}/{len(files)}")

    last_time = print_elapsed('treino paralelo concluído', start_time, last_time)

    # ==============================================================================
    # PASSO 5: IMPORTÂNCIA DAS FEATURES
    # ==============================================================================
    importance_values = np.abs(avg_coef)

    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importance_values
    }).sort_values(by='importance', ascending=False)

    print("\nRanking das Colunas Mais Importantes:")
    print(feature_importance.head(30).to_string(index=False))

    # Gráfico
    print("\n🖼️  Gerando gráfico (feature_importance.png)...")
    top_k = feature_importance.head(30)
    plt.figure(figsize=(12, 8))
    plt.barh(top_k['feature'].iloc[::-1], top_k['importance'].iloc[::-1], color='steelblue')
    plt.xlabel('Importância Média |coef| (Federated Avg)')
    plt.title(f'Top 30 Features — Regressão Logística ({len(valid_coefs)} modelos paralelos)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.close()
    print("✅ Gráfico salvo em feature_importance.png")

    # CSV
    output_csv = 'resultados_regressao_logistica_dask.csv'
    feature_importance.to_csv(output_csv, index=False)
    print(f"✅ Resultados exportados para: {output_csv}")

    print_elapsed('TOTAL', start_time, start_time)

    client.close()
    cluster.close()


if __name__ == '__main__':
    main()
