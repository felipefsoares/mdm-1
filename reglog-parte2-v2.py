import time
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
# SGDClassifier(loss='log_loss') é matematicamente equivalente à Regressão Logística
# e suporta partial_fit() para aprendizado incremental (out-of-core):
# nunca carrega mais de um arquivo por vez, eliminando o MemoryError.


def print_elapsed(label="", start_time=None, last_time=None):
    current_time = time.time()
    elapsed = current_time - last_time
    total_elapsed = current_time - start_time
    print(f"⏱️  {label} | Tempo decorrido: {elapsed:.2f}s | Total: {total_elapsed:.2f}s")
    return current_time


def load_and_align(path, all_columns):
    """Lê um CSV e garante que ele possua todas as colunas do conjunto global."""
    df = pd.read_csv(path)
    # Adiciona colunas faltantes com 0 e remove colunas extras (se houver)
    df = df.reindex(columns=all_columns, fill_value=0)
    df = df.astype('float32')
    if 'Alvo_Evadido' in df.columns:
        df['Alvo_Evadido'] = df['Alvo_Evadido'].astype('int32')
    return df


def main():
    # Inicializar cronômetro
    start_time = time.time()
    last_time = start_time

    # ==============================================================================
    # PASSO 1: DESCOBERTA DE COLUNAS
    # Leitura apenas dos cabeçalhos (nrows=0) — memória mínima
    # ==============================================================================
    print("🔍 Analisando cabeçalhos de todos os arquivos para alinhar colunas...")
    path_glob = './processados/dask/microdados_matriculas_*_quiquadrado_*.csv'
    files = sorted(glob.glob(path_glob))

    if not files:
        print(f"❌ Nenhum arquivo encontrado em: {path_glob}")
        return

    all_cols = set()
    for f in files:
        header = pd.read_csv(f, nrows=0)
        all_cols.update(header.columns)

    all_cols = sorted(list(all_cols))
    coluna_alvo = 'Alvo_Evadido'
    features = [c for c in all_cols if c != coluna_alvo]
    print(f"✨ Total de colunas únicas identificadas: {len(all_cols)}")
    print(f"  ✓ Features: {len(features)} | Alvo: {coluna_alvo}")
    print(f"  ✓ Arquivos a processar: {len(files)}")

    # ==============================================================================
    # PASSO 2: PASSAGEM 1 — descobrir min/max global das colunas a escalar
    # Lê apenas as colunas necessárias, arquivo por arquivo (sem jamais acumular tudo)
    # ==============================================================================
    colunas_para_escalar = [c for c in ['Renda Familiar', 'Fonte de Financiamento']
                            if c in all_cols]
    scale_min = {c: float('inf')  for c in colunas_para_escalar}
    scale_max = {c: float('-inf') for c in colunas_para_escalar}

    if colunas_para_escalar:
        print(f"\n📐 Passagem 1/2 — calculando min/max global de: {colunas_para_escalar}")
        for i, f in enumerate(files, 1):
            chunk = pd.read_csv(f, usecols=[c for c in colunas_para_escalar
                                            if c in pd.read_csv(f, nrows=0).columns])
            for col in colunas_para_escalar:
                if col in chunk.columns:
                    scale_min[col] = min(scale_min[col], float(chunk[col].min()))
                    scale_max[col] = max(scale_max[col], float(chunk[col].max()))
            print(f"  [{i}/{len(files)}] {f.split('/')[-1]}", end='\r')
        print()

    last_time = print_elapsed('passagem 1 concluída', start_time, last_time)

    # ==============================================================================
    # PASSO 3: PASSAGEM 2 — treino incremental com SGDClassifier (partial_fit)
    #
    # SGDClassifier(loss='log_loss') = Regressão Logística treinada por SGD.
    # partial_fit() atualiza os pesos com cada arquivo sem acumular dados em memória.
    # Nunca carrega mais de 1 arquivo (~160 MB) por vez → sem MemoryError.
    # ==============================================================================
    print("\n🚀 Passagem 2/2 — treino incremental (out-of-core)...")
    classes = np.array([0, 1])  # classes conhecidas a priori
    lr = SGDClassifier(
        loss='log_loss',   # equivalente à Regressão Logística
        max_iter=1,        # 1 época por arquivo; partial_fit acumula épocas
        tol=None,
        random_state=42,
        n_jobs=-1,
    )

    for i, f in enumerate(files, 1):
        print(f"  📂 [{i}/{len(files)}] {f.split('/')[-1]}")

        df_part = load_and_align(f, all_cols)

        # Aplicar scaling Min-Max nas colunas selecionadas
        for col in colunas_para_escalar:
            if col in df_part.columns:
                rng = scale_max[col] - scale_min[col] + 1e-12
                df_part[col] = (df_part[col] - scale_min[col]) / rng

        X_part = df_part[features].values.astype('float32')
        y_part = df_part[coluna_alvo].values.astype('int32')

        if len(X_part) == 0:
            print(f"  ⚠️  Arquivo vazio (0 linhas), ignorando.")
            continue

        lr.partial_fit(X_part, y_part, classes=classes)

    last_time = print_elapsed('treino concluído', start_time, last_time)

    # ==============================================================================
    # PASSO 4: IMPORTÂNCIA DAS FEATURES VIA COEFICIENTES
    # Para Regressão Logística, |coef_| é matematicamente equivalente ao SHAP
    # LinearExplainer para modelos lineares — sem dependências extras.
    # ==============================================================================

    # SGDClassifier retorna coef_ como numpy array diretamente
    coef = lr.coef_  # shape (1, n_features) para binário
    importance_values = np.abs(coef).mean(axis=0)

    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importance_values
    }).sort_values(by='importance', ascending=False)

    print("\nRanking das Colunas Mais Importantes:")
    print(feature_importance.head(30).to_string(index=False))

    # Visualização
    print("\n🖼️  Gerando gráfico de importância (feature_importance.png)...")
    top_k = feature_importance.head(30)
    plt.figure(figsize=(12, 8))
    plt.barh(top_k['feature'].iloc[::-1], top_k['importance'].iloc[::-1], color='steelblue')
    plt.xlabel('Importância Média |coef|')
    plt.title('Top 30 Features — Regressão Logística (SGD incremental)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.close()
    print("✅ Gráfico salvo em feature_importance.png")

    # Exportar CSV
    output_csv = 'resultados_regressao_logistica.csv'
    feature_importance.to_csv(output_csv, index=False)
    print(f"✅ Resultados exportados para: {output_csv}")
    print_elapsed('total', start_time, start_time)

if __name__ == '__main__':
    main()