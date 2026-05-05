import pandas as pd
import numpy as np
import os

def comparar_csvs(file1, file2):
    print(f"--- Analisando similaridade entre {file1} e {file2} ---\n")
    
    if not os.path.exists(file1) or not os.path.exists(file2):
        print("Erro: Um ou ambos os arquivos não foram encontrados.")
        return

    # Carregar os dados
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 1. Comparação de dimensões
    print(f"Dimensões {file1}: {df1.shape}")
    print(f"Dimensões {file2}: {df2.shape}")
    
    if df1.shape != df2.shape:
        print("AVISO: Os arquivos possuem dimensões diferentes.")
    else:
        print("Sucesso: Dimensões são idênticas.")

    # 2. Comparação de colunas
    if list(df1.columns) == list(df2.columns):
        print("Sucesso: Colunas são idênticas.")
    else:
        print(f"AVISO: Colunas diferentes.\n  {file1}: {df1.columns}\n  {file2}: {df2.columns}")

    # Ordenar por feature para garantir comparação correta
    df1 = df1.sort_values('feature').reset_index(drop=True)
    df2 = df2.sort_values('feature').reset_index(drop=True)

    # 3. Comparação de features
    features1 = set(df1['feature'])
    features2 = set(df2['feature'])
    
    if features1 == features2:
        print("Sucesso: O conjunto de features é idêntico.")
    else:
        diff1 = features1 - features2
        diff2 = features2 - features1
        if diff1: print(f"Features exclusivas de {file1}: {diff1}")
        if diff2: print(f"Features exclusivas de {file2}: {diff2}")

    # 4. Análise numérica (Chi-Square e Cramer's V)
    print("\n--- Métricas de Erro (Diferenças Numéricas) ---")
    for col in ['chi2_score', 'cramers_v', 'p_value']:
        if col in df1.columns and col in df2.columns:
            diff = np.abs(df1[col] - df2[col])
            mae = diff.mean()
            mse = (diff**2).mean()
            max_diff = diff.max()
            print(f"Coluna '{col}':")
            print(f"  - Erro Médio Absoluto (MAE): {mae:.2e}")
            print(f"  - Erro Quadrático Médio (MSE): {mse:.2e}")
            print(f"  - Diferença Máxima: {max_diff:.2e}")
        else:
            print(f"Aviso: Coluna '{col}' não encontrada em um dos arquivos.")

    # 5. Comparação de Ranking (Top 30)
    print("\n--- Comparação de Ranking (Top 30) ---")
    top30_df1 = pd.read_csv(file1).head(30)['feature'].tolist()
    top30_df2 = pd.read_csv(file2).head(30)['feature'].tolist()
    
    if top30_df1 == top30_df2:
        print("Sucesso: O ranking das top 30 features é exatamente o mesmo.")
    else:
        print("AVISO: O ranking das top 30 features apresenta divergências.")
        mismatches = [(i+1, f1, f2) for i, (f1, f2) in enumerate(zip(top30_df1, top30_df2)) if f1 != f2]
        for rank, f1, f2 in mismatches:
            print(f"  Posição {rank}: {file1} -> {f1} | {file2} -> {f2}")

    # 6. Verificação de identidade exata (Pandas equals)
    if df1.equals(df2):
        print("\nCONCLUSÃO: Os arquivos são ESTRUTURALMENTE IDÊNTICOS (exatamente os mesmos valores).")
    else:
        # Tentar com tolerância
        identico_tol = True
        for col in ['chi2_score', 'cramers_v', 'p_value']:
            if not np.allclose(df1[col], df2[col], atol=1e-5):
                identico_tol = False
                break
        
        if identico_tol:
            print("\nCONCLUSÃO: Os arquivos são PRATICAMENTE IDÊNTICOS (diferenças apenas de precisão numérica irrisória).")
        else:
            print("\nCONCLUSÃO: Existem diferenças significativas entre os arquivos.")

if __name__ == "__main__":
    file_cpu = 'resultados_qui_quadrado.csv'
    file_gpu = 'resultados_qui_quadrado-cuda-dask.csv'
    comparar_csvs(file_cpu, file_gpu)
