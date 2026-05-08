from datetime import datetime
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
import scipy.sparse
from scipy.stats import chi2_contingency
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
import glob

# Inicializar cronômetro
start_time = time.time()
last_time = start_time

def print_elapsed(label=""):
    global last_time
    current_time = time.time()
    elapsed = current_time - last_time
    total_elapsed = current_time - start_time
    print(f"⏱️  {label} | Tempo decorrido: {elapsed:.2f}s | Total: {total_elapsed:.2f}s")
    last_time = current_time

# --- PREPARANDO DADOS PARA SELECTKBEST ---

path_pattern = './processados_LabelEncoding/todos_anos_quiquadrado_*.csv'
arquivos = glob.glob(path_pattern)

if not arquivos:
    raise FileNotFoundError(f"Nenhum arquivo encontrado com o padrão: {path_pattern}")

print(f"Arquivos encontrados ({len(arquivos)}): {arquivos}")

lista_dfs = []
for f in arquivos:
    temp_df = pd.read_csv(f, sep=',', encoding='utf-8', low_memory=False)
    if not temp_df.empty:
        lista_dfs.append(temp_df)
    else:
        print(f"⚠️ Aviso: O arquivo {f} está vazio e será descartado.")

if not lista_dfs:
    raise ValueError("Nenhum dos arquivos encontrados contém dados.")

df = pd.concat(lista_dfs, ignore_index=True)
print_elapsed(f"CSV carregado. Total de linhas: {len(df)}")

# path = './processados_sem_OHE/dados_quiquadrado_reduzido.csv'
# df = pd.read_csv(path)
print_elapsed("Preparando os dados para SelectKBest com Label Encoding...")

# Criar uma cópia do DataFrame para esta operação para não afetar o 'df' original
df_le_chi2 = df.copy()

# Separar a variável alvo (y)
y_le = df_le_chi2['Alvo_Evadido']

# Remover a variável alvo do df_le_chi2 para formar X_le
X_le = df_le_chi2.drop(columns=['Alvo_Evadido'])

# Identificar colunas numéricas que podem conter NaNs
# chi2 não funciona com valores negativos ou NaN. LabelEncoder já garante não-negatividade.
# Precisamos tratar NaNs em colunas originalmente numéricas.

numeric_cols_with_nans = X_le.select_dtypes(include=['int64', 'float64']).columns[X_le.select_dtypes(include=['int64', 'float64']).isnull().any()].tolist()

print_elapsed(f"Colunas numéricas com NaNs identificadas: {numeric_cols_with_nans}")

# Preencher NaNs em colunas numéricas. Usaremos 0 como um placeholder.
# Alternativamente, poderíamos usar a média/mediana, mas 0 é seguro para chi2 se os valores são counts ou similares.
for col in numeric_cols_with_nans:
    X_le[col] = X_le[col].fillna(0)

# Garantir que todas as colunas são numéricas e não possuem NaNs
# Caso haja alguma coluna object que não foi LabelEncoded (o que não deve acontecer se a célula anterior rodou bem)
for col in X_le.columns:
    if X_le[col].dtype == 'object':
        print_elapsed(f"Aviso: Coluna '{col}' ainda é do tipo 'object'. Chi2 exige tipos numéricos.")
        # Para fins de demonstração, vamos tentar LabelEncode-la aqui se necessário
        # Mas o ideal é que todas as colunas categóricas já estivessem LE.
        
        le_temp = LabelEncoder()
        X_le[col] = le_temp.fit_transform(X_le[col].astype(str))

# Adicionado: Remover valores negativos nas colunas numéricas para chi2
# O chi2 não aceita valores negativos, então substituiremos por 0 se houver.
for col in X_le.select_dtypes(include=['int64', 'float64']).columns:
    if (X_le[col] < 0).any():
        print_elapsed(f"Aviso: Coluna '{col}' contém valores negativos. Substituindo por 0 para chi2.")
        X_le[col] = X_le[col].clip(lower=0)

print_elapsed("Aplicando SelectKBest com chi2...")

k_features_le = 30  # Número de features a serem selecionadas, ajustável

# Inicializar o seletor
selector_le = SelectKBest(chi2, k=min(k_features_le, X_le.shape[1]))

# Ajustar o seletor aos dados
selector_le.fit(X_le, y_le)

print_elapsed("Fim SelectKBest com chi2...")
# Obter os scores e os p-valores
scores_le = selector_le.scores_
p_values_le = selector_le.pvalues_

# Criar um DataFrame com as features e seus scores
feature_scores_le = pd.DataFrame({
    'Feature': X_le.columns,
    'Score_chi2': scores_le,
    'P_Value': p_values_le
})

# Classificar as features pelos scores em ordem decrescente
feature_scores_le = feature_scores_le.sort_values(by='Score_chi2', ascending=False)

# Exibir as N melhores features
print_elapsed(f"\nTop {min(k_features_le, X_le.shape[1])} Features Ranqueadas por Importância (Chi-Quadrado com Label Encoding):\n")
print_elapsed(feature_scores_le.head(k_features_le))

# Opcional: Obter os nomes das features selecionadas
selected_features_le = feature_scores_le['Feature'].head(k_features_le).tolist()
print_elapsed(f"\nNomes das {min(k_features_le, X_le.shape[1])} features selecionadas: {selected_features_le[:5]}... (primeiras 5)")

print("")
print_elapsed("============Calculando o V de Cramer para as features selecionadas...==========")
print("")


def cramers_v(x, y):
    """
    Calcula o V de Cramer entre duas variáveis categóricas.
    """
    contingency_table = pd.crosstab(x, y)
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    # Ajuste para evitar divisão por zero se r ou k forem 1
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))) if min((kcorr-1), (rcorr-1)) != 0 else np.nan

cramer_v_results = []
# Iterar apenas sobre as features que foram realmente selecionadas e são válidas (não NaN)
# Excluindo 'Numero de registros' e 'Matricula Atendida' que resultaram em NaN no chi2
valid_selected_features = [f for f in selected_features_le if f not in ['Numero de registros', 'Matricula Atendida']]

for feature in valid_selected_features:
    # Certifique-se de que as colunas são tratadas como categóricas para pd.crosstab
    v_value = cramers_v(df_le_chi2[feature].astype('category'), df_le_chi2['Alvo_Evadido'].astype('category'))
    cramer_v_results.append({'Feature': feature, 'Cramers_V': v_value})

cramer_v_df = pd.DataFrame(cramer_v_results)
cramer_v_df = cramer_v_df.sort_values(by='Cramers_V', ascending=False)

print_elapsed("\nResultados do V de Cramer (Maiores valores indicam maior associação):\n")
print(cramer_v_df)

data_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f"\n{'='*70}")
print(f"🕐 Fim do processamento: {data_hora}")
total_time = time.time() - start_time
print(f"⏱️ Tempo total do processamento: {total_time:.2f} segundos")