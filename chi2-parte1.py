
import gc
import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import unicodedata
import gc

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


def remover_acentos(texto):
    if not isinstance(texto, str):
        return texto
    # Normaliza para decompor caracteres acentuados (ex: 'ã' vira 'a' + '~')
    nfkd_form = unicodedata.normalize('NFKD', texto)
    # Filtra apenas os caracteres que não são marcas de acentuação e reconverte para string
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


# Configurações de exibição
pd.set_option('display.max_columns', None)
print_elapsed("Importações concluídas")

# 2. Carregar o arquivo CSV
# Devido ao tamanho (1GB), usamos o separador ';' conforme os dados de exemplo
# e 'low_memory=False' para evitar avisos de tipos mistos.
path_input = './datasets-nilo/microdados_matriculas_2023.csv'  # Altere para o caminho real do seu arquivo
df = pd.read_csv(path_input, sep=';', encoding='utf-8', low_memory=False).head(100000)
print_elapsed("CSV carregado")

# --- APLICANDO A LIMPEZA ---
# 1. Limpar nomes das colunas (Headers)
df.columns = [remover_acentos(col) for col in df.columns]
print_elapsed("Nomes de colunas limpos")

# 2. Limpar o conteúdo de todas as colunas de texto (Object)
colunas_texto = df.select_dtypes(include=['object']).columns
for col in colunas_texto:
    df[col] = df[col].apply(remover_acentos)
print_elapsed("Conteúdo de colunas de texto limpo")

# Opcional: Corrigir espaços extras que sobraram
df.columns = df.columns.str.strip()
print_elapsed("Espaços extras removidos")

print(df.columns.values)
print_elapsed("Colunas exibidas")


print_elapsed("Memória coletada")

print_elapsed("Iniciando etapa 2")
print("")
print("============================== etapa 2 ================================================================")
print("")

# --- FILTRAGEM E DEFINIÇÃO DO TARGET ---
print(f"qnt linhas antes da remoção dos alunos em curso {len(df)}")
print_elapsed("DataFrame recarregado")
# Remover espaços em branco que podem causar erros de comparação
df['Categoria da Situacao'] = df['Categoria da Situacao'].astype(str).str.strip()
print_elapsed("Espaços em branco removidos da coluna 'Categoria da Situacao'")
# df = df[df['Categoria da Situacao'] != 'Em Curso']

# Criar a coluna Alvo (Target): True (1) para "Evadidos", False (0) para os demais
# Isso prepara o campo exatamente para a Regressão Logística
df['Alvo_Evadido'] = (df['Categoria da Situacao'] == 'Evadidos').astype(int)
print_elapsed("Coluna Alvo_Evadido criada")

print(f"qnt linhas após remoção dos alunos em curso {len(df)}")
print_elapsed("Filtragem concluída")

# --- LIMPEZA ---

# 3. Remover colunas desnecessárias
colunas_para_remover = [
    'Ano', 'Vagas Regulares l1', 'Vagas Regulares l10', 'Vagas Regulares l13',
    'Vagas Regulares l14', 'Vagas Regulares l2', 'Vagas Regulares l5',
    'Vagas Regulares l6', 'Vagas Regulares l9', 'Codigo do Municipio com DV',
    'Situacao de Matricula', 'Vagas Extraordinarias l6', 'Vagas Extraordinarias l10',
    'Vagas Extraordinarias l13', 'Vagas Extraordinarias l14', 'Vagas Extraordinarias l2',
    'Vagas Extraordinarias l5', 'Vagas Extraordinarias l9', 'Vagas Extraordinarias l6',
    'Data de Fim Previsto do Ciclo', 'Data de Inicio do Ciclo', 'Data de Ocorrencia da Matricula', 'Mes De Ocorrencia da Situacao',
    'Idade', 'Fator Esforco Curso', 'Codigo da Matricula', 'Codigo do Ciclo Matricula', 'Vagas Regulares AC', 'Carga Horaria', 
    'Total de Inscritos', 'Codigo da Unidade de Ensino - SISTEC', 'Categoria da Situacao','Carga Horaria Minima','Cod Unidade', 
    'Unidade de Ensino', 'Vagas Extraordinarias AC', 'Co Inst','Vagas Extraordinarias l1', 'Regiao', 'UF', 'Numero de registros'
     
]
df = df.drop(columns=colunas_para_remover)
print_elapsed("Colunas desnecessárias removidas")

print("")
print("============================== etapa 3 ================================================================")
print("")
# 4. Tratamento de Datas
# Convertendo para datetime e extraindo informações numéricas (Regressão Logística não aceita objetos Data)
#colunas_datas = ['Data de Fim Previsto do Ciclo', 'Data de Inicio do Ciclo', 'Data de Ocorrencia da Matricula']
colunas_datas = []
for col in colunas_datas:
    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
    # Exemplo: Transformar data em "Dias desde a data mínima" ou apenas extrair o Ano/Mês
    # Aqui vamos transformar em timestamp numérico para o modelo
    df[f'{col}_timestamp'] = df[col].apply(lambda x: x.timestamp() if pd.notnull(x) else 0)
    #df = df.drop(columns=[col])

# --- TRATAMENTO DE VARIÁVEIS CATEGÓRICAS ---

# 5. Categóricos Ordinais: Renda Familiar
# Precisamos definir uma ordem lógica para a regressão entender a hierarquia
mapping_renda = {

    
   'Nao declarada': 0,
   '0<RFP<=0,5': 1,
   '0,5<RFP<=1,0': 2,
   '1,0<RFP<=1,5': 3,
   '1,5<RFP<=2,5': 4,
   '2,5<RFP<=3,5': 5,
   'RFP>3,5': 6
}
df['Renda Familiar'] = df['Renda Familiar'].map(mapping_renda).fillna(-1)

# 6. Categóricos Nominais (One-Hot Encoding) usando OneHotEncoder do scikit-learn para economizar memória
# Aplicando One-Hot Encoding para as colunas nominais especificadas.
# Usamos sparse_output=True para lidar com grande número de colunas e economizar memória.


colunas_onehotencoding = []     

colunas_label_encoding = [
    'Nome de Curso',
    'Instituicao',
    'Cor / Raca',
     'Turno',
    'Sexo', 'Subeixo Tecnologico', 'Tipo de Curso', 'Tipo de Oferta',
    'Eixo Tecnologico', 'Modalidade de Ensino', 
    'Matricula Atendida', 'Fonte de Financiamento', 'Municipio', 'Faixa Etaria',
    
]

le = LabelEncoder()
turno_mapping = {}  # Dicionário para armazenar o mapeamento de 'Turno'

for col in colunas_label_encoding:
    le_col = LabelEncoder()  # Um encoder específico para cada coluna
    df[col] = le_col.fit_transform(df[col].astype(str))
    
    # Se a coluna for 'Turno', salva o mapeamento
    if col == 'Turno':
        turno_mapping = dict(zip(le_col.classes_, le_col.transform(le_col.classes_)))

# Inicializa o OneHotEncoder
# handle_unknown='ignore' é útil para evitar erros se novas categorias aparecerem em dados futuros
# sparse_output=True garante que a saída seja uma matriz esparsa, otimizando o uso de memória
#encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

# Ajusta e transforma as colunas nominais selecionadas
#encoded_data = encoder.fit_transform(df[colunas_nominais_onehotencoding])

# Obtém os nomes das novas colunas geradas
#feature_names = encoder.get_feature_names_out(colunas_nominais_onehotencoding)

# Cria um DataFrame esparso a partir dos dados codificados
#encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_data, columns=feature_names)

# Remove as colunas nominais originais do DataFrame principal
#df = df.drop(columns=colunas_nominais_onehotencoding)

# Concatena o DataFrame principal com as novas colunas codificadas esparsas
#df = pd.concat([df, encoded_df], axis=1)

#df = pd.get_dummies(df, columns=colunas_onehotencoding, drop_first=True, dtype=int)
print(f"Quantidade de colunas do DataFrame: {df.shape[1]}")

path_output = './processados_sem_OHE/dados_quiquadrado_reduzido.csv'
df.to_csv(path_output, index=False)

print(f"Sucesso! {len(df)} linhas processadas.")
print(f"Arquivo salvo no Drive: {path_output}")


print("")
print("============================== FIM ================================================================")
print("")

print("\n📊 MAPEAMENTO DE LABEL ENCODING - CAMPO 'TURNO':")
print("=" * 60)
for valor_original, valor_codificado in sorted(turno_mapping.items(), key=lambda x: x[1]):
    print(f"  '{valor_original}' → {valor_codificado}")
print("=" * 60)

del df
gc.collect()

