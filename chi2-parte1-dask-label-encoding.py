
import gc
import time
from datetime import datetime
import argparse
import sys
import os

import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import unicodedata
import gc

# Configurações de processamento em chunks
CHUNK_SIZE_MB = 128  # Tamanho dos chunks em MB
NUM_WORKERS = 10
THREADS_PER_WORKER = 1
MEMORY_LIMIT_PER_WORKER = '3GB'

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
    """Remove acentos de um texto"""
    if not isinstance(texto, str):
        return texto
    # Normaliza para decompor caracteres acentuados (ex: 'ã' vira 'a' + '~')
    nfkd_form = unicodedata.normalize('NFKD', texto)
    # Filtra apenas os caracteres que não são marcas de acentuação e reconverte para string
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def remover_acentos_partition(partition):
    """Aplica remover_acentos a uma partição inteira do DataFrame"""
    # Limpar nomes das colunas
    partition.columns = [remover_acentos(col) for col in partition.columns]
    
    # Limpar o conteúdo de todas as colunas de texto (Object)
    colunas_texto = partition.select_dtypes(include=['object']).columns
    for col in colunas_texto:
        partition[col] = partition[col].apply(remover_acentos)
    
    # Remover espaços extras
    partition.columns = partition.columns.str.strip()
    
    return partition


def criar_meta_normalizado(df_sample):
    """
    Cria um metadado com nomes de colunas normalizados (sem acentos)
    a partir de uma amostra do DataFrame
    """
    # Normalizar nomes das colunas
    df_sample_norm = df_sample.copy()
    df_sample_norm.columns = [remover_acentos(col).strip() for col in df_sample_norm.columns]
    
    # Retornar uma cópia vazia com a mesma estrutura e tipos
    return df_sample_norm.head(0)


def aplicar_label_encoding_partition(partition, colunas_label_encoding, encoders_dict):
    """Aplica label encoding a uma partição usando encoders pré-treinados"""
    from sklearn.preprocessing import LabelEncoder
    
    for col in colunas_label_encoding:
        if col in partition.columns and col in encoders_dict:
            le = encoders_dict[col]
            # Usar transform com classes conhecidas, valores desconhecidos vão para -1
            partition[col] = partition[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    gc.collect()
    return partition

def main():
    cluster = LocalCluster(
        n_workers=NUM_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
        memory_limit=MEMORY_LIMIT_PER_WORKER,
        silence_logs=True
    )
    client = Client(cluster)
    
    try:
        # Configurações de exibição
        pd.set_option('display.max_columns', None)
        print_elapsed("Importações concluídas")
        
        # Imprimir data e hora
        data_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"\n{'='*70}")
        print(f"🕐 Início do processamento: {data_hora}")
        print(f"{'='*70}\n")

        # 2. Carregar TODOS os arquivos CSV da pasta
        path_input = './datasets-nilo/microdados_matriculas_*.csv'
        
        # Para a amostra (Pandas), pegamos o primeiro arquivo encontrado
        import glob
        lista_arquivos = glob.glob(path_input)
        if not lista_arquivos:
            raise FileNotFoundError(f"Nenhum arquivo encontrado em: {path_input}")
            
        print(f"Arquivos encontrados ({len(lista_arquivos)}): {lista_arquivos}")
        
        # Step 1: Ler amostra para inspeção de metadados
        print(f"Lendo amostra do primeiro arquivo para inspeção: {os.path.basename(lista_arquivos[0])}")
        df_sample = pd.read_csv(
            lista_arquivos[0], 
            sep=';', 
            encoding='utf-8', 
            nrows=1000  # Aumentar para melhor inferência
        )
        
        # Step 2: Normalizar nomes das colunas na amostra
        df_sample.columns = [remover_acentos(col).strip() for col in df_sample.columns]
        
        # Step 3: Limpar conteúdo de texto da amostra
        colunas_texto = df_sample.select_dtypes(include=['object']).columns
        for col in colunas_texto:
            df_sample[col] = df_sample[col].apply(remover_acentos)
        
        print_elapsed("Amostra preparada para inspeção")
        
        # Step 4: Criar dtype_dict baseado na amostra
        dtype_dict = {}
        for col in df_sample.columns:
            dtype_obj = df_sample[col].dtype
            if dtype_obj == 'float64' or col in ['Carga Horaria', 'Carga Horaria Minima', 'Idade']:
                dtype_dict[col] = 'float64'
            elif col in ['Ano']:
                dtype_dict[col] = 'int64'


        
        print(f"Tipos de dados detectados: {len(dtype_dict)} colunas com dtype específico")
        
        # Step 5: Ler CSV com Dask usando colnames_map para renomear
        # Ler nomes originais do CSV (usando o primeiro arquivo da lista)
        df_original_header = pd.read_csv(
            lista_arquivos[0], 
            sep=';', 
            encoding='utf-8', 
            nrows=0
        )
        
        # Criar mapa: nome_original -> nome_normalizado
        # Motivo: O script precisa remover acentos e espaços dos nomes das colunas. O Dask tem dificuldade em renomear colunas "on-the-fly" se você não disser de onde para onde o nome está mudando. Lemos nrows=0 (0 milissegundos) apenas para pegar os nomes originais e montar o dicionário colnames_map (ex: {"Renda Básica": "Renda Basica"}).
        colnames_map = {
            original: remover_acentos(original).strip() 
            for original in df_original_header.columns
        }
        
        # Ler CSV com Dask (com nomes originais primeiro)
        df = dd.read_csv(
            path_input, 
            sep=';', 
            encoding='utf-8', 
            blocksize=f'{CHUNK_SIZE_MB}MB',
            assume_missing=True,
            dtype={'Matrícula Atendida': 'object'}
        )
        
        print_elapsed(f"CSV carregado em {df.npartitions} partições com {len(df.columns)} colunas")
        
        # Step 6: Normalizar nomes de colunas do Dask DataFrame
        df.columns = [colnames_map[col] for col in df.columns]
        print_elapsed("Nomes de colunas normalizados")
        
        # Step 7: Criar meta correto com a amostra normalizada
        # O meta é uma amostra vazia do DataFrame que serve para dizer ao Dask como deve ser a estrutura do DataFrame
        # (quais colunas, qual ordem e qual tipo de dado)
        meta = df_sample.head(0).copy()
        
        print(f"Meta criado com {len(meta.columns)} colunas")
        
        df = df.dropna()
        print_elapsed("NaN removidos")

        print_elapsed(f"Preparação concluída. DataFrame pronto com {df.npartitions} partições")

        print_elapsed("Iniciando etapa 2")
        print("")
        print("============================== etapa 2 ================================================================")
        print("")

        # --- FILTRAGEM E DEFINIÇÃO DO TARGET (em chunks) ---
        def filtrar_e_criar_target(partition):
            """Filtra e cria coluna Alvo_Evadido em cada partição"""
            if 'Categoria da Situacao' in partition.columns:
                #partition['Categoria da Situacao'] = partition['Categoria da Situacao'].astype(str).str.strip()
                partition['Alvo_Evadido'] = (partition['Categoria da Situacao'] == 'Evadidos').astype(int) # é aqui que tem que modificar para adicionar uma terceira classe ou remover os em curso
            gc.collect()
            return partition

        # Criar meta com coluna Alvo_Evadido
        meta_with_target = meta.copy()
        if 'Alvo_Evadido' not in meta_with_target.columns:
            meta_with_target['Alvo_Evadido'] = 0
        
        # O map_partitions pega uma função normal que você criou (escrita para funcionar no Pandas) e a aplica individualmente  a cada um desses pequenos pedaços (partições) de forma paralela (como um map).
        #Imagine que você tem um livro de 12 mil páginas (seu arquivo gigante) e 12 trabalhadores (os 12 núcleos do seu processador).
        # O map_partitions distribui 1.000 páginas para cada trabalhador. Cada trabalhador lê seu "mini-livro" ao mesmo tempo que os outros (paralelismo) e aplica a função de limpeza.
        # No final, o Dask junta todas as páginas limpas de volta na ordem correta.

        #O parâmetro meta é como um "contrato" ou uma "planta baixa" que você entrega para o Dask antes dele começar a trabalhar.
        #Aviso prévio: O meta é um DataFrame vazio (sem linhas, só com o cabeçalho) que diz ao Dask exatamente como o resultado daquela função vai ficar.
        #Evita erros: No seu código, a função filtrar_e_criar_target cria uma coluna nova chamada Alvo_Evadido. Se você não passasse o meta=meta_with_target (que já tem essa coluna avisada nele), o Dask iria quebrar lá na frente dizendo: "Ei, você está tentando usar uma coluna 'Alvo_Evadido' que não existia no arquivo original e ninguém me avisou que ela ia ser criada!"
        df = df.map_partitions(filtrar_e_criar_target, meta=meta_with_target)
        print_elapsed("Filtragem e criação de Alvo_Evadido concluída")

        # --- LIMPEZA: Remover colunas desnecessárias (em chunks) ---
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
            'Unidade de Ensino', 'Vagas Extraordinarias AC', 'Co Inst','Vagas Extraordinarias l1', 'Regiao', 'UF', 'Numero de registros',
            'Matricula Atendida'
        ]
        
        # Remover apenas colunas que existem no DataFrame
        colunas_existentes = [col for col in colunas_para_remover if col in df.columns]
        print(f"Removendo {len(colunas_existentes)} colunas de {len(df.columns)}")
        df = df.drop(columns=colunas_existentes)
        
        # Atualizar meta com colunas removidas
        meta_with_target = meta_with_target.drop(columns=[col for col in colunas_existentes if col in meta_with_target.columns])
        
        print_elapsed("Colunas desnecessárias removidas")

        print("")
        print("============================== etapa 3 ================================================================")
        print("")
        
        # --- TRATAMENTO DE VARIÁVEIS CATEGÓRICAS ---
        # 5. Categóricos Ordinais: Renda Familiar (em chunks)
        def tratar_renda_familiar(partition):
            """Trata a coluna de Renda Familiar em cada partição"""
            mapping_renda = {
                'Nao declarada': 0,
                '0<RFP<=0,5': 1,
                '0,5<RFP<=1,0': 2,
                '1,0<RFP<=1,5': 3,
                '1,5<RFP<=2,5': 4,
                '2,5<RFP<=3,5': 5,
                'RFP>3,5': 6
            }
            if 'Renda Familiar' in partition.columns:
                partition['Renda Familiar'] = partition['Renda Familiar'].map(mapping_renda).fillna(-1).astype('int64')
            gc.collect()
            return partition

        if 'Renda Familiar' in df.columns:
            df = df.map_partitions(tratar_renda_familiar, meta=meta_with_target)
            print_elapsed("Tratamento de Renda Familiar concluído")
        else:
            print_elapsed("Coluna Renda Familiar não encontrada - pulando tratamento")

        # 6. Categóricos Nominais (Label Encoding) - Pré-treinar encoders
        colunas_label_encoding = [
            'Fonte de Financiamento',
            'Nome de Curso',     
            'Instituicao',
            'Cor / Raca',
            'Turno',
            'Sexo', 'Subeixo Tecnologico', 'Tipo de Curso', 'Tipo de Oferta',
            'Eixo Tecnologico', 'Modalidade de Ensino', 
            'Municipio', 'Faixa Etaria',
        ]

        # Pré-treinar os LabelEncoders em toda a distribuição
        print_elapsed("Iniciando treinamento dos LabelEncoders")
        encoders_dict = {}
        colunas_para_encode = []
        
        for col in colunas_label_encoding:
            if col in df.columns:
                colunas_para_encode.append(col)
                # Obter valores únicos da coluna em todo o Dask DataFrame
                unique_values = df[col].astype(str).unique().compute()
                le = LabelEncoder()
                le.fit(unique_values)
                encoders_dict[col] = le
        
        print_elapsed(f"LabelEncoders treinados para {len(encoders_dict)} colunas")

        # Aplicar label encoding usando map_partitions
        if colunas_para_encode:
            df = df.map_partitions(
                aplicar_label_encoding_partition, 
                colunas_label_encoding=colunas_para_encode,
                encoders_dict=encoders_dict,
                meta=meta_with_target
            )
            print_elapsed(f"Label Encoding aplicado a {df.npartitions} partições. Iniciando onehotencoding...")
        
        # 7. One-Hot Encoding para colunas nominais
        colunas_onehotencoding = [
            

        ]
        
        # Validar que as colunas existem
        colunas_onehotencoding = [col for col in colunas_onehotencoding if col in df.columns]


        if colunas_onehotencoding:
            # Converter para categorical dtype antes de get_dummies (requerido pelo Dask)
            # O que essa linha faz é forçar o Dask a dar uma "espiada" rápida em todas as partições do arquivo inteiro só para levantar o inventário de quais são todas as palavras possíveis (categorias) que existem naquelas colunas específicas.
            #  o Dask processa os dados aos pedaços. Imagine que o Dask está lendo a Partição 1 e lá dentro só existem alunos do turno "Matutino". Se o Dask usasse o get_dummies ali na hora, ele criaria apenas 1 coluna (Turno_Matutino). Aí ele vai ler a Partição 2, onde calhou de ter "Vespertino" e "Noturno". Ele criaria 2 colunas diferentes. No final, quando ele tentasse juntar os pedaços, ocorreria um erro fatal porque a Partição 1 teria um número de colunas diferente da Partição 2.
            df = df.categorize(columns=colunas_onehotencoding)
            print_elapsed(f"Colunas {colunas_onehotencoding} convertidas para categorical")
            
            # quando essa linha roda (dd.get_dummies), o Dask já sabe a lista completa de categorias. Se a Partição 1 só tiver alunos do turno "Matutino", o Dask será esperto o suficiente para criar a coluna Turno_Vespertino e Turno_Noturno e encher tudo com 0, garantindo que todas as partições tenham sempre exatamente as mesmas colunas antes de serem salvas no disco.
            df = dd.get_dummies(df, columns=colunas_onehotencoding, drop_first=True, dtype=int)
            print_elapsed("One-Hot Encoding aplicado")
        
        print(f"Quantidade de colunas do DataFrame após transformações: {df.shape[1]}. Gravando dados em chunks agora...")

        # Salvar resultado em partições
        # Como estamos processando todos os anos, usamos um nome genérico
        path_output = f'./processados_LabelEncoding/todos_anos_quiquadrado_*.csv'
        df.to_csv(path_output, index=False)
        print_elapsed(f"Arquivos salvos com sucesso em: {path_output}")

        print(f"✅ Sucesso! ")

        print("")
        print("============================== FIM ================================================================")
        print("")

        data_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"\n{'='*70}")
        print(f"🕐 Fim do processamento: {data_hora}")
        total_time = time.time() - start_time
        print(f"⏱️ Tempo total do processamento: {total_time:.2f} segundos")
        print(f"{'='*70}\n")

        # Exibir mapeamento de Turno se disponível
        if 'Turno' in encoders_dict:
            print("\n📊 MAPEAMENTO DE LABEL ENCODING - CAMPO 'TURNO':")
            print("=" * 60)
            le_turno = encoders_dict['Turno']
            for valor_original, valor_codificado in zip(le_turno.classes_, le_turno.transform(le_turno.classes_)):
                print(f"  '{valor_original}' → {valor_codificado}")
            print("=" * 60)

    
        del df
        gc.collect()

    finally:
        client.close()
        cluster.close()

if __name__ == '__main__':
    main()