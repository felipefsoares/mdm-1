#!/usr/bin/env python
"""
Script para consolidar arquivos CSV processados pelo chi2-parte1-dask.py

Uso:
    python consolidate_dados.py                    # Usa valores padrão
    python consolidate_dados.py -i ./processados/dask -o ./processados/consolidated.csv
"""

import pandas as pd
import os
import glob
import argparse
from pathlib import Path


def consolidar_dados(dask_output_dir, consolidated_output):
    """Consolida todos os arquivos CSV do diretório dask em um único arquivo"""
    
    # Encontrar todos os arquivos (padrão: *_quiquadrado_*.csv)
    arquivos = sorted(glob.glob(os.path.join(dask_output_dir, "*_quiquadrado_*.csv")))
    
    if not arquivos:
        print("❌ Nenhum arquivo encontrado")
        return False
    
    print(f"\n📂 Encontrados {len(arquivos)} arquivos:")
    for arquivo in arquivos:
        print(f"   - {os.path.basename(arquivo)}")
    
    # Ler e consolidar
    print("\n📥 Consolidando...")
    dfs = []
    total_linhas = 0
    
    for arquivo in arquivos:
        try:
            df = pd.read_csv(arquivo)
            dfs.append(df)
            total_linhas += len(df)
            print(f"   ✓ {os.path.basename(arquivo)}: {len(df)} linhas")
        except Exception as e:
            print(f"   ✗ Erro ao ler {arquivo}: {e}")
            return False
    
    if not dfs:
        print("❌ Nenhum arquivo foi lido com sucesso")
        return False
    
    # Concatenar
    df_consolidado = pd.concat(dfs, ignore_index=True)
    
    # Criar diretório de saída se não existir
    output_dir = os.path.dirname(consolidated_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Salvar
    df_consolidado.to_csv(consolidated_output, index=False)
    
    print(f"\n✅ Consolidação concluída!")
    print(f"   📊 Total de linhas: {len(df_consolidado)}")
    print(f"   📊 Total de colunas: {len(df_consolidado.columns)}")
    print(f"   💾 Arquivo: {consolidated_output}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolida arquivos CSV processados em um único arquivo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python consolidate_dados.py
  python consolidate_dados.py -i ./processados/dask -o ./processados/dados_consolidados.csv
        """
    )
    
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default="./processados/dask",
        help="Diretório contendo os arquivos CSV (padrão: ./processados/dask)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./processados/dados_consolidados.csv",
        help="Caminho para o arquivo de saída consolidado (padrão: ./processados/dados_consolidados.csv)"
    )
    
    args = parser.parse_args()
    
    # Validar se o diretório existe
    if not os.path.isdir(args.input_dir):
        print(f"❌ Erro: Diretório '{args.input_dir}' não encontrado")
        exit(1)
    
    # Executar consolidação
    sucesso = consolidar_dados(args.input_dir, args.output)
    exit(0 if sucesso else 1)
