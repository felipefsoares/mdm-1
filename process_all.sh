#!/bin/bash

# Script para processar todos os arquivos CSV de entrada e consolidar os resultados
# Uso: bash process_all.sh

set -e  # Sair se houver erro

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Registrar horários
INICIO=$(date '+%d/%m/%Y %H:%M:%S')
INICIO_EPOCH=$(date +%s)

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  Script de Processamento Batch - Microdados Educacionais com Dask       ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}⏰ Início do processamento: $INICIO${NC}"
echo ""

# Diretórios
DATASET_DIR="./datasets-nilo"
OUTPUT_DIR="./processados"
DASK_OUTPUT_DIR="./processados/dask"
CONSOLIDATED_OUTPUT="./processados/dados_consolidados.csv"

# Verificar se o diretório de entrada existe
if [ ! -d "$DATASET_DIR" ]; then
    echo -e "${RED}❌ Erro: Diretório $DATASET_DIR não encontrado${NC}"
    exit 1
fi

# Criar diretório de saída se não existir
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DASK_OUTPUT_DIR"

# Limpar saídas anteriores
echo -e "${YELLOW}🧹 Limpando saídas anteriores...${NC}"
rm -f "$DASK_OUTPUT_DIR/dados_quiquadrado_*.csv"
rm -f "$CONSOLIDATED_OUTPUT"

echo ""

# Loop sobre todos os arquivos CSV
csv_count=0
for arquivo in "$DATASET_DIR"/*.csv; do
    if [ -f "$arquivo" ]; then
        csv_count=$((csv_count + 1))
        filename=$(basename "$arquivo")
        
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}📊 [$csv_count] Processando: $filename${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        # Executar o script Python
        if python chi2-parte1-dask.py "$arquivo"; then
            echo -e "${GREEN}✅ $filename processado com sucesso${NC}"
        else
            echo -e "${RED}❌ Erro ao processar $filename${NC}"
            exit 1
        fi
        
        echo ""
    fi
done

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📦 Consolidando arquivos...${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Verificar se há arquivos para consolidar
if ! ls "$DASK_OUTPUT_DIR"/dados_quiquadrado_*.csv 1> /dev/null 2>&1; then
    echo -e "${RED}❌ Nenhum arquivo processado encontrado para consolidar${NC}"
    exit 1
fi

# Consolidar todos os arquivos em um
echo "Lendo e consolidando arquivos..."
python << 'EOF'
import pandas as pd
import os
import glob

dask_output_dir = "./processados/dask"
consolidated_output = "./processados/dados_consolidados.csv"

# Encontrar todos os arquivos
arquivos = sorted(glob.glob(os.path.join(dask_output_dir, "dados_quiquadrado_*.csv")))

if not arquivos:
    print("❌ Nenhum arquivo encontrado")
    exit(1)

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

# Concatenar
df_consolidado = pd.concat(dfs, ignore_index=True)

# Salvar
df_consolidado.to_csv(consolidated_output, index=False)

print(f"\n✅ Consolidação concluída!")
print(f"   📊 Total de linhas: {len(df_consolidado)}")
print(f"   📊 Total de colunas: {len(df_consolidado.columns)}")
print(f"   💾 Arquivo: {consolidated_output}")

EOF

FIM=$(date '+%d/%m/%Y %H:%M:%S')

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ Processamento Concluído com Sucesso!                                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}📊 Resumo:${NC}"
echo "   • Arquivos processados: $csv_count"
echo "   • Saída consolidada: $CONSOLIDATED_OUTPUT"
echo "   • Arquivos intermediários: $DASK_OUTPUT_DIR"
echo "   • Fim: $FIM"

echo ""
