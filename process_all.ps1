# Script para processar todos os arquivos CSV de entrada e consolidar os resultados
# Uso: powershell -ExecutionPolicy Bypass -File process_all.ps1

# Configuração
$DatasetDir = ".\datasets-nilo"
$OutputDir = ".\processados"
$DaskOutputDir = ".\processados\dask"
$ConsolidatedOutput = ".\processados\dados_consolidados.csv"

# Cores
$Green = "Green"
$Red = "Red"
$Yellow = "Yellow"

Write-Host "`n" -ForegroundColor Green
Write-Host "╔════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  Script de Processamento Batch - Microdados Educacionais com Dask         ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host "`n"

# Verificar se o diretório de entrada existe
if (-not (Test-Path $DatasetDir)) {
    Write-Host "❌ Erro: Diretório $DatasetDir não encontrado" -ForegroundColor Red
    exit 1
}

# Criar diretório de saída se não existir
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

if (-not (Test-Path $DaskOutputDir)) {
    New-Item -ItemType Directory -Path $DaskOutputDir | Out-Null
}

# Limpar saídas anteriores
Write-Host "🧹 Limpando saídas anteriores..." -ForegroundColor Yellow
Remove-Item "$DaskOutputDir\dados_quiquadrado_*.csv" -ErrorAction SilentlyContinue
Remove-Item $ConsolidatedOutput -ErrorAction SilentlyContinue

# Contar e processar arquivos CSV
$csvFiles = @(Get-ChildItem -Path $DatasetDir -Filter "*.csv" | Sort-Object Name)
$csvCount = $csvFiles.Count

if ($csvCount -eq 0) {
    Write-Host "❌ Nenhum arquivo CSV encontrado em $DatasetDir" -ForegroundColor Red
    exit 1
}

Write-Host "📂 Encontrados $csvCount arquivo(s) CSV para processar`n" -ForegroundColor Yellow

$index = 1
foreach ($arquivo in $csvFiles) {
    $filename = $arquivo.Name
    $fullPath = $arquivo.FullName
    
    Write-Host ("─" * 80) -ForegroundColor Yellow
    Write-Host "📊 [$index/$csvCount] Processando: $filename" -ForegroundColor Green
    Write-Host ("─" * 80) -ForegroundColor Yellow
    
    # Executar o script Python
    & python chi2-parte1-dask.py $fullPath
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $filename processado com sucesso`n" -ForegroundColor Green
    } else {
        Write-Host "❌ Erro ao processar $filename" -ForegroundColor Red
        exit 1
    }
    
    $index++
}

Write-Host ("─" * 80) -ForegroundColor Yellow
Write-Host "📦 Consolidando arquivos..." -ForegroundColor Green
Write-Host ("─" * 80) -ForegroundColor Yellow

# Consolidar usando Python
$pythonScript = @'
import pandas as pd
import os
import glob

dask_output_dir = r".\processados\dask"
consolidated_output = r".\processados\dados_consolidados.csv"

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
'@

# Executar script Python de consolidação
$pythonScript | & python -

Write-Host "`n"
Write-Host "╔════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  ✅ Processamento Concluído com Sucesso!                                   ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host "`n"

Write-Host "📊 Resumo:" -ForegroundColor Yellow
Write-Host "   • Arquivos processados: $csvCount"
Write-Host "   • Saída consolidada: $ConsolidatedOutput"
Write-Host "   • Arquivos intermediários: $DaskOutputDir"
Write-Host "`n"
