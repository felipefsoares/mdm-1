# 📊 Processamento Batch de Microdados Educacionais com Dask

Este conjunto de scripts automatiza o processamento de múltiplos arquivos CSV de microdados educacionais usando Dask para processamento em chunks distribuído.

## 📁 Estrutura de Diretórios

```
.
├── chi2-parte1-dask.py          # Script principal (parametrizado)
├── process_all.sh               # Script bash para processamento batch
├── process_all.ps1              # Script PowerShell para processamento batch
├── datasets-nilo/               # Entrada de dados
│   ├── microdados_matriculas_2023.csv
│   ├── microdados_matriculas_2022.csv
│   └── ... (outros CSVs)
└── processados/                 # Saída de dados
    ├── dask/
    │   ├── dados_quiquadrado_*.csv  (arquivos parciais)
    └── dados_consolidados.csv       (arquivo final consolidado)
```

## 🚀 Como Usar

### Opção 1: Processar um arquivo individual

```bash
python chi2-parte1-dask.py ./datasets-nilo/microdados_matriculas_2023.csv
```

### Opção 2: Processar todos os arquivos (Linux/Mac)

```bash
bash process_all.sh
```

### Opção 3: Processar todos os arquivos (Windows - PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File process_all.ps1
```

## 📋 O que cada script faz

### `chi2-parte1-dask.py`
- Lê um arquivo CSV parametrizado
- Normaliza nomes de colunas (remove acentos)
- Remove NaN
- Cria coluna alvo (Alvo_Evadido)
- Remove colunas desnecessárias
- Aplica transformações categóricas:
  - Mapeamento ordinal para Renda Familiar
  - Label Encoding para variáveis nominais
  - One-Hot Encoding para Nome de Curso
- Salva resultados em chunks em `processados/dask/`

### `process_all.sh` (Bash)
1. Loop sobre todos os CSVs em `datasets-nilo/`
2. Chama `chi2-parte1-dask.py` para cada arquivo
3. Consolida todos os arquivos `dados_quiquadrado_*.csv` em um único arquivo
4. Salva resultado em `processados/dados_consolidados.csv`

### `process_all.ps1` (PowerShell)
- Mesma funcionalidade que o script bash, mas para Windows

## ⚙️ Configurações

Você pode ajustar as seguintes configurações em `chi2-parte1-dask.py`:

```python
CHUNK_SIZE_MB = 50              # Tamanho dos chunks em MB
NUM_WORKERS = 10                # Número de workers Dask
THREADS_PER_WORKER = 1          # Threads por worker
MEMORY_LIMIT_PER_WORKER = '3GB' # Limite de memória por worker
```

## 📊 Saídas

### Arquivos Intermediários
- `./processados/dask/dados_quiquadrado_*.csv` - Chunks processados (um por partição)

### Arquivo Final
- `./processados/dados_consolidados.csv` - Todos os dados consolidados em um único arquivo

## 🔍 Exemplo de Execução

```bash
# Processar arquivo individual
$ python chi2-parte1-dask.py ./datasets-nilo/microdados_matriculas_2023.csv

# Ou processar todos os arquivos
$ bash process_all.sh

# Saída esperada:
# ✅ 7 arquivos processados
# ✅ 1.234.567 linhas consolidadas
# 💾 Salvo em ./processados/dados_consolidados.csv
```

## 📝 Tratamentos Aplicados

1. **Normalização de Nomes**: Remove acentos e espaços extras
2. **Limpeza de Texto**: Remove acentos do conteúdo
3. **Remoção de Colunas**: Remove colunas desnecessárias
4. **Variáveis Ordinais**: Mapeia Renda Familiar em ordem lógica (0-6)
5. **Label Encoding**: Converte variáveis categóricas para números
6. **One-Hot Encoding**: Expande Nome de Curso em colunas binárias
7. **Categorização**: Converte para dtype categórico para otimização

## 🐛 Troubleshooting

### "Arquivo não encontrado"
- Verifique se o caminho está correto
- Use caminhos relativos a partir do diretório do script

### "Memória insuficiente"
- Reduza `CHUNK_SIZE_MB`
- Reduza `NUM_WORKERS`
- Reduza `MEMORY_LIMIT_PER_WORKER`

### Erros de encoding
- Certifique-se que os CSVs usam UTF-8
- Verifique o separador (padrão: `;`)

## 📈 Performance

Tempos típicos (com 10 workers, 3GB por worker):
- Leitura: ~30s
- Processamento: ~5-10 min
- Consolidação: ~10s
- **Total: ~15 min** para ~7 arquivos de 1GB cada

## 📞 Suporte

Para problemas ou dúvidas, verifique:
1. Os logs no terminal
2. Os timestamps de início/fim exibidos
3. A estrutura de diretórios



## Testes

transforme este codigo para suportar o processamento em chinks do dask
Principais Mudanças:
1. Configurações de Chunk Customizáveis
•	CHUNK_SIZE_MB: Tamanho dos chunks (padrão 100MB)
•	Configuração do cluster com workers e limites de memória
2. Processamento com map_partitions
•	remover_acentos_partition(): Limpa acentos em cada chunk/partição
•	filtrar_e_criar_target(): Cria coluna-alvo em paralelo
•	tratar_renda_familiar(): Mapeia valores ordinais por chunk
•	aplicar_label_encoding_partition(): Aplica encodings pré-treinados
3. Label Encoding Otimizado
•	Pré-treina os LabelEncoders uma única vez com todos os valores únicos
•	Aplica os encoders pré-treinados a cada partição via map_partitions
•	Evita retreinamento desnecessário e garante consistência
4. Melhorias de Logs
•	Exibe número de partições processadas
•	Mostra status de cada etapa com tempo decorrido
5. Gerenciamento de Recursos
•	Bloco try/finally garante limpeza do cluster e cliente Dask
•	Melhor controle de memória com processamento distribuído
O código agora processa os dados de forma distribuída, aproveitando melhor os recursos e sendo escalável para arquivos maiores!
 
Verificação de tipos e tratamento de dados via dask
1. Leitura e Normalização de Colunas
•	Amostra inicial de 1000 linhas para melhor inferência de tipos
•	Criação automática de mapa: nome_original → nome_normalizado
•	Limpeza de acentos e espaços em branco de todos os nomes de colunas
2. Mapeamento de Metadados Correto
•	criar_meta_normalizado() - cria metadados com nomes normalizados
•	Meta criado a partir da amostra já processada
•	Garante que map_partitions recebe meta com as colunas corretas
3. Estratégia de Processamento sem Desalinhamento
•	Normalização feita antes de usar map_partitions (não dentro)
•	Meta sempre reflete os nomes normlizados
•	Validação de colunas antes de operações (verifica existência)
4. Label Encoding Robusto
•	Lista dinâmica de colunas a codificar (verifica quais existem)
•	Trata valores desconhecidos com valor padrão (-1)
•	Pré-treina encoders uma única vez
5. Tratamento de Erros
•	Blocos if col in df.columns antes de qualquer operação
•	Validação de dados antes de aplicar transformações
•	Relatório de quantos encoders foram treinados vs solicitados
 
Relatório do pré processamento paralelo com DASK processamento
- rodada 1
CHUNK_SIZE_MB = 100  # Tamanho dos chunks em MB
NUM_WORKERS = 6
THREADS_PER_WORKER = 1
MEMORY_LIMIT_PER_WORKER = '2GB'
Após vários minutos tentando rodas, os Workers começaram morrer e ressucitar sem parar, na etapada de onehotencoding, tentar novamente com mais memória e chinks menores
- rodada 2
Início do processamento: 27/04/2026 09:55:39
Fim do processamento: 27/04/2026 10:04:31
Tempo total: 8 minutos e 52 segundos.
CHUNK_SIZE_MB = 50   # Tamanho dos chunks em MB
NUM_WORKERS = 6
THREADS_PER_WORKER = 1
MEMORY_LIMIT_PER_WORKER = '3GB'

-rodada 3
Tentando agora com 10 workers
Início do processamento: 27/04/2026 10:07:17
Fim do processamento: 27/04/2026 10:15:16
Tempo total:7 minutos e 59 segundos.

CHUNK_SIZE_MB = 50   # Tamanho dos chunks em MB
NUM_WORKERS = 10
THREADS_PER_WORKER = 1
MEMORY_LIMIT_PER_WORKER = '3GB'
Memória em uso: 22,8GB



- rodada 4
processando todos os arquivos
⏰ Início do processamento: 27/04/2026 10:30:18

