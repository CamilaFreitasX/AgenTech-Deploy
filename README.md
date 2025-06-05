# 🤖 Agente Inteligente para Análise de Notas Fiscais

> **Análise de dados de notas fiscais usando IA conversacional com Google Gemini**

Uma aplicação avançada que permite fazer consultas em linguagem natural sobre arquivos CSV de notas fiscais, utilizando **Google Gemini API** e **LangChain** para processamento inteligente de dados.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Google Gemini](https://img.shields.io/badge/Google-Gemini%20API-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-yellow.svg)

## 🌟 Características Principais

- ✅ **Interface Web Intuitiva** com Streamlit
- ✅ **Google Gemini API** para processamento de linguagem natural
- ✅ **1.5 bilhão de tokens gratuitos por dia**
- ✅ **Descompactação automática** de arquivos ZIP
- ✅ **Identificação automática** de tipos de arquivo (cabeçalho/itens)
- ✅ **Consultas inteligentes** usando agentes LangChain
- ✅ **Respostas em português brasileiro**
- ✅ **Validação de dados** integrada
- ✅ **Dados de exemplo** para teste imediato
- ✅ **Suporte a múltiplos CSVs** simultaneamente

## 🚀 Instalação e Configuração

### Pré-requisitos
- Python 3.8 ou superior
- Conta Google (para API Key gratuita)
- Conexão com internet

### Método 1: Instalação Automática (Recomendado)

```bash
# Execute o script de instalação
python install_google.py

Trae

O usuário quer que eu coloque o detalhamento completo no README. Vou atualizar o arquivo README.md com todas as informações detalhadas sobre a aplicação.

README.md :( 2 - 100 )

Perfeito! Vou criar um README.md completo e detalhado com todas as informações sobre a aplicação. Aqui está o conteúdo atualizado:

```
# 🤖 Agente Inteligente para Análise de Notas Fiscais

> **Análise de dados de notas fiscais usando IA 
conversacional com Google Gemini**

Uma aplicação avançada que permite fazer consultas em 
linguagem natural sobre arquivos CSV de notas fiscais, 
utilizando **Google Gemini API** e **LangChain** para 
processamento inteligente de dados.

![Python](https://img.shields.io/badge/Python-3.8+-blue.
svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28
+-red.svg)
![Google Gemini](https://img.shields.io/badge/
Google-Gemini%20API-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1
+-yellow.svg)

## 🌟 Características Principais

- ✅ **Interface Web Intuitiva** com Streamlit
- ✅ **Google Gemini API** para processamento de linguagem 
natural
- ✅ **1.5 bilhão de tokens gratuitos por dia**
- ✅ **Descompactação automática** de arquivos ZIP
- ✅ **Identificação automática** de tipos de arquivo 
(cabeçalho/itens)
- ✅ **Consultas inteligentes** usando agentes LangChain
- ✅ **Respostas em português brasileiro**
- ✅ **Validação de dados** integrada
- ✅ **Dados de exemplo** para teste imediato
- ✅ **Suporte a múltiplos CSVs** simultaneamente

## 🚀 Instalação e Configuração

### Pré-requisitos
- Python 3.8 ou superior
- Conta Google (para API Key gratuita)
- Conexão com internet

### Método 1: Instalação Automática (Recomendado)

```bash
# Execute o script de instalação
python install_google.py
```
O script irá:

1. Verificar se o Python está instalado
2. Instalar automaticamente todas as dependências
3. Configurar o ambiente virtual (opcional)
4. Orientar sobre a configuração da API Key
### Método 2: Instalação Manual
```
# 1. Clone ou baixe o projeto
# 2. Instale as dependências
pip install -r requirements_google.txt

# 3. Configure a API Key (veja seção abaixo)
# 4. Execute a aplicação
streamlit run main_google.py
```
## 🔒 Segurança e Configuração

### ⚠️ IMPORTANTE: Configuração da API Key

**NUNCA** compartilhe sua Google API Key publicamente. Este repositório está configurado para proteger informações sensíveis:

1. **Arquivo .env**: Contém sua API Key (ignorado pelo Git)
2. **Arquivo .env.example**: Template para configuração

### Configuração Inicial

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/AgenTech.git
cd AgenTech

# 2. Copie o arquivo de exemplo
copy .env.example .env

# 3. Edite o .env e adicione sua API Key
# GOOGLE_API_KEY=sua_api_key_aqui

# 4. Instale as dependências
pip install -r requirements_google.txt

# 5. Execute a aplicação
streamlit run main_google.py
```
O script irá:

1. Verificar se o Python está instalado
2. Instalar automaticamente todas as dependências
3. Configurar o ambiente virtual (opcional)
4. Orientar sobre a configuração da API Key
### Método 2: Instalação Manual
```
# 1. Clone ou baixe o projeto
# 2. Instale as dependências
pip install -r requirements_google.txt

# 3. Configure a API Key (veja seção abaixo)
# 4. Execute a aplicação
streamlit run main_google.py
```
### 🔑 Configuração da Google API Key Obter API Key (GRATUITA)
1. Acesse Google AI Studio
2. Faça login com sua conta Google
3. Clique em "Create API Key"
4. Copie a chave gerada Configurar no Sistema
Opção 1: Arquivo .env (Recomendado)

```
# Crie um arquivo .env na pasta do projeto
echo GOOGLE_API_KEY=sua_api_key_aqui > .env
```
Opção 2: Variável de Ambiente

```
# Windows
set GOOGLE_API_KEY=sua_api_key_aqui

# Linux/Mac
export GOOGLE_API_KEY=sua_api_key_aqui
```
Opção 3: Interface da Aplicação

- A aplicação permite inserir a API Key diretamente na interface web
## 🎯 Como Usar
### 1. Executar a Aplicação
```
streamlit run main_google.py
```
A aplicação abrirá automaticamente no navegador em http://localhost:8501

### 2. Configurar API Key
- Se não configurou via arquivo .env, insira sua API Key na barra lateral
- A chave será validada automaticamente
### 3. Carregar Dados Opção A: Usar Dados de Exemplo
- Clique em "Usar dados de exemplo" para carregar dados de demonstração
- Ideal para testar a aplicação imediatamente Opção B: Upload de Arquivo ZIP
- Faça upload de um arquivo ZIP contendo seus CSVs
- A aplicação descompacta e identifica automaticamente os arquivos
- Suporta múltiplos CSVs de cabeçalho e itens
### 4. Fazer Consultas Exemplos de Perguntas
Análises Básicas:

- "Qual o valor total das notas fiscais?"
- "Quantas notas fiscais foram emitidas?"
- "Qual a média de valor por nota?"
Análises por Período:

- "Quais foram as vendas em janeiro de 2024?"
- "Mostre o faturamento por mês"
- "Qual foi o melhor mês de vendas?"
Análises por Cliente:

- "Quem são os 5 maiores clientes?"
- "Qual cliente comprou mais produtos?"
- "Mostre o ranking de clientes por valor"
Análises por Produto:

- "Quais são os produtos mais vendidos?"
- "Qual produto tem maior margem?"
- "Mostre a análise ABC dos produtos"
Análises Avançadas:

- "Identifique tendências de vendas"
- "Faça uma análise de sazonalidade"
- "Quais produtos estão em declínio?"
- "Analise a concentração de vendas por região"
### 5. Interpretar Resultados
- As respostas são geradas em português brasileiro
- Incluem análises contextuais e insights
- Podem conter tabelas, gráficos textuais e resumos
- Detalhes técnicos ficam ocultos por padrão (disponíveis em expander)
## 📁 Estrutura de Arquivos
```
project/
├── main_google.py              # Aplicação principal 
Streamlit
├── utils_google.py             # Funções utilitárias
├── requirements_google.txt     # Dependências Python
├── install_google.py          # Script de instalação 
automática
├── test_app_google.py         # Testes da aplicação
├── .env                       # Configurações (API Key)
├── demo-gemini-notas.zip      # Dados de exemplo
└── README.md                  # Este arquivo
```
## 📊 Formato dos Dados
### Arquivo de Cabeçalho (header/cabecalho)
```
numero_nota,data_emissao,cliente,valor_total,status
001,2024-01-15,Cliente A,1500.00,Paga
002,2024-01-16,Cliente B,2300.50,Pendente
```
### Arquivo de Itens
```
numero_nota,produto,quantidade,valor_unitario,valor_total
001,Produto X,2,750.00,1500.00
002,Produto Y,1,2300.50,2300.50
```
### Campos Reconhecidos Automaticamente
Cabeçalho:

- Número da nota fiscal
- Data de emissão
- Cliente/Destinatário
- Valor total
- Status/Situação
Itens:

- Número da nota fiscal (chave de ligação)
- Produto/Serviço
- Quantidade
- Valor unitário
- Valor total do item
## ⚙️ Configuração Avançada
### Variáveis de Ambiente
```
# .env
GOOGLE_API_KEY=sua_api_key_aqui
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```
### Modelos Disponíveis
- gemini-1.5-flash (padrão) - Rápido e eficiente
- gemini-1.5-pro - Mais preciso para análises complexas
- gemini-1.0-pro - Versão estável anterior
### Personalização do Agente
O agente pode ser personalizado editando main_google.py :

```
# Configurações do agente
AGENT_CONFIG = {
    'model': 'gemini-1.5-flash',
    'temperature': 0.1,
    'language': 'português brasileiro',
    'max_tokens': 4000
}
```
## 🛠️ Tecnologias Utilizadas
- Streamlit - Interface web interativa
- Google Gemini API - Modelo de linguagem avançado
- LangChain - Framework para aplicações com LLM
- Pandas - Manipulação de dados
- Python - Linguagem de programação
## 🔧 Solução de Problemas
### Erro de API Key
```
Erro: Invalid API Key
```
Solução: Verifique se a API Key está correta e ativa no Google AI Studio

### Erro de Dependências
```
ModuleNotFoundError: No module named 'streamlit'
```
Solução: Execute pip install -r requirements_google.txt

### Erro de Arquivo CSV
```
Erro ao processar arquivo CSV
```
Solução: Verifique se o arquivo está no formato correto e não está corrompido

### Performance Lenta
- Use arquivos CSV menores para testes
- Considere usar o modelo gemini-1.5-flash para respostas mais rápidas
- Verifique sua conexão com internet
### Respostas em Inglês
- O sistema foi configurado para responder em português
- Se persistir, reinicie a aplicação
## 🚀 Melhorias Futuras
- Suporte a outros formatos (Excel, JSON)
- Visualizações gráficas interativas
- Exportação de relatórios
- Histórico de consultas
- API REST para integração
- Suporte a múltiplos idiomas
- Cache de respostas
- Análises preditivas
## 🤝 Como Contribuir
1. Faça um fork do projeto
2. Crie uma branch para sua feature ( git checkout -b feature/AmazingFeature )
3. Commit suas mudanças ( git commit -m 'Add some AmazingFeature' )
4. Push para a branch ( git push origin feature/AmazingFeature )
5. Abra um Pull Request
## 📄 Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## 🔗 Links Úteis
- Google AI Studio - Obter API Key
- Documentação Gemini - Documentação oficial
- LangChain Docs - Documentação LangChain
- Streamlit Docs - Documentação Streamlit
## ⚠️ Avisos Importantes
- Gratuidade: A Google oferece 1.5 bilhão de tokens gratuitos por dia
- Privacidade: Seus dados são processados localmente, apenas as consultas são enviadas para a API
- Limites: Respeite os limites de uso da API para evitar bloqueios
- Backup: Sempre mantenha backup dos seus dados originais

Desenvolvido com ❤️ para análise inteligente de notas fiscais
Para suporte técnico ou dúvidas, abra uma issue no repositório do projeto.