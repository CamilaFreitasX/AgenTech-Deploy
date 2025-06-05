import streamlit as st
import pandas as pd
import zipfile
import os
import tempfile
from io import StringIO

# Import do agente CSV
try:
    from langchain_experimental.agents import create_csv_agent
    from langchain.agents import create_pandas_dataframe_agent
except ImportError:
    try:
        from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
    except ImportError:
        st.error("Erro: Não foi possível importar os agentes do LangChain")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from utils_google import NotaFiscalValidator, extract_zip_file
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

class CSVAnalysisAgent:
    def __init__(self, google_api_key=None):
        """Inicializa o agente de análise CSV com Google Gemini"""
        self.google_api_key = google_api_key
        self.agents = {}
        self.dataframes = {}
        self.file_info = {}
        
    def create_llm(self):
        """Cria uma instância do modelo Google Gemini com configurações otimizadas"""
        if not self.google_api_key:
            raise ValueError("Google API Key é necessária para usar o agente")
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,  # Reduzido para maior precisão
                convert_system_message_to_human=True,
                max_tokens=2048,
                top_p=0.8
            )
            return llm
        except Exception as e:
            st.error(f"Erro ao criar modelo Gemini: {str(e)}")
            return None
    
    def load_csv_data(self, file_path, file_type):
        """Carrega dados CSV e cria agente específico"""
        try:
            # Tentar diferentes encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Não foi possível ler o arquivo com nenhum encoding testado")
            
            # Limpar nomes das colunas
            df.columns = df.columns.str.strip()
            
            # Converter valores monetários se existirem
            for col in df.columns:
                if 'valor' in col.lower() or 'preco' in col.lower():
                    if df[col].dtype == 'object':
                        # Remover símbolos monetários e converter
                        df[col] = df[col].astype(str).str.replace(r'[R$\s]', '', regex=True)
                        df[col] = df[col].str.replace(',', '.', regex=False)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            
            self.dataframes[file_type] = df
            self.file_info[file_type] = {
                'path': file_path,
                'shape': df.shape,
                'columns': df.columns.tolist()
            }
            
            return True
            
        except Exception as e:
            st.error(f"Erro ao carregar arquivo {file_type}: {str(e)}")
            return False
    
    def create_general_agent(self):
        """Cria um agente geral que pode acessar todos os dataframes"""
        try:
            if not self.dataframes:
                print("❌ Erro: Nenhum dataframe carregado")
                return None
            
            llm = self.create_llm()
            if llm is None:
                print("❌ Erro: Não foi possível criar LLM")
                return None
            
            # Usar pandas dataframe agent para melhor controle
            all_dfs = list(self.dataframes.values())
            df_names = list(self.dataframes.keys())
            
            if len(all_dfs) == 0:
                print("❌ Erro: Nenhum dataframe válido encontrado")
                return None
            
            try:
                # Criar prefix detalhado para o agente
                prefix = f"""
Você é um especialista em análise de dados financeiros de notas fiscais no Brasil.

DATAFRAMES DISPONÍVEIS:
{chr(10).join([f"- df_{i} ({name}): {df.shape[0]} linhas, {df.shape[1]} colunas" for i, (name, df) in enumerate(zip(df_names, all_dfs))])}

REGRAS CRÍTICAS PARA ANÁLISE:
1. SEMPRE agrupe dados por fornecedor antes de calcular valores
2. SEMPRE some todos os valores por fornecedor (não pegue valores individuais)
3. Use pandas groupby() para agregações
4. Para encontrar maior valor: df.groupby('coluna_fornecedor')['coluna_valor'].sum().idxmax()
5. SEMPRE converta colunas de valor para numérico antes de somar
6. Responda SEMPRE em português brasileiro
7. Mostre o código pandas executado
8. Valide resultados com múltiplas verificações

ESTRUTURA PADRÃO DE RESPOSTA:
1. Código pandas executado
2. Resultado da análise
3. Verificação dos dados
4. Resposta final clara e precisa

IMPORTANTE: Nunca invente dados - use apenas o que está nos dataframes carregados.
"""
                
                general_agent = create_pandas_dataframe_agent(
                    llm,
                    all_dfs,
                    verbose=True,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    allow_dangerous_code=True,
                    handle_parsing_errors=True,
                    max_iterations=50,
                    early_stopping_method="generate",
                    prefix=prefix,
                    agent_executor_kwargs={
                        "handle_parsing_errors": True,
                        "max_execution_time": 900  # 15 minutos
                    }
                )
                
                print(f"✅ Agente pandas criado com {len(all_dfs)} dataframes")
                return general_agent
                
            except Exception as e:
                print(f"❌ Erro ao criar agente pandas: {str(e)}")
                
                # Fallback para CSV agent com arquivo único
                valid_paths = []
                for file_type in self.dataframes.keys():
                    file_path = self.file_info.get(file_type, {}).get('path')
                    if file_path and os.path.exists(file_path):
                        valid_paths.append(file_path)
                
                if valid_paths:
                    try:
                        print(f"⚠️ Fallback: criando CSV agent para {valid_paths[0]}")
                        general_agent = create_csv_agent(
                            llm,
                            valid_paths[0],
                            verbose=True,
                            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            allow_dangerous_code=True,
                            handle_parsing_errors=True,
                            max_iterations=50
                        )
                        print(f"⚠️ CSV Agent criado para: {valid_paths[0]}")
                        return general_agent
                    except Exception as e2:
                        print(f"❌ Erro no fallback CSV: {str(e2)}")
                        return None
                
                return None
                
        except Exception as e:
            print(f"❌ Erro geral ao criar agente: {str(e)}")
            st.error(f"Erro ao criar agente geral: {str(e)}")
            return None
    
    def analyze_data_directly(self, question):
        """Análise direta usando pandas sem LangChain (fallback)"""
        try:
            # Identificar o dataframe de cabeçalho
            cabecalho_df = None
            for file_type, df in self.dataframes.items():
                if 'cabecalho' in file_type.lower() or 'header' in file_type.lower():
                    cabecalho_df = df
                    break
            
            if cabecalho_df is None:
                # Pegar o primeiro dataframe disponível
                cabecalho_df = list(self.dataframes.values())[0]
            
            # Perguntas sobre maior fornecedor
            if any(keyword in question.lower() for keyword in ['maior', 'fornecedor', 'montante', 'valor']):
                return self._analyze_biggest_supplier(cabecalho_df)
            
            # Outras análises podem ser adicionadas aqui
            return "Análise não implementada para este tipo de pergunta."
            
        except Exception as e:
            return f"Erro na análise direta: {str(e)}"
    
    def _analyze_biggest_supplier(self, df):
        """Analisa o fornecedor com maior montante"""
        try:
            # Verificar estrutura do dataframe
            print("Colunas disponíveis:", df.columns.tolist())
            
            # Procurar colunas relevantes
            valor_col = None
            fornecedor_col = None
            
            for col in df.columns:
                if 'valor' in col.lower() and 'nota' in col.lower():
                    valor_col = col
                if 'razao' in col.lower() or 'fornecedor' in col.lower() or 'emitente' in col.lower():
                    fornecedor_col = col
            
            if not valor_col or not fornecedor_col:
                return f"❌ Colunas necessárias não encontradas. Disponíveis: {df.columns.tolist()}"
            
            # Converter coluna de valor para numérico
            df_work = df.copy()
            df_work[valor_col] = pd.to_numeric(df_work[valor_col], errors='coerce')
            
            # Remover valores nulos
            df_work = df_work.dropna(subset=[valor_col, fornecedor_col])
            
            # Agrupar por fornecedor e somar valores
            resultado = df_work.groupby(fornecedor_col)[valor_col].sum().sort_values(ascending=False)
            
            # Obter o maior
            maior_fornecedor = resultado.index[0]
            maior_valor = resultado.iloc[0]
            
            # Preparar resposta
            response = f"""
## 📊 Análise de Fornecedores - Maior Montante

### 🥇 Resultado Principal:
**Fornecedor com maior montante total:** {maior_fornecedor}
**Valor total:** R$ {maior_valor:,.2f}

### 📋 Código Pandas Executado:
```python
# 1. Converter coluna de valor para numérico
df['{valor_col}'] = pd.to_numeric(df['{valor_col}'], errors='coerce')

# 2. Agrupar por fornecedor e somar todos os valores
resultado = df.groupby('{fornecedor_col}')['{valor_col}'].sum()

# 3. Ordenar de forma descendente
resultado_ordenado = resultado.sort_values(ascending=False)

# 4. Obter o maior fornecedor
maior_fornecedor = resultado_ordenado.index[0]
maior_valor = resultado_ordenado.iloc[0]
```

### 🏆 Top 5 Fornecedores:
"""
            
            for i, (fornecedor, valor) in enumerate(resultado.head().items()):
                response += f"{i+1}. **{fornecedor}**: R$ {valor:,.2f}\n"
            
            response += f"\n### 📈 Estatísticas:\n"
            response += f"- Total de fornecedores únicos: {len(resultado)}\n"
            response += f"- Valor total geral: R$ {resultado.sum():,.2f}\n"
            response += f"- Valor médio por fornecedor: R$ {resultado.mean():,.2f}\n"
            
            return response
            
        except Exception as e:
            return f"Erro na análise do maior fornecedor: {str(e)}"
    
    def query(self, question, agent_type="geral"):
        """Executa uma consulta usando o agente especificado ou análise direta"""
        try:
            # Primeiro tentar com o agente LangChain
            if agent_type == "geral":
                agent = self.create_general_agent()
                if agent is not None:
                    return self._query_with_agent(agent, question)
            else:
                agent = self.agents.get(agent_type)
                if agent is not None:
                    return self._query_with_agent(agent, question)
            
            # Se o agente falhar, usar análise direta
            print("⚠️ Agente não disponível, usando análise direta...")
            return self.analyze_data_directly(question)
                
        except Exception as e:
            print(f"❌ Erro na consulta: {str(e)}")
            # Fallback para análise direta
            try:
                return self.analyze_data_directly(question)
            except Exception as e2:
                return f"Erro ao processar consulta: {str(e)} | Erro fallback: {str(e2)}"
    
    def _query_with_agent(self, agent, question):
        """Executa consulta com o agente LangChain"""
        try:
            enhanced_question = f"""
PERGUNTA: {question}

INSTRUÇÕES CRÍTICAS PARA ANÁLISE DE DADOS:

🎯 OBJETIVO: Fornecer análise precisa e confiável dos dados de notas fiscais

📋 METODOLOGIA OBRIGATÓRIA:
1. Sempre verificar a estrutura dos dados primeiro (df.info(), df.head())
2. Converter colunas monetárias para numérico: pd.to_numeric(df['coluna'], errors='coerce')
3. Para análises por fornecedor: SEMPRE usar df.groupby('RAZÃO SOCIAL EMITENTE').sum()
4. Ordenar resultados: .sort_values(ascending=False)
5. Mostrar código pandas executado
6. Validar resultados com múltiplas verificações

⚠️ REGRAS CRÍTICAS:
- NUNCA pegar valores individuais de notas fiscais
- SEMPRE agrupar e somar por fornecedor
- SEMPRE converter valores para numérico antes de somar
- Responder em português brasileiro
- Mostrar top 5 resultados para verificação

💡 EXEMPLO DE CÓDIGO CORRETO:
```python
# Verificar dados
print("Estrutura dos dados:")
print(df.info())
print("Primeiras linhas:")
print(df.head())

# Converter valores
df['VALOR NOTA FISCAL'] = pd.to_numeric(df['VALOR NOTA FISCAL'], errors='coerce')

# Agrupar por fornecedor
resultado = df.groupby('RAZÃO SOCIAL EMITENTE')['VALOR NOTA FISCAL'].sum()
resultado_ordenado = resultado.sort_values(ascending=False)

# Mostrar resultado
print("Maior fornecedor:", resultado_ordenado.index[0])
print("Valor:", resultado_ordenado.iloc[0])
```

RESPONDA DE FORMA ESTRUTURADA:
1. Código executado
2. Resultado principal
3. Top 5 para verificação
4. Conclusão clara
"""
            
            response = agent.invoke({"input": enhanced_question})
            
            if isinstance(response, dict):
                result = response.get("output", response.get("result", ""))
            else:
                result = str(response)
            
            # Limpar resposta
            if result and str(result).strip():
                result_str = str(result).strip()
                
                # Remover linhas de debug
                lines = result_str.split('\n')
                clean_lines = []
                for line in lines:
                    if not any(debug_term in line.lower() for debug_term in 
                             ['debug:', 'executando', 'observation:', 'action:', 'action input:']):
                        clean_lines.append(line)
                
                result_str = '\n'.join(clean_lines).strip()
                
                # Traduzir termos em inglês
                translations = {
                    'The supplier with the highest total received value is:': 'O fornecedor com o maior valor total recebido é:',
                    'Final Answer:': 'Resposta Final:',
                    'with a total of': 'com um total de',
                    'Total value:': 'Valor total:',
                    'Supplier:': 'Fornecedor:'
                }
                
                for en_term, pt_term in translations.items():
                    result_str = result_str.replace(en_term, pt_term)
                    
                return result_str
            else:
                return "O agente não conseguiu processar a consulta adequadamente."
                
        except Exception as e:
            print(f"❌ Erro na consulta com agente: {str(e)}")
            raise e

    def get_data_summary(self):
        """Retorna um resumo dos dados carregados"""
        summary = {}
        for file_type, df in self.dataframes.items():
            summary[file_type] = {
                'linhas': len(df),
                'colunas': len(df.columns),
                'colunas_lista': df.columns.tolist(),
                'tipos': df.dtypes.to_dict(),
                'primeiras_linhas': df.head().to_dict('records'),
                'valores_nulos': df.isnull().sum().to_dict(),
                'memoria_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
        return summary

def main():
    st.set_page_config(
        page_title="Agente de Análise de Notas Fiscais - Google Gemini",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🤖 Agente Inteligente para Análise de Notas Fiscais")
    st.markdown("### Powered by Google Gemini API & LangChain")
    st.markdown("---")
    
    # Verificar API Key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        st.error("⚠️ GOOGLE_API_KEY não encontrada no arquivo .env")
        st.info("💡 Adicione sua chave da API do Google no arquivo .env")
        return
    
    # Inicializar agente
    if 'csv_agent' not in st.session_state:
        st.session_state.csv_agent = CSVAnalysisAgent(google_api_key)
    
    # Upload de arquivos
    st.header("📁 Upload de Arquivos")
    uploaded_file = st.file_uploader(
        "Faça upload do arquivo ZIP contendo os CSVs de notas fiscais",
        type=['zip'],
        help="Upload do arquivo ZIP contendo os arquivos CSV das notas fiscais"
    )
    
    if uploaded_file is not None:
        with st.spinner("🔄 Processando arquivos..."):
            try:
                temp_dir = extract_zip_file(uploaded_file)
                
                if temp_dir:
                    st.success("✅ Arquivo ZIP extraído com sucesso!")
                    
                    validator = NotaFiscalValidator()
                    csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
                    
                    if not csv_files:
                        st.error("❌ Nenhum arquivo CSV encontrado no ZIP")
                        return
                    
                    loaded_files = []
                    for csv_file in csv_files:
                        file_path = os.path.join(temp_dir, csv_file)
                        file_type = validator.identify_file_type(file_path)
                        
                        if file_type != "unknown":
                            success = st.session_state.csv_agent.load_csv_data(file_path, file_type)
                            if success:
                                loaded_files.append(f"{csv_file} ({file_type})")
                            else:
                                st.warning(f"⚠️ Falha ao carregar: {csv_file}")
                        else:
                            st.warning(f"⚠️ Tipo de arquivo não reconhecido: {csv_file}")
                    
                    if loaded_files:
                        st.success(f"✅ Arquivos carregados com sucesso:")
                        for file in loaded_files:
                            st.write(f"  • {file}")
                        
                        # Mostrar resumo dos dados
                        with st.expander("📊 Resumo Detalhado dos Dados Carregados"):
                            summary = st.session_state.csv_agent.get_data_summary()
                            for file_type, info in summary.items():
                                st.subheader(f"📄 {file_type.title()}")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("📏 Linhas", f"{info['linhas']:,}")
                                with col2:
                                    st.metric("📋 Colunas", info['colunas'])
                                with col3:
                                    st.metric("💾 Memória", f"{info['memoria_mb']} MB")
                                
                                st.write("**🏷️ Colunas disponíveis:**")
                                cols_text = ", ".join([f"`{col}`" for col in info['colunas_lista']])
                                st.markdown(cols_text)
                                
                                if st.checkbox(f"👀 Ver amostra dos dados - {file_type}", key=f"show_{file_type}"):
                                    df_display = pd.DataFrame(info['primeiras_linhas'])
                                    st.dataframe(df_display, use_container_width=True)
                                
                                if st.checkbox(f"🔍 Ver informações de qualidade - {file_type}", key=f"quality_{file_type}"):
                                    st.write("**Valores nulos por coluna:**")
                                    nulls_df = pd.DataFrame(list(info['valores_nulos'].items()), 
                                                          columns=['Coluna', 'Valores Nulos'])
                                    st.dataframe(nulls_df, use_container_width=True)
                    else:
                        st.error("❌ Nenhum arquivo foi carregado com sucesso")
                else:
                    st.error("❌ Falha ao extrair o arquivo ZIP")
                    
            except Exception as e:
                st.error(f"❌ Erro durante o processamento: {str(e)}")
    
    # Interface de consultas
    if hasattr(st.session_state, 'csv_agent') and st.session_state.csv_agent.dataframes:
        st.header("🔍 Análise Inteligente dos Dados")
        
        # Abas para diferentes tipos de análise
        tab1, tab2, tab3 = st.tabs(["💬 Perguntas Livres", "📊 Análises Rápidas", "💡 Exemplos"])
        
        with tab1:
            st.subheader("Faça sua pergunta sobre os dados")
            
            user_question = st.text_area(
                "Digite sua pergunta:",
                placeholder="Ex: Qual é o fornecedor que teve maior montante recebido?",
                height=100,
                help="Seja específico na sua pergunta para obter melhores resultados"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🚀 Executar Consulta", type="primary", use_container_width=True):
                    if user_question.strip():
                        with st.spinner("🔄 Analisando dados..."):
                            response = st.session_state.csv_agent.query(user_question)
                            
                            st.markdown("### 📋 Resultado da Análise:")
                            st.markdown(response)
                    else:
                        st.warning("⚠️ Por favor, digite uma pergunta.")
            
            with col2:
                if st.button("🧹 Limpar", use_container_width=True):
                    st.rerun()
        
        with tab2:
            st.subheader("Análises Pré-definidas")
            
            quick_analyses = {
                "🏆 Maior Fornecedor": "Qual é o fornecedor que teve maior montante recebido?",
                "📦 Maior Volume": "Qual item teve maior volume entregue em quantidade?",
                "📄 Total de Notas": "Quantas notas fiscais foram emitidas no total?",
                "💰 Valor Total": "Qual é a soma total de todos os valores das notas fiscais?",
                "🥇 Top 5 Fornecedores": "Quais são os 5 fornecedores com maior valor total?",
                "📊 Estatísticas Gerais": "Qual é a média de valor por item e quantos itens diferentes foram comprados?"
            }
            
            cols = st.columns(2)
            for i, (title, question) in enumerate(quick_analyses.items()):
                with cols[i % 2]:
                    if st.button(title, key=f"quick_{i}", use_container_width=True):
                        with st.spinner(f"🔄 Executando: {title}"):
                            response = st.session_state.csv_agent.query(question)
                            st.markdown(f"### {title}")
                            st.markdown(response)
        
        with tab3:
            st.subheader("💡 Exemplos de Perguntas")
            st.markdown("""
            **🏢 Análises por Fornecedor:**
            - Qual é o fornecedor que teve maior montante recebido?
            - Quais são os 10 principais fornecedores por valor?
            - Quantos fornecedores únicos existem nos dados?
            
            **📦 Análises de Produtos/Itens:**
            - Qual item teve maior volume entregue (em quantidade)?
            - Qual é o produto mais caro?
            - Quantos itens diferentes foram comprados?
            
            **💰 Análises Financeiras:**
            - Qual é a soma total de todos os valores das notas fiscais?
            - Qual é a média de valor por item?
            - Qual é a distribuição de valores por fornecedor?
            
            **📊 Análises Estatísticas:**
            - Quantas notas fiscais foram emitidas no total?
            - Em que período foram emitidas as notas fiscais?
            - Qual é o valor médio por nota fiscal?
            
            **🔍 Análises Específicas:**
            - Mostre detalhes sobre o fornecedor [NOME]
            - Quais produtos foram fornecidos pela empresa [NOME]?
            - Qual foi o valor total gasto com [CATEGORIA/PRODUTO]?
            """)
    
    # Sidebar com informações
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Sobre esta Aplicação")
    st.sidebar.markdown("""
    Esta aplicação utiliza inteligência artificial para analisar dados de notas fiscais:
    
    **🧠 Tecnologias:**
    - **Google Gemini 1.5 Flash** - IA generativa
    - **LangChain** - Framework para agentes
    - **Streamlit** - Interface web
    - **Pandas** - Análise de dados
    
    **⚡ Funcionalidades:**
    - Upload de arquivos ZIP
    - Análise automática de CSVs
    - Consultas em linguagem natural
    - Análises estatísticas avançadas
    """)
    
    st.sidebar.markdown("### 🔧 Status do Sistema")
    if hasattr(st.session_state, 'csv_agent') and st.session_state.csv_agent.dataframes:
        st.sidebar.success(f"✅ {len(st.session_state.csv_agent.dataframes)} arquivo(s) carregado(s)")
        for file_type, df in st.session_state.csv_agent.dataframes.items():
            st.sidebar.write(f"📄 {file_type}: {df.shape[0]:,} linhas")
    else:
        st.sidebar.info("ℹ️ Aguardando upload de arquivos")
    
    st.sidebar.markdown("### 🔗 Links Úteis")
    st.sidebar.markdown("""
    - [Google AI Studio](https://ai.google.dev/)
    - [LangChain Docs](https://python.langchain.com/)
    - [Streamlit Docs](https://docs.streamlit.io/)
    """)

if __name__ == "__main__":
    main()
