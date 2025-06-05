# No início do arquivo, substitua:
import streamlit as st
import pandas as pd
import zipfile
import os
import tempfile
from io import StringIO

# Tente este import alternativo:
try:
    from langchain_experimental.agents import create_csv_agent
except ImportError:
    from langchain.agents import create_csv_agent

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from utils_google import NotaFiscalValidator, extract_zip_file
import warnings
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import warnings
warnings.filterwarnings("ignore")

class CSVAnalysisAgent:
    def __init__(self, google_api_key=None):
        """Inicializa o agente de análise CSV com Google Gemini"""
        self.google_api_key = google_api_key
        self.agents = {}
        self.dataframes = {}
        self.file_info = {}
        
    def create_llm(self):
        """Cria uma instância do modelo Google Gemini"""
        if not self.google_api_key:
            raise ValueError("Google API Key é necessária para usar o agente")
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,  # Aumentar para mais flexibilidade
                convert_system_message_to_human=True
            )
            return llm
        except Exception as e:
            st.error(f"Erro ao criar modelo Gemini: {str(e)}")
            return None
    
    def load_csv_data(self, file_path, file_type):
        """Carrega dados CSV e cria agente específico"""
        try:
            # Lê o arquivo CSV
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # Armazena o dataframe
            self.dataframes[file_type] = df
            self.file_info[file_type] = {
                'path': file_path,
                'shape': df.shape,
                'columns': df.columns.tolist()
            }
            
            # Cria o LLM
            llm = self.create_llm()
            if llm is None:
                return False
            
            # Cria agente específico para este CSV
            agent = create_csv_agent(
                llm,
                file_path,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )
            
            self.agents[file_type] = agent
            return True
            
        except Exception as e:
            st.error(f"Erro ao carregar arquivo {file_type}: {str(e)}")
            return False
    
    def create_general_agent(self):
        """Cria um agente geral que pode acessar todos os dataframes"""
        try:
            if not self.dataframes:
                print("Erro: Nenhum dataframe carregado")
                return None
            
            llm = self.create_llm()
            if llm is None:
                print("Erro: Não foi possível criar LLM")
                return None
            
            # Coleta TODOS os caminhos de arquivos válidos usando file_info
            valid_paths = []
            for file_type, df in self.dataframes.items():
                # CORREÇÃO: usar self.file_info ao invés de getattr
                file_path = self.file_info.get(file_type, {}).get('path')
                if file_path and os.path.exists(file_path):
                    valid_paths.append(file_path)
                    print(f"Arquivo válido: {file_path}")
                else:
                    print(f"Arquivo não encontrado para {file_type}: {file_path}")
            
            if not valid_paths:
                print("Erro: Nenhum arquivo CSV válido encontrado")
                return None
            
            print(f"Criando agente com {len(valid_paths)} arquivos: {valid_paths}")
            
            # SEMPRE tenta criar com TODOS os arquivos primeiro
            try:
                # Linha ~120
                general_agent = create_csv_agent(
                    llm,
                    valid_paths,
                    verbose=True,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    allow_dangerous_code=True,
                    handle_parsing_errors=True,
                    max_iterations=20,  # Aumentar de 10 para 20
                    early_stopping_method="generate",
                    agent_executor_kwargs={
                        "handle_parsing_errors": True,
                        "max_execution_time": 300  # Aumentar de 120 para 300 segundos
                    }
                )
                
                print(f"✅ Agente criado com sucesso para {len(valid_paths)} arquivos")
                return general_agent
                
            except Exception as e:
                print(f"❌ Erro ao criar agente com múltiplos arquivos: {str(e)}")
                
                # Se falhar com múltiplos arquivos, tenta abordagem pandas
                if len(valid_paths) > 1:
                    try:
                        print("🔄 Tentando abordagem pandas com todos os dataframes...")
                        
                        from langchain.agents import create_pandas_dataframe_agent
                        
                        # Converte todos os dataframes em uma lista
                        all_dfs = list(self.dataframes.values())
                        df_names = list(self.dataframes.keys())
                        
                        general_agent = create_pandas_dataframe_agent(
                            llm,
                            all_dfs,
                            verbose=True,
                            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            allow_dangerous_code=True,
                            handle_parsing_errors=True,
                            prefix=f"Você tem acesso aos seguintes dataframes: {', '.join(df_names)}. Use df_0 para {df_names[0]}, df_1 para {df_names[1]}, etc."
                        )
                        
                        print(f"✅ Agente pandas criado com {len(all_dfs)} dataframes")
                        return general_agent
                        
                    except Exception as e2:
                        print(f"❌ Erro na abordagem pandas: {str(e2)}")
                    
                # Último recurso: apenas um arquivo
                try:
                    print(f"⚠️ Fallback: criando agente apenas para {valid_paths[0]}")
                    general_agent = create_csv_agent(
                        llm,
                        valid_paths[0],
                        verbose=True,
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        allow_dangerous_code=True,
                        handle_parsing_errors=True
                    )
                    print(f"⚠️ Agente criado apenas para: {valid_paths[0]}")
                    return general_agent
                except Exception as e3:
                    print(f"❌ Erro no fallback: {str(e3)}")
                    return None
                
        except Exception as e:
            print(f"❌ Erro geral: {str(e)}")
            st.error(f"Erro ao criar agente geral: {str(e)}")
            return None
    
    def query(self, question, agent_type="geral"):
        """Executa uma consulta usando o agente especificado"""
        try:
            # Debug apenas no console, não na resposta
            # print(f"\n=== DEBUG: Iniciando consulta ===")
            # print(f"Pergunta: {question}")
            # print(f"Dados disponíveis: {list(self.dataframes.keys())}")
            
            if agent_type == "geral":
                agent = self.create_general_agent()
                if agent is None:
                    return "Erro: Não foi possível criar o agente geral."
            else:
                agent = self.agents.get(agent_type)
                if agent is None:
                    return f"Erro: Agente para {agent_type} não encontrado."
            
            # Instrução MUITO específica para o agente - SEMPRE EM PORTUGUÊS
            enhanced_question = f"""
            Você é um especialista em análise de dados financeiros de notas fiscais.
            
            DADOS DISPONÍVEIS: {', '.join(self.dataframes.keys())}
            PERGUNTA: {question}
            
            METODOLOGIA OBRIGATÓRIA (baseada no ChatGPT):
            1. Carregue o arquivo de cabeçalho das notas fiscais
            2. Converta a coluna 'VALOR NOTA FISCAL' para float
            3. Agrupe por 'RAZÃO SOCIAL EMITENTE' (nome do fornecedor)
            4. Some os valores para obter total por fornecedor
            5. Ordene de forma descendente
            6. Identifique o fornecedor com maior valor
            
            CÓDIGO PANDAS OBRIGATÓRIO:
            ```python
            import pandas as pd
            
            # 1. Verificar estrutura dos dados
            print("Colunas disponíveis:")
            print(df.columns.tolist())
            print("\nPrimeiras linhas:")
            print(df.head())
            
            # 2. Converter coluna de valor para numérico
            df['VALOR NOTA FISCAL'] = pd.to_numeric(df['VALOR NOTA FISCAL'], errors='coerce')
            
            # 3. Agrupar por fornecedor e somar valores
            resultado = df.groupby('RAZÃO SOCIAL EMITENTE')['VALOR NOTA FISCAL'].sum()
            
            # 4. Ordenar de forma descendente
            resultado_ordenado = resultado.sort_values(ascending=False)
            
            # 5. Obter o maior
            maior_fornecedor = resultado_ordenado.index[0]
            maior_valor = resultado_ordenado.iloc[0]
            
            print(f"\nFornecedor com maior montante: {maior_fornecedor}")
            print(f"Valor total: R$ {maior_valor:,.2f}")
            
            # 6. Mostrar top 5 para verificação
            print("\nTop 5 fornecedores:")
            for i, (fornecedor, valor) in enumerate(resultado_ordenado.head().items()):
                print(f"{i+1}. {fornecedor}: R$ {valor:,.2f}")
            ```
            
            INSTRUÇÕES CRÍTICAS:
            - Use EXATAMENTE os nomes das colunas: 'VALOR NOTA FISCAL' e 'RAZÃO SOCIAL EMITENTE'
            - SEMPRE converta valores para numérico antes de somar
            - SEMPRE mostre o código executado
            - SEMPRE responda em português brasileiro
            - Use vírgula como separador decimal (R$ 1.234,56)
            - NUNCA invente dados - use apenas o que está nos arquivos
            - Se der erro, verifique os nomes das colunas com df.columns
            """
            
            INSTRUÇÕES CRÍTICAS:
            1. SEMPRE use pandas para análise: df.groupby(), df.sum(), df.max()
            2. Para encontrar maior valor: use df.groupby('fornecedor').sum().idxmax()
            3. SEMPRE verifique os dados com df.head(), df.info(), df.describe()
            4. Para valores monetários: use format(valor, ',.2f').replace(',', 'X').replace('.', ',').replace('X', '.')
            5. SEMPRE responda em português brasileiro
            6. Use dados do arquivo 'cabecalho' para totais por fornecedor
            7. NUNCA invente dados - apenas use o que está nos arquivos
            8. Mostre o código pandas executado
            9. Verifique múltiplas vezes os cálculos
            10. Se houver dúvida, reanalise os dados
            
            EXEMPLO DE CÓDIGO OBRIGATÓRIO:
            ```python
            # Verificar dados
            print(df_cabecalho.columns)
            print(df_cabecalho.head())
            
            # Agrupar por fornecedor e somar valores
            resultado = df_cabecalho.groupby('nome_fornecedor')['valor_total'].sum()
            maior_fornecedor = resultado.idxmax()
            maior_valor = resultado.max()
            
            print(f"Fornecedor: {maior_fornecedor}")
            print(f"Valor: R$ {maior_valor:,.2f}")
            ```
            
            # print("Executando consulta...")  # Debug apenas no console
            response = agent.invoke({"input": enhanced_question})
            # print(f"Tipo da resposta: {type(response)}")  # Debug apenas no console
            
            if isinstance(response, dict):
                result = response.get("output", response.get("result", ""))
            else:
                result = str(response)
            
            # print(f"Resultado: {result[:200]}...")  # Debug apenas no console
            
            # Pós-processamento para garantir português e limpar debug
            if result and str(result).strip():
                result_str = str(result).strip()
                
                # Remove linhas de debug que possam ter vazado
                lines = result_str.split('\n')
                clean_lines = []
                for line in lines:
                    # Remove linhas que contêm informações de debug
                    if not any(debug_term in line.lower() for debug_term in 
                             ['debug:', 'tipo da resposta', 'executando', '===', 'print(']):
                        clean_lines.append(line)
                
                result_str = '\n'.join(clean_lines).strip()
                
                # Se a resposta ainda estiver em inglês, força tradução básica
                if any(word in result_str.lower() for word in ['the supplier', 'with the highest', 'total received', 'final answer']):
                    result_str = result_str.replace('The supplier with the highest total received value is:', 'O fornecedor com o maior valor total recebido é:')
                    result_str = result_str.replace('Final Answer:', 'Resposta Final:')
                    result_str = result_str.replace('with a total of', 'com um total de')
                    
                return result_str
            else:
                return "O agente não conseguiu processar a consulta. Verifique se os dados foram carregados corretamente."
                
        except Exception as e:
            # print(f"Erro geral: {str(e)}")  # Debug apenas no console
            return f"Erro ao processar consulta: {str(e)}"
    
    def get_data_summary(self):
        """Retorna um resumo dos dados carregados"""
        summary = {}
        for file_type, df in self.dataframes.items():
            summary[file_type] = {
                'linhas': len(df),
                'colunas': len(df.columns),
                'colunas_lista': df.columns.tolist(),
                'tipos': df.dtypes.to_dict(),
                'primeiras_linhas': df.head().to_dict('records')
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
    
    # Carrega a API key do arquivo .env
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        st.error("⚠️ GOOGLE_API_KEY não encontrada no arquivo .env")
        return
    
    # Inicializa o agente
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
        with st.spinner("Processando arquivos..."):
            # Extrai o arquivo ZIP
            temp_dir = extract_zip_file(uploaded_file)
            
            if temp_dir:
                st.success("✅ Arquivo ZIP extraído com sucesso!")
                
                # Processa os arquivos CSV
                validator = NotaFiscalValidator()
                csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
                
                loaded_files = []
                for csv_file in csv_files:
                    file_path = os.path.join(temp_dir, csv_file)
                    file_type = validator.identify_file_type(file_path)
                    
                    if file_type != "unknown":
                        success = st.session_state.csv_agent.load_csv_data(file_path, file_type)
                        if success:
                            loaded_files.append(f"{csv_file} ({file_type})")
                
                if loaded_files:
                    st.success(f"✅ Arquivos carregados: {', '.join(loaded_files)}")
                    
                    # Mostra resumo dos dados
                    with st.expander("📊 Resumo dos Dados Carregados"):
                        summary = st.session_state.csv_agent.get_data_summary()
                        for file_type, info in summary.items():
                            st.subheader(f"📄 {file_type.title()}")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Linhas", info['linhas'])
                                st.metric("Colunas", info['colunas'])
                            with col2:
                                st.write("**Colunas:**")
                                st.write(", ".join(info['colunas_lista']))
                            
                            # Mostra primeiras linhas
                            if st.checkbox(f"Ver primeiras linhas - {file_type}", key=f"show_{file_type}"):
                                df_display = pd.DataFrame(info['primeiras_linhas'])
                                st.dataframe(df_display, use_container_width=True)
    
    # Interface de consultas
    if hasattr(st.session_state, 'csv_agent') and st.session_state.csv_agent.dataframes:
        st.header("🔍 Faça suas perguntas")
        
        # Exemplos de perguntas
        with st.expander("💡 Exemplos de Perguntas"):
            st.markdown("""
            - Qual é o fornecedor que teve maior montante recebido?
            - Qual item teve maior volume entregue (em quantidade)?
            - Quantas notas fiscais foram emitidas no total?
            - Qual é a soma total de todos os valores das notas fiscais?
            - Quais são os 5 fornecedores com maior valor total?
            - Qual é a média de valor por item?
            - Quantos itens diferentes foram comprados?
            - Qual é o produto mais caro?
            - Em que período foram emitidas as notas fiscais?
            - Qual é a distribuição de valores por fornecedor?
            """)
        
        # Campo de pergunta
        user_question = st.text_area(
            "Digite sua pergunta sobre os dados:",
            placeholder="Ex: Qual é o fornecedor que teve maior montante recebido?",
            height=100
        )
        
        if st.button("🚀 Executar Consulta", type="primary"):
            if user_question.strip():
                with st.spinner("Processando consulta..."):
                    response = st.session_state.csv_agent.query(user_question)
                    
                    # Exibe apenas a resposta limpa
                    st.markdown("### 📋 Resposta:")
                    
                    # Cria um expander para detalhes técnicos (opcional)
                    with st.expander("🔧 Detalhes Técnicos (clique para ver)", expanded=False):
                        st.write("Tipo da resposta:", type(response))
                        st.write("Conteúdo bruto:", response)
                    
                    # Exibe a resposta principal de forma limpa
                    if response and str(response).strip():
                        # Remove possíveis prefixos de debug
                        clean_response = str(response).strip()
                        
                        # Remove linhas que começam com "Tipo da resposta" ou similar
                        lines = clean_response.split('\n')
                        clean_lines = []
                        for line in lines:
                            if not any(debug_prefix in line.lower() for debug_prefix in 
                                     ['tipo da resposta', 'conteúdo da resposta', 'debug:', '===']):
                                clean_lines.append(line)
                        
                        final_response = '\n'.join(clean_lines).strip()
                        
                        # Exibe a resposta final
                        st.markdown(final_response)
                    else:
                        st.error("Não foi possível obter uma resposta válida.")
            else:
                st.warning("⚠️ Por favor, digite uma pergunta.")
    
    # Informações adicionais
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Sobre")
    st.sidebar.markdown("""
    Esta aplicação utiliza:
    - **Google Gemini API** para processamento de linguagem natural
    - **LangChain** para criação de agentes inteligentes
    - **Streamlit** para interface web
    - **Pandas** para manipulação de dados
    """)
    
    st.sidebar.markdown("### 🔗 Links Úteis")
    st.sidebar.markdown("""
    - [Google AI Studio](https://ai.google.dev/)
    - [LangChain Documentation](https://python.langchain.com/)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    """)

if __name__ == "__main__":
    main()
    
    # 🚨 Problema Confirmado: Aplicação Ainda Dá Respostas Incorretas
    
    # Vejo que mesmo após as tentativas de melhorias, a aplicação ainda está fornecendo dados incorretos:
    
    # - **Resposta da Aplicação:** EDITORA FTD S.A. com R$ 6.712,16
    # - **Resposta Correta:** CHEMYUNION LTDA com R$ 1.292.418,75
    
    # Isso indica que as mudanças ainda não foram aplicadas ou não são suficientes.
    
    # 🔧 Solução Definitiva: Aplicar Mudanças Mais Robustas
    
    # Vamos implementar melhorias mais específicas no código:
    
    # ### **1. Verificar se as Mudanças Foram Aplicadas**
    
    # Primeiro, confirme se o arquivo local tem essas configurações:
    
    # Linha ~43 - Temperature
    temperature=0.5,  # AUMENTAR AINDA MAIS para 0.5
    
    # Linha ~126 - Max Iterations  
    max_iterations=30,  # AUMENTAR para 30
    
    # Linha ~130 - Max Execution Time
    "max_execution_time": 600  # AUMENTAR para 600 segundos