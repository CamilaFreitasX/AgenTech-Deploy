import streamlit as st
import pandas as pd
import zipfile
import os
import tempfile
from io import StringIO

# Tente este import alternativo:
try:
    from langchain_experimental.agents import create_csv_agent
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent 
except ImportError:
    from langchain.agents import create_csv_agent
    # Para versões mais antigas de langchain, create_pandas_dataframe_agent pode estar em langchain.agents
    try:
        from langchain.agents import create_pandas_dataframe_agent
    except ImportError:
        st.error("Não foi possível importar 'create_pandas_dataframe_agent'. Verifique a instalação do LangChain.")
        # Permitir que o app continue se este agente específico não for crucial ou usado.
        pass


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType # AgentType ainda é usado
from utils_google import NotaFiscalValidator, extract_zip_file # Supondo que utils_google.py exista e funcione
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
        self.cabecalho_file_type_name = "cabecalho" 

    def create_llm(self):
        """Cria uma instância do modelo Google Gemini"""
        if not self.google_api_key:
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            if not self.google_api_key:
                raise ValueError("Google API Key é necessária para usar o agente e não foi encontrada.")
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.google_api_key,
                temperature=0.5,
                convert_system_message_to_human=True,
            )
            return llm
        except Exception as e:
            st.error(f"Erro ao criar modelo Gemini: {str(e)}")
            return None
    
    def load_csv_data(self, file_path, file_type):
        """Carrega dados CSV."""
        try:
            try:
                df = pd.read_csv(file_path, encoding='utf-8', sep=None, engine='python', on_bad_lines='warn')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1', sep=None, engine='python', on_bad_lines='warn')
            except Exception as e_read: 
                st.error(f"Erro ao ler CSV {os.path.basename(file_path)}: {e_read}. Tentando com delimitador ';'...")
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', sep=';', on_bad_lines='warn')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='latin1', sep=';', on_bad_lines='warn')
                except Exception as e_read_semi:
                     st.error(f"Não foi possível ler o arquivo {os.path.basename(file_path)} mesmo com sep=';': {e_read_semi}")
                     return False

            self.dataframes[file_type] = df
            self.file_info[file_type] = {
                'path': file_path, 
                'shape': df.shape,
                'columns': df.columns.tolist()
            }
            return True
            
        except Exception as e:
            st.error(f"Erro geral ao carregar arquivo {file_type} ({os.path.basename(file_path)}): {str(e)}")
            return False
    
    def create_general_agent(self):
        """Cria um agente geral que pode acessar todos os dataframes carregados."""
        try:
            if not self.dataframes:
                print("Erro: Nenhum dataframe carregado.")
                st.error("Nenhum dataframe carregado para criar o agente geral.")
                return None
            
            llm = self.create_llm()
            if llm is None:
                print("Erro: Não foi possível criar LLM para o agente geral.")
                st.error("Não foi possível criar o modelo de linguagem para o agente geral.")
                return None
            
            ordered_df_tuples = []
            for file_type_key in self.dataframes.keys(): 
                df_instance = self.dataframes.get(file_type_key)
                if isinstance(df_instance, pd.DataFrame):
                     ordered_df_tuples.append((file_type_key, df_instance))
                else:
                    print(f"Aviso: '{file_type_key}' não é um DataFrame válido. Será ignorado.")

            if not ordered_df_tuples:
                print("Erro: Nenhum dataframe válido para o agente.")
                st.error("Nenhum dataframe válido encontrado para criar o agente.")
                return None

            list_of_dfs = [df_tuple[1] for df_tuple in ordered_df_tuples]
            list_of_df_names = [df_tuple[0] for df_tuple in ordered_df_tuples]
            
            cabecalho_df_index = -1
            st.session_state.pop('cabecalho_df_variable_name', None) 
            try:
                cabecalho_df_index = list_of_df_names.index(self.cabecalho_file_type_name)
                st.session_state.cabecalho_df_variable_name = f"df_{cabecalho_df_index}"
                print(f"Índice do DataFrame de cabeçalho ('{self.cabecalho_file_type_name}'): {cabecalho_df_index} (será {st.session_state.cabecalho_df_variable_name})")
            except ValueError:
                print(f"Aviso: DataFrame do tipo '{self.cabecalho_file_type_name}' não encontrado na lista: {list_of_df_names}")

            # Estratégia Principal: create_pandas_dataframe_agent se houver DataFrames
            if list_of_dfs:
                try:
                    print(f"🔄 Tentando criar agente pandas com {len(list_of_dfs)} dataframes: {list_of_df_names}")
                    
                    prefix_parts = [
                        "Você é um agente de análise de dados altamente competente, especializado em notas fiscais brasileiras.",
                        f"Você tem acesso a {len(list_of_dfs)} dataframes pandas nomeados df_0, df_1, ...:",
                    ]
                    for i, name in enumerate(list_of_df_names):
                        cols_list = self.dataframes[name].columns.to_list() if isinstance(self.dataframes.get(name), pd.DataFrame) else ["Colunas não disponíveis"]
                        prefix_parts.append(f"- df_{i}: (Tipo Original: '{name}'). Colunas: {cols_list}")

                    if cabecalho_df_index != -1:
                        prefix_parts.append(f"IMPORTANTE: O dataframe 'df_{cabecalho_df_index}' (tipo '{self.cabecalho_file_type_name}') é o principal para análise de totais por fornecedor, pois contém os cabeçalhos das notas fiscais. Use-o para essa finalidade.")
                    else:
                        prefix_parts.append(f"IMPORTANTE: O dataframe do tipo '{self.cabecalho_file_type_name}' é o principal para análise de totais por fornecedor. Identifique qual df_X (df_0, df_1, etc.) corresponde a este tipo e use-o para essa finalidade.")
                    
                    prefix_message = "\n".join(prefix_parts)
                    input_for_agent = list_of_dfs[0] if len(list_of_dfs) == 1 else list_of_dfs

                    general_agent = create_pandas_dataframe_agent(
                        llm,
                        input_for_agent, 
                        verbose=True,
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        allow_dangerous_code=True,
                        prefix=prefix_message,
                        # Args do AgentExecutor passados diretamente:
                        handle_parsing_errors=" агентом будет предпринята попытка исправить ошибку синтаксического анализа ",
                        max_iterations=15,
                        early_stopping_method="generate",
                        max_execution_time=600 
                    )
                    print(f"✅ Agente pandas criado com sucesso para {len(list_of_dfs)} dataframes.")
                    st.session_state.current_agent_type = "pandas_multi"
                    return general_agent
                
                except Exception as e_pandas:
                    print(f"❌ Erro ao criar agente pandas: {str(e_pandas)}")
                    st.warning(f"Falha ao criar agente pandas: {e_pandas}. Tentando fallback para agente CSV (se houver caminhos válidos)...")
                    st.session_state.pop('cabecalho_df_variable_name', None) 

            # Fallback para create_csv_agent (usando caminhos)
            valid_paths = [self.file_info.get(ft, {}).get('path') for ft_name in list_of_df_names 
                           if self.file_info.get(ft_name, {}).get('path') and os.path.exists(self.file_info.get(ft_name, {}).get('path'))]


            if not valid_paths:
                st.error("Nenhum caminho de arquivo CSV válido encontrado para o agente de fallback.")
                return None
            
            agent_path_input = valid_paths[0] if len(valid_paths) == 1 else valid_paths
            print(f"Tentando criar agente CSV com {len(valid_paths)} arquivos: {agent_path_input}")
            try:
                general_agent = create_csv_agent(
                    llm,
                    agent_path_input, 
                    verbose=True,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    allow_dangerous_code=True,
                    # Args do AgentExecutor passados diretamente:
                    handle_parsing_errors=True, 
                    max_iterations=30,
                    early_stopping_method="generate",
                    max_execution_time=600 
                )
                print(f"✅ Agente CSV (path-based) criado com sucesso para {len(valid_paths)} arquivos.")
                st.session_state.current_agent_type = "csv_multi_path" if isinstance(agent_path_input, list) else "csv_single_path"
                
                if st.session_state.current_agent_type == "csv_single_path":
                    first_file_type_path_fallback = None # Renomeada para evitar conflito de escopo
                    for f_type_fallback, info_fallback in self.file_info.items():
                        if info_fallback['path'] == agent_path_input: 
                            first_file_type_path_fallback = f_type_fallback
                            break
                    if first_file_type_path_fallback == self.cabecalho_file_type_name:
                        st.session_state.cabecalho_df_variable_name = "df" 
                    else:
                        st.session_state.pop('cabecalho_df_variable_name', None)
                return general_agent
            except Exception as e_csv:
                print(f"❌ Erro ao criar agente CSV (path-based): {str(e_csv)}")
                st.error(f"Falha crítica ao criar qualquer tipo de agente: {e_csv}") 
                return None

        except Exception as e_general:
            print(f"❌ Erro geral não capturado anteriormente ao criar agente: {str(e_general)}")
            st.error(f"Erro geral e fatal ao criar agente de análise: {str(e_general)}")
            return None

    def query(self, question):
        """Executa uma consulta usando o agente geral."""
        try:
            agent = self.create_general_agent() 
            if agent is None:
                return "Erro: Não foi possível inicializar o agente de análise para a consulta."

            target_df_variable_for_prompt = "df" 
            identified_cabecalho_type_for_prompt = f"'{self.cabecalho_file_type_name}'"
            
            current_agent_context = st.session_state.get("current_agent_type")
            cabecalho_df_var_from_session = st.session_state.get("cabecalho_df_variable_name")

            if current_agent_context == "pandas_multi" and cabecalho_df_var_from_session:
                target_df_variable_for_prompt = cabecalho_df_var_from_session
                print(f"Query: Usando '{target_df_variable_for_prompt}' para código pandas (agente pandas_multi).")
            elif current_agent_context in ["csv_single_path", "csv_single_fallback"] and cabecalho_df_var_from_session == "df": # "csv_single_fallback" foi removido antes
                target_df_variable_for_prompt = "df"
                print(f"Query: Usando 'df' para código pandas (agente CSV único sobre cabeçalho).")
            elif current_agent_context == "csv_multi_path":
                print(f"Query: Agente csv_multi_path. O agente precisará identificar e carregar o CSV '{self.cabecalho_file_type_name}'.")
                identified_cabecalho_type_for_prompt = f"o arquivo CSV correspondente a '{self.cabecalho_file_type_name}'"
            else: 
                 print(f"AVISO: Contexto do agente é '{current_agent_context}'. O nome da variável do dataframe de cabeçalho não foi determinado explicitamente para o prompt. O Agente precisará inferir qual dataframe usar para '{self.cabecalho_file_type_name}'.")
                 identified_cabecalho_type_for_prompt = f"o dataframe/arquivo correspondente a '{self.cabecalho_file_type_name}'"


            coluna_fornecedor = 'RAZÃO SOCIAL EMITENTE'
            coluna_valor = 'VALOR NOTA FISCAL'

            pandas_code_block = f"""
```python
import pandas as pd

# O agente deve usar o dataframe que representa os dados de {identified_cabecalho_type_for_prompt}.
# No contexto de um agente pandas_multi, esta variável já foi definida (ex: df_0, df_1), use '{target_df_variable_for_prompt}'.
# No contexto de um agente CSV (path-based), carregue o arquivo CSV de cabeçalho em um dataframe (ex: {target_df_variable_for_prompt} = pd.read_csv('path_do_cabecalho.csv')).
# O código abaixo assume que '{target_df_variable_for_prompt}' é o dataframe de cabeçalho já disponível ou carregado.

print(f"Iniciando análise no dataframe '{target_df_variable_for_prompt}' (espera-se que seja {identified_cabecalho_type_for_prompt}).")
try:
    # Esta parte é crucial: o agente deve assegurar que '{target_df_variable_for_prompt}' é o dataframe correto.
    # Se o agente precisar carregar um CSV, ele deve fazer isso aqui.
    # Exemplo de lógica que o agente pode precisar executar internamente (não para copiar literalmente):
    # if not isinstance({target_df_variable_for_prompt}, pd.DataFrame) and isinstance({target_df_variable_for_prompt}, str) and ".csv" in {target_df_variable_for_prompt}:
    #     # Supondo que target_df_variable_for_prompt contém o caminho para o CSV de cabeçalho
    #     actual_df_for_analysis = pd.read_csv({target_df_variable_for_prompt}) 
    # else:
    #     actual_df_for_analysis = {target_df_variable_for_prompt} # Já é um DataFrame

    # Para o código abaixo, vamos assumir que 'actual_df_for_analysis' é o nome do dataframe que o agente usará.
    # Por simplicidade no template, continuaremos usando '{target_df_variable_for_prompt}', mas o agente deve entender este mapeamento.

    print("Colunas disponíveis em '{target_df_variable_for_prompt}':", {target_df_variable_for_prompt}.columns.tolist())
    print("\\nPrimeiras 5 linhas de '{target_df_variable_for_prompt}':\\n", {target_df_variable_for_prompt}.head())

    if '{coluna_valor}' not in {target_df_variable_for_prompt}.columns:
        raise ValueError("Coluna de valor '{coluna_valor}' NÃO ENCONTRADA no dataframe '{target_df_variable_for_prompt}'. Verifique o nome da coluna e o dataframe usado.")
    if '{coluna_fornecedor}' not in {target_df_variable_for_prompt}.columns:
        raise ValueError("Coluna de fornecedor '{coluna_fornecedor}' NÃO ENCONTRADA no dataframe '{target_df_variable_for_prompt}'. Verifique o nome da coluna e o dataframe usado.")

    df_analysis = {target_df_variable_for_prompt}.copy() 
    if df_analysis['{coluna_valor}'].dtype == 'object':
        df_analysis.loc[:, '{coluna_valor}'] = df_analysis['{coluna_valor}'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    
    df_analysis.loc[:, '{coluna_valor}'] = pd.to_numeric(df_analysis['{coluna_valor}'], errors='coerce')
    
    print(f"Valores ausentes em '{coluna_valor}' após conversão numérica: {{df_analysis['{coluna_valor}'].isnull().sum()}}")
    df_analysis.dropna(subset=['{coluna_valor}', '{coluna_fornecedor}'], inplace=True) 

    if df_analysis.empty:
        raise ValueError("O DataFrame ficou vazio após limpeza de dados (NaN em valor ou fornecedor). Não é possível prosseguir.")

    print(f"Agrupando por '{coluna_fornecedor}' e somando '{coluna_valor}'. Total de linhas para agrupar: {{len(df_analysis)}}")
    resultado = df_analysis.groupby('{coluna_fornecedor}')['{coluna_valor}'].sum()

    resultado_ordenado = resultado.sort_values(ascending=False)

    if not resultado_ordenado.empty:
        maior_fornecedor = resultado_ordenado.index[0]
        maior_valor = resultado_ordenado.iloc[0]
        
        # Formatação para o padrão monetário brasileiro R$ xxx.xxx,xx
        # Usando separador de milhar '.' e decimal ','
        maior_valor_formatado = f"R$ {{:_.2f}}".format(maior_valor).replace('.', '#TEMP#').replace(',', '.').replace('#TEMP#', ',').replace('_', '.')
        # Correção: O de cima pode não funcionar bem com o _ para .
        # Melhor: formatar para string, depois substituir.
        # Ex: 1292418.75 -> "1.292.418,75"
        s_valor = f"{{maior_valor:.2f}}" # "1292418.75"
        partes = s_valor.split('.')
        inteiro = partes[0]
        decimal = partes[1] if len(partes) > 1 else "00"
        inteiro_formatado = ""
        if len(inteiro) > 3:
            for i, digito in enumerate(reversed(inteiro)):
                if i > 0 and i % 3 == 0:
                    inteiro_formatado = "." + inteiro_formatado
                inteiro_formatado = digito + inteiro_formatado
        else:
            inteiro_formatado = inteiro
        maior_valor_formatado = f"R$ {{inteiro_formatado}},{{decimal}}"


        print(f"\\nFORNECEDOR COM MAIOR MONTANTE TOTAL CONSOLIDADO: {{maior_fornecedor}}")
        print(f"VALOR TOTAL CONSOLIDADO: {{maior_valor_formatado}}")

        print("\\nTOP 5 FORNECEDORES (VALORES TOTAIS CONSOLIDADOS):")
        for i, (fornecedor_item, valor_item) in enumerate(resultado_ordenado.head().items()):
            s_valor_item = f"{{valor_item:.2f}}"
            partes_item = s_valor_item.split('.')
            inteiro_item = partes_item[0]
            decimal_item = partes_item[1] if len(partes_item) > 1 else "00"
            inteiro_fmt_item = ""
            if len(inteiro_item) > 3:
                 for j, digito_item in enumerate(reversed(inteiro_item)):
                    if j > 0 and j % 3 == 0:
                        inteiro_fmt_item = "." + inteiro_fmt_item
                    inteiro_fmt_item = digito_item + inteiro_fmt_item
            else:
                inteiro_fmt_item = inteiro_item
            valor_fmt_item = f"R$ {{inteiro_fmt_item}},{{decimal_item}}"
            print(f"{{i+1}}. {{fornecedor_item}}: {{valor_fmt_item}}")
    else:
        print("Não foi possível calcular os resultados. O dataframe agrupado está vazio (nenhum fornecedor encontrado ou todos os valores eram inválidos).")
        
except Exception as e_pandas_code:
    print(f"ERRO AO EXECUTAR O CÓDIGO PANDAS DENTRO DO AGENTE: {{str(e_pandas_code)}}")
    print("Verifique se o dataframe '{target_df_variable_for_prompt}' foi carregado/identificado corretamente pelo agente e se as colunas '{coluna_fornecedor}' e '{coluna_valor}' existem e são adequadas para a análise.")

```"""
            available_data_message = f"DADOS DISPONÍVEIS NO SISTEMA: {', '.join(self.dataframes.keys()) if self.dataframes else 'Nenhum dataframe carregado diretamente no sistema Python. Agente pode precisar carregar de caminhos.'}"
            
            guidance_on_df_usage = (
                f"Você DEVE USAR o dataframe '{target_df_variable_for_prompt}' (que o sistema identificou como correspondendo a {identified_cabecalho_type_for_prompt}) para esta análise, pois ele contém os cabeçalhos das notas fiscais."
                if cabecalho_df_var_from_session or (current_agent_context in ["csv_single_path"] and identified_cabecalho_type_for_prompt == f"'{self.cabecalho_file_type_name}'") # csv_single_fallback removido
                else f"Você deve priorizar o uso dos dados do arquivo/dataframe do tipo '{self.cabecalho_file_type_name}'. Se múltiplos dataframes (df_0, df_1, ...) ou arquivos CSV estiverem disponíveis, identifique qual deles corresponde a '{self.cabecalho_file_type_name}' (o prefixo do agente, se aplicável, já deu essa informação) e use-o para a análise de totais por fornecedor. O código pandas fornecido usa '{target_df_variable_for_prompt}' como placeholder para este dataframe de cabeçalho."
            )

            enhanced_question = f"""
OBJETIVO PRINCIPAL: Identificar o fornecedor com o MAIOR VALOR TOTAL CONSOLIDADO de notas fiscais.

{available_data_message}
{guidance_on_df_usage}

PERGUNTA ORIGINAL DO USUÁRIO: {question}

⚠️ ATENÇÃO CRÍTICA E OBRIGATÓRIA - LEIA COM ATENÇÃO MÁXIMA:
Para determinar o "maior" fornecedor, NÃO olhe para valores de notas fiscais individuais.
VOCÊ DEVE, OBRIGATORIAMENTE, SEGUIR ESTES PASSOS:
1. AGRUPAR OS DADOS POR FORNECEDOR (usando a coluna '{coluna_fornecedor}').
2. SOMAR (usar a função `.sum()`) TODOS os valores de notas fiscais (da coluna '{coluna_valor}') para CADA fornecedor individualmente.
Isto resultará no MONTANTE TOTAL CONSOLIDADO para cada fornecedor. A análise é sobre este valor consolidado.

METODOLOGIA OBRIGATÓRIA PARA RESPONDER CORRETAMENTE:
1. IDENTIFICAÇÃO DO DATAFRAME: Certifique-se de que está usando o DataFrame correto que contém os dados de cabeçalho das notas fiscais (referido como '{target_df_variable_for_prompt}' no código abaixo, que deve corresponder a {identified_cabecalho_type_for_prompt}). Se for um agente baseado em CSV, carregue o arquivo CSV de cabeçalho apropriado.
2. COLUNAS CRUCIAIS: As colunas são '{coluna_fornecedor}' (nome do fornecedor) e '{coluna_valor}' (valor da nota).
3. CONVERSÃO DE VALOR (ESSENCIAL!): ANTES DE QUALQUER CÁLCULO, a coluna '{coluna_valor}' DEVE ser convertida para um tipo numérico (float). É comum ela estar como string (ex: "1.234,56" ou "6712.16"). O código pandas fornecido abaixo já inclui a lógica para converter strings no formato "X.XXX,YY" para o numérico XXXX.YY (removendo '.' e trocando ',' por '.'). Valores que não puderem ser convertidos devem se tornar NaN e as linhas correspondentes (ou pelo menos com valor NaN) devem ser removidas antes da agregação.
4. AGRUPAMENTO: Agrupe os dados pela coluna '{coluna_fornecedor}'.
5. SOMA CONSOLIDADA: SOME (use `.sum()`) TODOS os valores da coluna '{coluna_valor}' para CADA fornecedor.
6. ORDENAÇÃO: Ordene os fornecedores de forma descendente pelo valor total CONSOLIDADO.
7. RESULTADO: O fornecedor no topo desta lista ordenada é a resposta.

❌ EXEMPLO CLARÍSSIMO DO ERRO GRAVE A SER EVITADO (NÃO FAÇA ISSO DE FORMA ALGUMA):
- ERRADO: Olhar uma única nota fiscal da "EDITORA FTD S.A." no valor de R$ 6.712,16 e, por ser um valor alto individual, concluir que este é o resultado. ISTO ESTÁ COMPLETAMENTE ERRADO.
- CORRETO: Somar TODAS as notas fiscais da "EDITORA FTD S.A." (ex: o total consolidado dela pode ser R$ 292.486,11) e comparar esta SOMA CONSOLIDADA com a SOMA CONSOLIDADA de TODAS as notas de outros fornecedores.
- EXEMPLO DE RESULTADO CORRETO ESPERADO (valores meramente ilustrativos para o conceito): "O fornecedor CHEMYUNION LTDA teve o maior montante total consolidado, com R$ 1.292.418,75."

➡️ CÓDIGO PANDAS OBRIGATÓRIO - Execute a lógica descrita abaixo. Adapte '{target_df_variable_for_prompt}' para o nome correto do dataframe de cabeçalho se ele for diferente (ex: df_0, df_1, ou o resultado de pd.read_csv('caminho/para/cabecalho.csv')). O agente é responsável por garantir que '{target_df_variable_for_prompt}' no código abaixo seja o dataframe correto de cabeçalho.
{pandas_code_block}

FORMATO DA RESPOSTA FINAL (em português brasileiro):
- "O fornecedor com o maior montante total consolidado é: [Nome do Fornecedor]."
- "Valor total consolidado: [Valor Formatado no padrão R$ xxx.xxx,xx]."
- Inclua uma seção "Código Pandas Chave Executado e Observações:" onde você mostra o trecho principal do código pandas que você efetivamente usou para o cálculo (ou um resumo claro da sua execução, qual dataframe foi usado, e a forma como '{coluna_valor}' foi tratada e agregada) e quaisquer observações importantes (ex: número de linhas analisadas, tratamento de erros).
- "Top 5 fornecedores por valor total consolidado:"
  1. [Fornecedor A]: [Valor A Formatado R$ xxx.xxx,xx]
  2. [Fornecedor B]: [Valor B Formatado R$ xxx.xxx,xx]
  ... e assim por diante.

INSTRUÇÕES CRÍTICAS ADICIONAIS (RELEIA ANTES DE RESPONDER):
1. FOCO ABSOLUTO NA AGREGAÇÃO: `groupby('{coluna_fornecedor}')['{coluna_valor}'].sum()`. É a chave.
2. VALIDAÇÃO DE DADOS: SEMPRE verifique `df.info()`, `df.head()` do dataframe de cabeçalho. A conversão da coluna de valor é a etapa mais crítica antes da soma.
3. FORMATAÇÃO MONETÁRIA BRASILEIRA: Apresente valores finais como R$ XX.XXX.XXX,XX (ponto como separador de milhar, vírgula para decimal). O código exemplo tenta fazer isso. Certifique-se que a saída esteja correta.
4. RESPONDA EM PORTUGUÊS BRASILEIRO.
5. NÃO INVENTE DADOS. Use apenas os dados dos arquivos fornecidos.
6. SEJA EXPLÍCITO sobre qual arquivo/dataframe foi usado para a análise principal e como chegou ao resultado.
7. SE O DATAFRAME DE CABEÇALHO ESTIVER VAZIO ou não contiver as colunas necessárias, informe isso claramente.
"""
            
            response = agent.invoke({"input": enhanced_question})
            
            result = ""
            if isinstance(response, dict):
                result = response.get("output", response.get("result", response.get("answer", "")))
                if not result and response: 
                    for val in response.values():
                        if isinstance(val, str):
                            result = val
                            break
            else:
                result = str(response)
            
            if result and result.strip():
                result_str = result.strip()
                lines = result_str.split('\n')
                clean_lines = []
                in_tool_code_block = False 
                for line in lines:
                    if line.strip().startswith("```python"):
                        in_tool_code_block = True
                        clean_lines.append(line)
                        continue
                    if line.strip().startswith("```") and in_tool_code_block:
                        in_tool_code_block = False
                        clean_lines.append(line)
                        continue
                    
                    if in_tool_code_block: 
                         clean_lines.append(line)
                         continue

                    debug_terms_to_remove = [
                        '> entering new agentexecutor chain', 
                        '> entering new llmchain object',
                        'invoking agent with', 'invoking llm', 
                        'tool execution result', 'action:', 'action input:', 'observation:',
                    ]
                    temp_line_lower = line.lower()
                    # Evitar remover linhas que contenham "thought:" mas são parte da resposta útil,
                    # a menos que sejam apenas "Thought: ..." no início de uma linha de log.
                    # A heurística aqui é se a linha *começa* com "thought:" (ignorando espaços)
                    # e é provável que seja um log.
                    is_log_line = False
                    if temp_line_lower.lstrip().startswith("thought:"):
                        is_log_line = True
                    
                    for term in debug_terms_to_remove:
                        if temp_line_lower.startswith(term):
                            is_log_line = True
                            break
                    
                    if not is_log_line:
                        if temp_line_lower.strip() not in ["okay.", "got it.", "sure."]:
                            # Remover "Final Answer:" se estiver no início da linha
                            if line.lstrip().startswith("Final Answer:"):
                                clean_lines.append(line.lstrip()[len("Final Answer:"):].lstrip())
                            elif line.lstrip().startswith("FINAL ANSWER:"):
                                clean_lines.append(line.lstrip()[len("FINAL ANSWER:"):].lstrip())
                            else:
                                clean_lines.append(line)

                result_str = '\n'.join(clean_lines).strip()

                result_str = result_str.replace('The supplier with the highest total consolidated amount is:', 'O fornecedor com o maior montante total consolidado é:')
                result_str = result_str.replace('Total consolidated value:', 'Valor total consolidado:')
                result_str = result_str.replace('Top 5 suppliers by total consolidated value:', 'Top 5 fornecedores por valor total consolidado:')
                result_str = result_str.replace('Key Pandas Code Executed and Observations:', 'Código Pandas Chave Executado e Observações:')

                return result_str
            else:
                return "O agente não conseguiu processar a consulta ou não retornou uma resposta estruturada. Verifique os logs do console para depuração."
                
        except Exception as e:
            st.error(f"Erro crítico ao processar consulta na camada de query: {str(e)}")
            import traceback
            traceback.print_exc() 
            return f"Erro crítico ao processar sua consulta: {str(e)}. Por favor, verifique os logs do console do servidor ou tente novamente."

    def get_data_summary(self):
        """Retorna um resumo dos dados carregados."""
        summary = {}
        if not self.dataframes:
            return {"message": "Nenhum dado carregado ainda."}
        for file_type, df_instance in self.dataframes.items():
            if isinstance(df_instance, pd.DataFrame):
                summary[file_type] = {
                    'linhas': len(df_instance),
                    'colunas': len(df_instance.columns),
                    'colunas_lista': df_instance.columns.tolist(),
                    'tipos': {col: str(dtype) for col, dtype in df_instance.dtypes.to_dict().items()},
                    'primeiras_linhas': df_instance.head().to_dict('records')
                }
            else:
                summary[file_type] = {"error": f"'{file_type}' não é um DataFrame válido ou não foi carregado."}
        return summary

def main():
    st.set_page_config(
        page_title="Agente de Análise de Notas Fiscais - Google Gemini",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🤖 Agente Inteligente para Análise de Notas Fiscais")
    st.markdown("### Utilizando Google Gemini API & LangChain")
    st.markdown("---")
    
    if 'csv_agent_instance' not in st.session_state:
        try:
            # A API Key será lida de .env dentro de create_llm se não fornecida aqui
            st.session_state.csv_agent_instance = CSVAnalysisAgent() 
        except ValueError as e: 
            st.error(f"Erro de inicialização do agente: {e}")
            return
        except Exception as e_init: # Outras exceções na inicialização
            st.error(f"Erro inesperado na inicialização do CSVAnalysisAgent: {e_init}")
            return

    csv_analyzer = st.session_state.csv_agent_instance

    st.header("📁 1. Upload do Arquivo ZIP com CSVs")
    uploaded_file = st.file_uploader(
        "Faça upload do arquivo ZIP contendo os CSVs de notas fiscais (ex: cabecalho.csv, itens.csv)",
        type=['zip'],
        help=f"O ZIP deve conter arquivos CSV. Um deles deve ser o de '{csv_analyzer.cabecalho_file_type_name}' das notas fiscais para a análise principal."
    )
    
    if uploaded_file is not None:
        with st.spinner("Processando arquivo ZIP e carregando dados..."):
            csv_analyzer.dataframes = {}
            csv_analyzer.file_info = {}
            st.session_state.pop('cabecalho_df_variable_name', None) 
            st.session_state.pop('current_agent_type', None)

            temp_dir_path = None # Para garantir que tem um valor
            try:
                # Assegurar que utils_google.py e suas funções estão disponíveis e corretas.
                # Se extract_zip_file ou NotaFiscalValidator não estiverem definidos ou importados,
                # o código falhará aqui.
                if 'extract_zip_file' not in globals() or 'NotaFiscalValidator' not in globals():
                     st.error("Funções utilitárias 'extract_zip_file' ou 'NotaFiscalValidator' não encontradas. Verifique 'utils_google.py'.")
                     return # Parar execução se utilitários não estiverem lá

                temp_dir_path = extract_zip_file(uploaded_file) 
            except NameError as ne:
                st.error(f"Erro de nome: {ne}. A função 'extract_zip_file' ou 'NotaFiscalValidator' não foi definida ou importada corretamente de 'utils_google.py'.")
                return
            except Exception as e_zip:
                st.error(f"Falha ao extrair o arquivo ZIP: {e_zip}")
                temp_dir_path = None # Atribuição explícita em caso de falha
            
            if temp_dir_path: # Procede apenas se temp_dir_path for um caminho válido
                st.success(f"✅ Arquivo ZIP extraído com sucesso para o diretório temporário.")
                
                try:
                    validator = NotaFiscalValidator() 
                except Exception as e_val_init:
                     st.error(f"Erro ao inicializar NotaFiscalValidator: {e_val_init}. Assegure-se que 'utils_google.py' está correto.")
                     return # Parar se o validador falhar

                csv_files_found = [f for f in os.listdir(temp_dir_path) if f.lower().endswith('.csv')]
                
                if not csv_files_found:
                    st.warning("Nenhum arquivo CSV encontrado no ZIP.")
                else:
                    loaded_files_summary = []
                    for csv_file_name in csv_files_found:
                        file_full_path = os.path.join(temp_dir_path, csv_file_name)
                        try:
                            file_type_identified = validator.identify_file_type(file_full_path) 
                        except Exception as e_ident:
                            st.warning(f"Erro ao identificar tipo do arquivo {csv_file_name} com NotaFiscalValidator: {e_ident}. Usando nome do arquivo como tipo.")
                            file_type_identified = os.path.splitext(csv_file_name)[0].lower().replace(" ", "_")


                        if file_type_identified == "unknown": 
                            file_type_identified = f"tipo_desconhecido_{os.path.splitext(csv_file_name)[0].replace(' ', '_')}"

                        success = csv_analyzer.load_csv_data(file_full_path, file_type_identified)
                        if success:
                            loaded_files_summary.append(f"{csv_file_name} (tipo: '{file_type_identified}')")
                
                    if loaded_files_summary:
                        st.success(f"✅ Arquivos CSV carregados: {', '.join(loaded_files_summary)}")
                        
                        if csv_analyzer.cabecalho_file_type_name not in csv_analyzer.dataframes:
                            st.error(f"‼️ ATENÇÃO: O arquivo/tipo de dados '{csv_analyzer.cabecalho_file_type_name}' é ESSENCIAL para a análise principal de totais por fornecedor e NÃO FOI ENCONTRADO ou identificado corretamente. Verifique o conteúdo do ZIP e a lógica de identificação em 'utils_google.py'. A análise principal pode falhar.")
                        else:
                            st.info(f"👍 Arquivo/tipo de dados '{csv_analyzer.cabecalho_file_type_name}' carregado. Pronto para análise de totais por fornecedor.")

                        with st.expander("📊 Resumo dos Dados Carregados"):
                            summary = csv_analyzer.get_data_summary()
                            for file_type_sum, info_sum in summary.items():
                                st.subheader(f"📄 {file_type_sum.replace('_', ' ').title()}")
                                if "error" in info_sum:
                                    st.error(info_sum["error"])
                                    continue
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Linhas", info_sum.get('linhas', 'N/A'))
                                with col2:
                                    st.metric("Colunas", info_sum.get('colunas', 'N/A'))
                                st.write("**Nomes das Colunas:**")
                                st.caption(", ".join(info_sum.get('colunas_lista', [])))
                                
                                if st.checkbox(f"Ver primeiras 5 linhas - {file_type_sum}", key=f"show_head_{file_type_sum}"):
                                    df_preview = pd.DataFrame(info_sum.get('primeiras_linhas', []))
                                    st.dataframe(df_preview, use_container_width=True)
                    else:
                        st.error("Nenhum arquivo CSV pôde ser carregado do ZIP ou todos falharam.")
            # Limpar uploaded_file para permitir novo upload do mesmo arquivo (se desejado)
            st.session_state.uploaded_file_state = None 


    if hasattr(csv_analyzer, 'dataframes') and csv_analyzer.dataframes:
        st.header("🔍 2. Faça sua Pergunta ao Agente")
        
        with st.expander("💡 Exemplos de Perguntas (foco na análise de totais consolidados do 'cabecalho')"):
            st.markdown(f"""
            - Qual é o fornecedor que teve maior montante recebido (total consolidado)? (Pergunta principal)
            - Quais são os 5 fornecedores com maior valor total consolidado?
            - Qual a soma total de '{csv_analyzer.dataframes[csv_analyzer.cabecalho_file_type_name].columns[csv_analyzer.dataframes[csv_analyzer.cabecalho_file_type_name].columns.str.contains('VALOR', case=False)][0] if csv_analyzer.cabecalho_file_type_name in csv_analyzer.dataframes and any(csv_analyzer.dataframes[csv_analyzer.cabecalho_file_type_name].columns.str.contains('VALOR', case=False)) else 'VALOR NOTA FISCAL'}' no arquivo '{csv_analyzer.cabecalho_file_type_name}'?
            - Quantas notas fiscais (linhas) existem no arquivo '{csv_analyzer.cabecalho_file_type_name}'?
            - Descreva as colunas do arquivo '{csv_analyzer.cabecalho_file_type_name}'.
            """)
        
        default_question = "Qual é o fornecedor que teve maior montante recebido (total consolidado)?"
        if 'user_question_input_memory' not in st.session_state:
            st.session_state.user_question_input_memory = default_question

        user_question = st.text_area(
            "Sua pergunta:",
            value=st.session_state.user_question_input_memory,
            height=100,
            key="user_question_input"
        )
        st.session_state.user_question_input_memory = user_question 
        
        if st.button("🚀 Executar Consulta", type="primary", key="run_query_button"):
            if not user_question.strip():
                st.warning("⚠️ Por favor, digite uma pergunta.")
            elif csv_analyzer.cabecalho_file_type_name not in csv_analyzer.dataframes and "total" in user_question.lower() and "fornecedor" in user_question.lower() : # Checagem mais específica
                 st.error(f"‼️ Não é possível executar a consulta principal de totais por fornecedor. O arquivo/tipo de dados '{csv_analyzer.cabecalho_file_type_name}' não foi carregado. Faça o upload de um ZIP contendo-o.")
            else:
                with st.spinner("🧠 O Agente Gemini está processando sua consulta... Por favor, aguarde."):
                    response_from_agent = csv_analyzer.query(user_question) 
                    
                    st.markdown("### 📋 Resposta do Agente:")
                    if response_from_agent and isinstance(response_from_agent, str) and response_from_agent.strip():
                        st.markdown(response_from_agent)
                    elif not response_from_agent:
                         st.error("O agente não retornou uma resposta ou a resposta foi vazia.")
                    else: 
                        st.error(f"Resposta inesperada do agente (tipo: {type(response_from_agent)}): {response_from_agent}")
                    
                    if st.session_state.get("current_agent_type"):
                        st.caption(f"Debug Info: Agente usado: {st.session_state.current_agent_type}, Var. Cabeçalho no prompt: {st.session_state.get('cabecalho_df_variable_name', 'N/A')}")

    else:
        st.info("Faça o upload de um arquivo ZIP na seção acima para começar a análise.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Sobre a Aplicação")
    st.sidebar.markdown("""
    Esta aplicação demonstra o uso de Inteligência Artificial Generativa para analisar dados de notas fiscais em formato CSV.
    - **IA:** Google Gemini API (Modelo gemini-1.5-flash)
    - **Orquestração:** LangChain (Agentes Inteligentes)
    - **Interface:** Streamlit
    - **Dados:** Pandas
    """)
    
    st.sidebar.markdown("### 🔗 Links Úteis")
    st.sidebar.markdown("""
    - [Google AI Studio](https://ai.google.dev/)
    - [Documentação LangChain (Python)](https://python.langchain.com/)
    - [Documentação Streamlit](https://docs.streamlit.io/)
    """)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Versão da Aplicação: 1.2.0 (Fix: Erro max_execution_time)")


if __name__ == "__main__":
    # É crucial que o arquivo utils_google.py com as classes NotaFiscalValidator 
    # e a função extract_zip_file esteja no mesmo diretório ou acessível no PYTHONPATH.
    # Se precisar de um dummy para testar a interface do Streamlit sem a lógica completa:
    # try:
    #     from utils_google import NotaFiscalValidator, extract_zip_file
    # except ImportError:
    #     print("AVISO: utils_google.py não encontrado. Usando implementações dummy para NotaFiscalValidator e extract_zip_file.")
    #     class NotaFiscalValidator:
    #         def identify_file_type(self, file_path_str): # Renomear para evitar conflito
    #             if "cabecalho" in str(file_path_str).lower(): return "cabecalho"
    #             if "itens" in str(file_path_str).lower(): return "itens"
    #             return "unknown"
    #     def extract_zip_file(uploaded_file_obj): # Renomear para evitar conflito
    #         import tempfile, zipfile, os
    #         temp_dir_obj = tempfile.mkdtemp() # Renomear para evitar conflito
    #         try:
    #             with zipfile.ZipFile(uploaded_file_obj, 'r') as zip_ref:
    #                 zip_ref.extractall(temp_dir_obj)
    #             return temp_dir_obj
    #         except Exception as e_dummy_zip:
    #             print(f"Erro no dummy extract_zip_file: {e_dummy_zip}")
    #             # Criar um diretório vazio para evitar que o resto falhe se o zip for inválido no teste
    #             if not os.path.exists(temp_dir_obj): os.makedirs(temp_dir_obj)
    #             return temp_dir_obj
    #     # Atribuir globalmente para que o main() as encontre
    #     globals()['NotaFiscalValidator'] = NotaFiscalValidator
    #     globals()['extract_zip_file'] = extract_zip_file

    main()
