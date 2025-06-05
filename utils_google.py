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
from langchain.agents.agent_types import AgentType
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
                temperature=0.2, # Reduzir um pouco a temperatura para tarefas prescritivas
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
                # Tentar com sep=None para autodeterminar, depois com vírgula, depois com ponto e vírgula
                df = pd.read_csv(file_path, encoding='utf-8', sep=None, engine='python', on_bad_lines='warn')
            except (UnicodeDecodeError, pd.errors.ParserError):
                try:
                    df = pd.read_csv(file_path, encoding='latin1', sep=None, engine='python', on_bad_lines='warn')
                except (UnicodeDecodeError, pd.errors.ParserError):
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8', sep=',', on_bad_lines='warn')
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        try:
                            df = pd.read_csv(file_path, encoding='latin1', sep=',', on_bad_lines='warn')
                        except (UnicodeDecodeError, pd.errors.ParserError):
                            try:
                                df = pd.read_csv(file_path, encoding='utf-8', sep=';', on_bad_lines='warn')
                            except (UnicodeDecodeError, pd.errors.ParserError):
                                try:
                                    df = pd.read_csv(file_path, encoding='latin1', sep=';', on_bad_lines='warn')
                                except Exception as e_final:
                                    st.error(f"Não foi possível ler o arquivo CSV {os.path.basename(file_path)} com várias tentativas: {e_final}")
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
                        handle_parsing_errors="Se ocorrer um erro de parsing na saída da LLM, tente corrigi-lo ou peça para a LLM reformatar a saída.", # String mais descritiva
                        max_iterations=20, # Aumentado um pouco para acomodar o pipeline detalhado
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

            valid_paths = [self.file_info.get(ft_name, {}).get('path') for ft_name in list_of_df_names
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
                    handle_parsing_errors=True,
                    max_iterations=20, # Aumentado
                    early_stopping_method="generate",
                    max_execution_time=600
                )
                print(f"✅ Agente CSV (path-based) criado com sucesso para {len(valid_paths)} arquivos.")
                st.session_state.current_agent_type = "csv_multi_path" if isinstance(agent_path_input, list) else "csv_single_path"

                if st.session_state.current_agent_type == "csv_single_path":
                    first_file_type_path_fallback = None
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

            # Determinar a variável do DataFrame de cabeçalho para o prompt
            target_df_variable_for_prompt = "df" # Padrão
            # Guia para o agente sobre qual arquivo/tipo de dados focar
            guidance_for_cabecalho_data = f"o arquivo/DataFrame do tipo '{self.cabecalho_file_type_name}' (que pode ser um arquivo como '202401_NFs_Cabecalho.csv')"

            current_agent_context = st.session_state.get("current_agent_type")
            cabecalho_df_var_from_session = st.session_state.get("cabecalho_df_variable_name")

            if current_agent_context == "pandas_multi" and cabecalho_df_var_from_session:
                target_df_variable_for_prompt = cabecalho_df_var_from_session
                guidance_for_cabecalho_data = f"o DataFrame '{target_df_variable_for_prompt}' (que representa os dados do tipo '{self.cabecalho_file_type_name}')"
                print(f"Query: Usando '{target_df_variable_for_prompt}' para o pipeline (agente pandas_multi).")
            elif current_agent_context == "csv_single_path" and cabecalho_df_var_from_session == "df":
                guidance_for_cabecalho_data = f"o DataFrame '{target_df_variable_for_prompt}' (carregado do arquivo CSV único que é do tipo '{self.cabecalho_file_type_name}')"
                print(f"Query: Usando 'df' para o pipeline (agente CSV único sobre cabeçalho).")
            elif current_agent_context == "csv_multi_path":
                guidance_for_cabecalho_data = f"o arquivo CSV que corresponde ao tipo '{self.cabecalho_file_type_name}' (ex: '202401_NFs_Cabecalho.csv'). Você precisará carregá-lo se ainda não o fez."
                print(f"Query: Agente csv_multi_path. O agente precisará carregar o CSV '{self.cabecalho_file_type_name}'.")
            else:
                 print(f"AVISO: Contexto do agente é '{current_agent_context}'. O nome da variável do dataframe de cabeçalho não foi determinado explicitamente para o prompt. O Agente precisará inferir qual dataframe/arquivo usar para '{self.cabecalho_file_type_name}'.")


            # O novo prompt fornecido pelo usuário:
            # Atenção: As chaves duplas {{ }} são para escapar chaves dentro de f-strings, se este prompt fosse um f-string.
            # Como ele será uma string multi-linhas normal, não precisa de escape especial além das aspas da string.
            # O nome do arquivo "202401_NFs_Cabecalho.csv" é específico. O agente deve ser instruído a usar
            # o dataframe de cabeçalho que ele tem acesso, independentemente do nome original do arquivo,
            # mas o prompt pode mencionar esse nome como um exemplo do *tipo* de arquivo.

            # Construir o prompt
            # Usar ''' para string multi-linhas é mais fácil para incorporar o prompt do usuário.
            enhanced_question = f"""
Você é um agente de análise de dados Python/Pandas extremamente preciso e orientado a processos.
Sua tarefa é analisar dados de Notas Fiscais para identificar o fornecedor com o maior montante total consolidado, seguindo RIGOROSAMENTE o pipeline de dados abaixo.

**Contexto dos Dados Disponíveis:**
Você tem acesso a dados de notas fiscais. O foco principal da análise deve ser {guidance_for_cabecalho_data}.
Se você estiver trabalhando com DataFrames já carregados (ex: `df_0`, `df_1`), {target_df_variable_for_prompt} é o nome que você deve usar para se referir ao DataFrame de cabeçalho nas etapas de processamento de dados abaixo.
Se você precisar carregar um arquivo CSV, o arquivo de cabeçalho (ex: `202401_NFs_Cabecalho.csv`) deve ser carregado com `sep=","`.

**Pipeline de Dados Obrigatório (Siga EXATAMENTE estas etapas):**

1.  **Importar/Selecionar o DataFrame de Cabeçalho Integral:**
    * Selecione/use o DataFrame `{target_df_variable_for_prompt}` que representa os dados de cabeçalho.
    * Se estiver carregando de um arquivo CSV (ex: "202401_NFs_Cabecalho.csv"), use `pd.read_csv("caminho_para_o_arquivo_cabecalho.csv", sep=",")`. Carregue 100% das linhas sem filtragem prévia nesta etapa.
    * **Verificação Crítica Inicial:** Confirme se o DataFrame `{target_df_variable_for_prompt}` contém as colunas "RAZÃO SOCIAL EMITENTE" e "VALOR NOTA FISCAL". Se não, informe o erro e pare.

2.  **Validar Integridade e Limpar Dados em `{target_df_variable_for_prompt}`:**
    * Remova todas as linhas onde "RAZÃO SOCIAL EMITENTE" ou "VALOR NOTA FISCAL" sejam ausentes (NaN). Código: `{target_df_variable_for_prompt} = {target_df_variable_for_prompt}.dropna(subset=["RAZÃO SOCIAL EMITENTE", "VALOR NOTA FISCAL"])`
    * Normalize a coluna "RAZÃO SOCIAL EMITENTE" para consistência e para evitar duplicidades ocultas. Aplique as seguintes transformações, nesta ordem:
        * Remover espaços em branco no início e no fim (`.str.strip()`).
        * Converter para maiúsculas (`.str.upper()`).
        * Normalizar para remover acentuação (ex: "NFKD", encode "ascii" com "ignore", decode "utf-8"). Código:
            ```python
            {target_df_variable_for_prompt}["RAZÃO SOCIAL EMITENTE"] = (
                {target_df_variable_for_prompt}["RAZÃO SOCIAL EMITENTE"]
                .str.strip()
                .str.upper()
                .str.normalize("NFKD")
                .str.encode("ascii", errors="ignore")
                .str.decode("utf-8")
            )
            ```

3.  **Padronizar Tipo de Dado da Coluna de Valor em `{target_df_variable_for_prompt}`:**
    * Converta a coluna "VALOR NOTA FISCAL" para o tipo `float`. É crucial que ela esteja como string antes das substituições. Primeiro, garanta que é string, depois remova os separadores de milhar (ponto) e substitua a vírgula decimal por ponto. Código:
        ```python
        {target_df_variable_for_prompt}["VALOR NOTA FISCAL"] = (
            {target_df_variable_for_prompt}["VALOR NOTA FISCAL"]
            .astype(str)  # Garantir que é string
            .str.replace(".", "", regex=False) # Remove separador de milhar
            .str.replace(",", ".", regex=False) # Substitui vírgula decimal por ponto
            .astype(float) # Converte para float
        )
        ```

4.  **Agregar por Fornecedor em `{target_df_variable_for_prompt}`:**
    * Calcule o total consolidado de "VALOR NOTA FISCAL" para cada "RAZÃO SOCIAL EMITENTE".
    * Use: `totais_por_fornecedor = {target_df_variable_for_prompt}.groupby("RAZÃO SOCIAL EMITENTE")["VALOR NOTA FISCAL"].sum()`

5.  **Ordenar e Extrair o Top 1 Fornecedor:**
    * Ordene `totais_por_fornecedor` de forma decrescente para obter um ranking.
        ```python
        ranking_fornecedores = totais_por_fornecedor.sort_values(ascending=False)
        ```
    * O fornecedor líder é o primeiro do ranking. Se o ranking estiver vazio, informe que nenhum fornecedor foi encontrado.
        ```python
        if not ranking_fornecedores.empty:
            fornecedor_lider = ranking_fornecedores.index[0]
            valor_lider = ranking_fornecedores.iloc[0]
        else:
            fornecedor_lider = "Nenhum fornecedor encontrado após filtros."
            valor_lider = 0.0
        ```

6.  **Output Executivo (Formato da Resposta Final):**
    * Apresente o resultado claramente.
    * Use o seguinte formato para a resposta principal:
        "O fornecedor com o maior montante total consolidado é: [Nome do Fornecedor Líder]."
        "Valor total consolidado: R$ [Valor Líder Formatado]."
        (Onde [Valor Líder Formatado] deve ser no padrão brasileiro, ex: R$ 1.234.567,89. Use a formatação de string apropriada em Python para isso, como no exemplo de código abaixo).
    * **Opcional, mas recomendado para robustez:** Se solicitado ou como parte de uma análise completa, mostre também o Top 5 fornecedores em uma tabela ou lista formatada.
    * **Inclua sempre uma seção "Detalhes da Execução do Pipeline:"** onde você resume brevemente as etapas chave que executou, qual DataFrame/arquivo usou (confirmando o uso de {target_df_variable_for_prompt} ou o arquivo carregado), e o número de linhas antes e depois da limpeza, se relevante.

**Exemplo de Código Python Completo para o Pipeline (Adapte `{target_df_variable_for_prompt}` conforme necessário):**
```python
import pandas as pd
import unicodedata # Para normalização mais robusta se necessário

# Supondo que '{target_df_variable_for_prompt}' é o DataFrame de cabeçalho já carregado e correto.
# Se '{target_df_variable_for_prompt}' for um nome de arquivo, o agente deve carregá-lo primeiro:
# Ex: {target_df_variable_for_prompt} = pd.read_csv("caminho_para_202401_NFs_Cabecalho.csv", sep=",")

# --- Início do Pipeline ---
df_processado = {target_df_variable_for_prompt}.copy() # Trabalhar com uma cópia

# 2. Validar integridade e limpar dados
print(f"Linhas antes da limpeza: {{len(df_processado)}}")
df_processado = df_processado.dropna(subset=["RAZÃO SOCIAL EMITENTE", "VALOR NOTA FISCAL"])
print(f"Linhas após dropna: {{len(df_processado)}}")

df_processado.loc[:, "RAZÃO SOCIAL EMITENTE"] = (
    df_processado["RAZÃO SOCIAL EMITENTE"]
    .str.strip()
    .str.upper()
    .str.normalize("NFKD")
    .str.encode("ascii", errors="ignore")
    .str.decode("utf-8")
)

# 3. Padronizar tipo de dado
df_processado.loc[:, "VALOR NOTA FISCAL"] = (
    df_processado["VALOR NOTA FISCAL"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# 4. Agregar por fornecedor
totais = df_processado.groupby("RAZÃO SOCIAL EMITENTE")["VALOR NOTA FISCAL"].sum()

# 5. Ordenar decrescentemente e extrair top 1
ranking = totais.sort_values(ascending=False)

fornecedor_mais_valioso = "N/A"
valor_mais_alto = 0.0
top_5_fornecedores_texto = "Nenhum fornecedor no ranking."

if not ranking.empty:
    fornecedor_mais_valioso = ranking.index[0]
    valor_mais_alto = ranking.iloc[0]

    # Formatação do valor para o padrão brasileiro R$ xxx.xxx,xx
    s_valor = f"{{valor_mais_alto:.2f}}" # Ex: "1234567.89"
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
    valor_formatado_br = f"R$ {{inteiro_formatado}},{{decimal}}"

    print(f"Fornecedor com maior montante recebido: {{fornecedor_mais_valioso}} – {{valor_formatado_br}}")

    # Gerar Top 5 para output
    top_5_list = []
    for i, (fornec, val) in enumerate(ranking.head(5).items()):
        s_val_item = f"{{val:.2f}}"
        partes_item = s_val_item.split('.')
        int_item = partes_item[0]
        dec_item = partes_item[1] if len(partes_item) > 1 else "00"
        int_fmt_item = ""
        if len(int_item) > 3:
            for idx, d_item in enumerate(reversed(int_item)):
                if idx > 0 and idx % 3 == 0: int_fmt_item = "." + int_fmt_item
                int_fmt_item = d_item + int_fmt_item
        else: int_fmt_item = int_item
        val_fmt_br_item = f"R$ {{int_fmt_item}},{{dec_item}}"
        top_5_list.append(f"{{i+1}}. {{fornec}}: {{val_fmt_br_item}}")
    top_5_fornecedores_texto = "\\n".join(top_5_list)
    print("\\nTop 5 Fornecedores:")
    print(top_5_fornecedores_texto)
else:
    print("Ranking de fornecedores está vazio após processamento.")

# --- Fim do Pipeline ---
# A resposta final para o usuário deve vir da execução deste print ou dos valores formatados.
# O agente deve apresentar:
# 1. O fornecedor líder e seu valor.
# 2. A seção "Detalhes da Execução do Pipeline:".
# 3. O Top 5 formatado.
