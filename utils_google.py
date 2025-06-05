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
    try:
        from langchain.agents import create_pandas_dataframe_agent
    except ImportError:
        st.error("Não foi possível importar 'create_pandas_dataframe_agent'. Verifique a instalação do LangChain.")
        pass # Permitir que o app continue se este agente específico não for crucial ou usado.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from utils_google import NotaFiscalValidator, extract_zip_file # Supondo que utils_google.py exista e funcione
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

class CSVAnalysisAgent:
    def __init__(self, google_api_key=None):
        self.google_api_key = google_api_key
        self.agents = {}
        self.dataframes = {}
        self.file_info = {}
        self.cabecalho_file_type_name = "cabecalho"

    def create_llm(self):
        if not self.google_api_key:
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            if not self.google_api_key:
                raise ValueError("Google API Key é necessária para usar o agente e não foi encontrada.")
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.google_api_key,
                temperature=0.1, # Temperatura baixa para seguir instruções precisas
                convert_system_message_to_human=True,
            )
            return llm
        except Exception as e:
            st.error(f"Erro ao criar modelo Gemini: {str(e)}")
            return None

    def load_csv_data(self, file_path, file_type):
        try:
            df = None
            encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            separators_to_try = [None, ',', ';'] # None para autodetectar
            
            for enc in encodings_to_try:
                for sep_try in separators_to_try:
                    try:
                        df = pd.read_csv(file_path, encoding=enc, sep=sep_try, engine='python' if sep_try is None else 'c', on_bad_lines='warn')
                        if df is not None and not df.empty:
                            print(f"Arquivo {os.path.basename(file_path)} lido com encoding '{enc}' e separador '{sep_try if sep_try else 'autodetectado'}'.")
                            break # Sucesso
                    except Exception:
                        continue # Tentar próxima combinação
                if df is not None and not df.empty:
                    break
            
            if df is None or df.empty:
                st.error(f"Não foi possível ler o arquivo CSV {os.path.basename(file_path)} com as tentativas de encoding/separador.")
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
        try:
            if not self.dataframes:
                print("Erro: Nenhum dataframe carregado.")
                st.error("Nenhum dataframe carregado para criar o agente geral.")
                return None
            llm = self.create_llm()
            if llm is None: return None

            ordered_df_tuples = [(ft, df) for ft, df in self.dataframes.items() if isinstance(df, pd.DataFrame)]
            if not ordered_df_tuples:
                st.error("Nenhum dataframe válido encontrado para criar o agente.")
                return None

            list_of_dfs = [df_tuple[1] for df_tuple in ordered_df_tuples]
            list_of_df_names = [df_tuple[0] for df_tuple in ordered_df_tuples]

            cabecalho_df_index = -1
            st.session_state.pop('cabecalho_df_variable_name', None)
            try:
                cabecalho_df_index = list_of_df_names.index(self.cabecalho_file_type_name)
                st.session_state.cabecalho_df_variable_name = f"df_{cabecalho_df_index}"
            except ValueError:
                print(f"Aviso: DataFrame '{self.cabecalho_file_type_name}' não encontrado.")

            if list_of_dfs:
                try:
                    prefix_parts = ["Você é um agente de análise de dados Python/Pandas altamente competente."]
                    prefix_parts.append(f"Você tem acesso a {len(list_of_dfs)} dataframes pandas: {', '.join([f'df_{i} (tipo: {name})' for i, name in enumerate(list_of_df_names)])}.")
                    if cabecalho_df_index != -1:
                        prefix_parts.append(f"O dataframe 'df_{cabecalho_df_index}' (tipo '{self.cabecalho_file_type_name}') é o principal para análise de notas fiscais.")
                    prefix_message = "\n".join(prefix_parts)
                    input_for_agent = list_of_dfs[0] if len(list_of_dfs) == 1 else list_of_dfs

                    general_agent = create_pandas_dataframe_agent(
                        llm, input_for_agent, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        allow_dangerous_code=True, prefix=prefix_message,
                        handle_parsing_errors="Se ocorrer um erro de parsing na saída da LLM, tente corrigi-lo ou peça para a LLM reformatar a saída.",
                        max_iterations=20, early_stopping_method="generate", max_execution_time=700) # Aumentado max_execution_time
                    st.session_state.current_agent_type = "pandas_multi"
                    return general_agent
                except Exception as e_pandas:
                    print(f"Erro ao criar agente pandas: {e_pandas}")
                    st.warning(f"Falha agente pandas: {e_pandas}. Fallback para CSV.")

            valid_paths = [info.get('path') for ft, info in self.file_info.items() if info.get('path') and os.path.exists(info.get('path'))]
            if not valid_paths:
                st.error("Nenhum caminho CSV válido para fallback.")
                return None
            agent_path_input = valid_paths[0] if len(valid_paths) == 1 else valid_paths
            try:
                general_agent = create_csv_agent(
                    llm, agent_path_input, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    allow_dangerous_code=True, handle_parsing_errors=True,
                    max_iterations=20, early_stopping_method="generate", max_execution_time=700) # Aumentado max_execution_time
                st.session_state.current_agent_type = "csv_multi_path" if isinstance(agent_path_input, list) else "csv_single_path"
                if st.session_state.current_agent_type == "csv_single_path":
                    # ... (lógica para st.session_state.cabecalho_df_variable_name) ...
                    pass # Simplificado por agora
                return general_agent
            except Exception as e_csv:
                st.error(f"Falha crítica ao criar agente CSV: {e_csv}")
                return None
        except Exception as e_general:
            st.error(f"Erro geral fatal ao criar agente: {e_general}")
            return None

    def query(self, question):
        try:
            agent = self.create_general_agent()
            if agent is None:
                return "Erro: Não foi possível inicializar o agente de análise para a consulta."

            target_df_variable_for_prompt = "df"
            guidance_for_cabecalho_data = f"o arquivo/DataFrame do tipo '{self.cabecalho_file_type_name}' (que pode ser um arquivo como '202401_NFs_Cabecalho.csv')"
            current_agent_context = st.session_state.get("current_agent_type")
            cabecalho_df_var_from_session = st.session_state.get("cabecalho_df_variable_name")

            if current_agent_context == "pandas_multi" and cabecalho_df_var_from_session:
                target_df_variable_for_prompt = cabecalho_df_var_from_session
                guidance_for_cabecalho_data = f"o DataFrame '{target_df_variable_for_prompt}' (que representa os dados do tipo '{self.cabecalho_file_type_name}')"
            elif current_agent_context == "csv_single_path" and cabecalho_df_var_from_session == "df":
                guidance_for_cabecalho_data = f"o DataFrame '{target_df_variable_for_prompt}' (carregado do arquivo CSV único que é do tipo '{self.cabecalho_file_type_name}')"
            elif current_agent_context == "csv_multi_path":
                guidance_for_cabecalho_data = f"o arquivo CSV que corresponde ao tipo '{self.cabecalho_file_type_name}' (ex: '202401_NFs_Cabecalho.csv'). Você precisará carregá-lo se ainda não o fez."
            
            # Novo prompt detalhado fornecido pelo usuário.
            # Atenção redobrada ao início f""" e ao final """.
            # Todas as chaves {} que são literais para o LLM ver no código de exemplo devem ser escapadas como {{}}
            enhanced_question = f"""
Você é um agente de análise de dados Python/Pandas extremamente preciso e orientado a processos.
Sua tarefa é analisar dados de Notas Fiscais para identificar o fornecedor com o maior montante total consolidado, seguindo RIGOROSAMENTE o pipeline de dados abaixo.

**Contexto dos Dados Disponíveis:**
Você tem acesso a dados de notas fiscais. O foco principal da análise deve ser {guidance_for_cabecalho_data}.
Se você estiver trabalhando com DataFrames já carregados (ex: `df_0`, `df_1`), `{target_df_variable_for_prompt}` é o nome que você deve usar para se referir ao DataFrame de cabeçalho nas etapas de processamento de dados abaixo.
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
            # Certifique-se que a coluna existe antes de modificar
            if "RAZÃO SOCIAL EMITENTE" in {target_df_variable_for_prompt}.columns:
                {target_df_variable_for_prompt}["RAZÃO SOCIAL EMITENTE"] = (
                    {target_df_variable_for_prompt}["RAZÃO SOCIAL EMITENTE"]
                    .astype(str) # Garantir que é string para aplicar métodos .str
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
        # Certifique-se que a coluna existe
        if "VALOR NOTA FISCAL" in {target_df_variable_for_prompt}.columns:
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
    * **Inclua sempre uma seção "Detalhes da Execução do Pipeline:"** onde você resume brevemente as etapas chave que executou, qual DataFrame/arquivo usou (confirmando o uso de `{target_df_variable_for_prompt}` ou o arquivo carregado), e o número de linhas antes e depois da limpeza, se relevante.

**Exemplo de Código Python Completo para o Pipeline (Adapte `{target_df_variable_for_prompt}` conforme necessário e use este código como base para sua execução):**
```python
import pandas as pd
# import unicodedata # Já está implícito na normalização de string

# Supondo que '{target_df_variable_for_prompt}' é o DataFrame de cabeçalho já carregado e correto.
# Se '{target_df_variable_for_prompt}' for um nome de arquivo, o agente deve carregá-lo primeiro:
# Ex: df_para_analise = pd.read_csv("caminho_para_202401_NFs_Cabecalho.csv", sep=",")
# E então usar 'df_para_analise' no lugar de '{target_df_variable_for_prompt}' abaixo.
# Para este exemplo, assumimos que '{target_df_variable_for_prompt}' já é o DataFrame correto.

df_processado = {target_df_variable_for_prompt}.copy()

# Verificação inicial das colunas
required_cols = ["RAZÃO SOCIAL EMITENTE", "VALOR NOTA FISCAL"]
if not all(col in df_processado.columns for col in required_cols):
    print(f"ERRO: Colunas {{required_cols}} não encontradas em {{target_df_variable_for_prompt}}.")
    # O agente deve parar aqui e reportar o erro.
else:
    # 2. Validar integridade e limpar dados
    linhas_antes_dropna = len(df_processado)
    df_processado = df_processado.dropna(subset=required_cols)
    linhas_apos_dropna = len(df_processado)
    print(f"Detalhes da Execução do Pipeline: Iniciado com {{linhas_antes_dropna}} linhas. Após remover NaN em colunas chave: {{linhas_apos_dropna}} linhas.")

    df_processado.loc[:, "RAZÃO SOCIAL EMITENTE"] = (
        df_processado["RAZÃO SOCIAL EMITENTE"]
        .astype(str)
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

    fornecedor_mais_valioso = "N/A (Ranking vazio ou erro)"
    valor_mais_alto_formatado = "R$ 0,00"
    top_5_fornecedores_texto = "Nenhum fornecedor no ranking."

    if not ranking.empty:
        fornecedor_mais_valioso = ranking.index[0]
        valor_mais_alto = ranking.iloc[0]

        # Formatação do valor para o padrão brasileiro R$ xxx.xxx,xx
        s_valor = f"{{valor_mais_alto:.2f}}"
        partes = s_valor.split('.')
        inteiro = partes[0]
        decimal = partes[1] if len(partes) > 1 else "00"
        decimal = decimal.ljust(2, '0') # Garantir dois dígitos decimais
        inteiro_formatado = ""
        if len(inteiro) > 3:
            for i, digito in enumerate(reversed(inteiro)):
                if i > 0 and i % 3 == 0:
                    inteiro_formatado = "." + inteiro_formatado
                inteiro_formatado = digito + inteiro_formatado
        else:
            inteiro_formatado = inteiro
        valor_mais_alto_formatado = f"R$ {{inteiro_formatado}},{{decimal}}"

        print(f"O fornecedor com o maior montante total consolidado é: {{fornecedor_mais_valioso}}.")
        print(f"Valor total consolidado: {{valor_mais_alto_formatado}}")

        top_5_list = []
        for i, (fornec, val) in enumerate(ranking.head(5).items()):
            s_val_item = f"{{val:.2f}}"
            partes_item = s_val_item.split('.')
            int_item = partes_item[0]
            dec_item = partes_item[1] if len(partes_item) > 1 else "00"
            dec_item = dec_item.ljust(2, '0')
            int_fmt_item = ""
            if len(int_item) > 3:
                for idx, d_item in enumerate(reversed(int_item)):
                    if idx > 0 and idx % 3 == 0: int_fmt_item = "." + int_fmt_item
                    int_fmt_item = d_item + int_fmt_item
            else: int_fmt_item = int_item
            val_fmt_br_item = f"R$ {{int_fmt_item}},{{dec_item}}"
            top_5_list.append(f"{{i+1}}. {{fornec}}: {{val_fmt_br_item}}")
        top_5_fornecedores_texto = "\\n".join(top_5_list)
        print("\\nTop 5 fornecedores por valor total consolidado:")
        print(top_5_fornecedores_texto)
    else:
        print("Ranking de fornecedores está vazio após processamento.")
