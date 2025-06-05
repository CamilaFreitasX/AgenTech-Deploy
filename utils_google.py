import pandas as pd
import os
import tempfile
import zipfile
import streamlit as st
from typing import Union, Optional

class NotaFiscalValidator:
    """Classe para validar e identificar tipos de arquivos de notas fiscais"""
    
    def __init__(self):
        # Colunas típicas de cabeçalho de nota fiscal
        self.header_columns = [
            'numero', 'serie', 'data_emissao', 'cnpj_fornecedor', 
            'nome_fornecedor', 'valor_total', 'valor_liquido',
            'desconto', 'acrescimo', 'situacao'
        ]
        
        # Colunas típicas de itens de nota fiscal
        self.item_columns = [
            'numero_nf', 'item', 'codigo_produto', 'descricao',
            'quantidade', 'unidade', 'valor_unitario', 'valor_total',
            'ncm', 'cfop'
        ]
    
    def identify_file_type(self, file_path: str) -> str:
        """
        Identifica se o arquivo é de cabeçalho ou itens de nota fiscal
        
        Args:
            file_path: Caminho para o arquivo CSV
            
        Returns:
            str: 'cabecalho', 'itens' ou 'unknown'
        """
        try:
            # Carrega apenas as primeiras linhas para análise
            df = pd.read_csv(file_path, nrows=5)
            columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Verifica se é arquivo de cabeçalho
            header_match = sum(1 for col in self.header_columns if any(hcol in col for hcol in columns))
            
            # Verifica se é arquivo de itens
            item_match = sum(1 for col in self.item_columns if any(icol in col for icol in columns))
            
            # Verifica pelo nome do arquivo também
            filename = os.path.basename(file_path).lower()
            
            if 'cabecalho' in filename or 'header' in filename or header_match >= 3:
                return 'cabecalho'
            elif 'item' in filename or 'itens' in filename or item_match >= 3:
                return 'itens'
            else:
                # Análise adicional baseada no conteúdo
                if 'numero' in columns and 'serie' in columns:
                    return 'cabecalho'
                elif 'quantidade' in columns and 'valor_unitario' in columns:
                    return 'itens'
                else:
                    return 'unknown'
                    
        except Exception as e:
            st.error(f"Erro ao analisar arquivo {file_path}: {str(e)}")
            return 'unknown'
    
    def validate_csv_structure(self, file_path: str, file_type: str) -> bool:
        """
        Valida se a estrutura do CSV está correta
        
        Args:
            file_path: Caminho para o arquivo CSV
            file_type: Tipo do arquivo ('cabecalho' ou 'itens')
            
        Returns:
            bool: True se válido, False caso contrário
        """
        try:
            df = pd.read_csv(file_path)
            
            if df.empty:
                st.error(f"Arquivo {file_path} está vazio")
                return False
            
            # Validações específicas por tipo
            if file_type == 'cabecalho':
                return self._validate_header_structure(df)
            elif file_type == 'itens':
                return self._validate_items_structure(df)
            else:
                return True  # Para tipos desconhecidos, assumimos válido
                
        except Exception as e:
            st.error(f"Erro ao validar estrutura do arquivo {file_path}: {str(e)}")
            return False
    
    def _validate_header_structure(self, df: pd.DataFrame) -> bool:
        """Valida estrutura específica do cabeçalho"""
        required_numeric_cols = ['valor_total', 'valor_liquido']
        
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if any(req_col in col_lower for req_col in required_numeric_cols):
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        # Tenta converter strings com vírgula para float
                        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
                    except:
                        st.warning(f"Coluna {col} deveria ser numérica")
        
        return True
    
    def _validate_items_structure(self, df: pd.DataFrame) -> bool:
        """Valida estrutura específica dos itens"""
        required_numeric_cols = ['quantidade', 'valor_unitario', 'valor_total']
        
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if any(req_col in col_lower for req_col in required_numeric_cols):
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        # Tenta converter strings com vírgula para float
                        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
                    except:
                        st.warning(f"Coluna {col} deveria ser numérica")
        
        return True
    
    def get_column_info(self, file_path: str) -> dict:
        """
        Retorna informações sobre as colunas do arquivo
        
        Args:
            file_path: Caminho para o arquivo CSV
            
        Returns:
            dict: Informações sobre as colunas
        """
        try:
            df = pd.read_csv(file_path, nrows=100)  # Amostra para análise
            
            info = {
                'total_columns': len(df.columns),
                'total_rows': len(df),
                'columns': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'null_counts': df.isnull().sum().to_dict(),
                'sample_data': df.head().to_dict('records')
            }
            
            # Identifica colunas numéricas e de texto
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            info['numeric_columns'] = numeric_cols
            info['text_columns'] = text_cols
            
            return info
            
        except Exception as e:
            st.error(f"Erro ao analisar colunas do arquivo {file_path}: {str(e)}")
            return {}

def extract_zip_file(uploaded_file) -> Optional[str]:
    """
    Extrai arquivo ZIP para diretório temporário
    
    Args:
        uploaded_file: Arquivo ZIP carregado via Streamlit
        
    Returns:
        str: Caminho do diretório temporário ou None em caso de erro
    """
    try:
        # Cria diretório temporário
        temp_dir = tempfile.mkdtemp()
        
        # Salva o arquivo ZIP temporariamente
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Extrai o arquivo ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Remove o arquivo ZIP
        os.remove(zip_path)
        
        # Verifica se há arquivos CSV
        csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
        
        if not csv_files:
            st.error("Nenhum arquivo CSV encontrado no ZIP")
            return None
        
        st.success(f"Encontrados {len(csv_files)} arquivo(s) CSV: {', '.join(csv_files)}")
        return temp_dir
        
    except zipfile.BadZipFile:
        st.error("Arquivo ZIP inválido ou corrompido")
        return None
    except Exception as e:
        st.error(f"Erro ao extrair arquivo ZIP: {str(e)}")
        return None

def format_currency(value: Union[float, int]) -> str:
    """
    Formata valor como moeda brasileira
    
    Args:
        value: Valor numérico
        
    Returns:
        str: Valor formatado como moeda
    """
    try:
        return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return str(value)

def parse_currency(value: str) -> float:
    """
    Converte string de moeda para float
    
    Args:
        value: String representando valor monetário
        
    Returns:
        float: Valor numérico
    """
    try:
        # Remove símbolos e converte vírgula para ponto
        clean_value = str(value).replace("R$", "").replace(".", "").replace(",", ".").strip()
        return float(clean_value)
    except:
        return 0.0

def validate_date_format(date_str: str) -> bool:
    """
    Valida se a data está no formato esperado (AAAA-MM-DD HH:MM:SS)
    
    Args:
        date_str: String da data
        
    Returns:
        bool: True se válida, False caso contrário
    """
    try:
        pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
        return True
    except:
        try:
            pd.to_datetime(date_str, format='%Y-%m-%d')
            return True
        except:
            return False

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e padroniza DataFrame
    
    Args:
        df: DataFrame a ser limpo
        
    Returns:
        pd.DataFrame: DataFrame limpo
    """
    # Copia o DataFrame
    cleaned_df = df.copy()
    
    # Remove espaços em branco dos nomes das colunas
    cleaned_df.columns = cleaned_df.columns.str.strip()
    
    # Remove linhas completamente vazias
    cleaned_df = cleaned_df.dropna(how='all')
    
    # Converte colunas de valores monetários
    for col in cleaned_df.columns:
        if 'valor' in col.lower() or 'preco' in col.lower():
            try:
                cleaned_df[col] = cleaned_df[col].apply(lambda x: parse_currency(x) if pd.notna(x) else 0.0)
            except:
                pass
    
    # Converte colunas de data
    for col in cleaned_df.columns:
        if 'data' in col.lower() or 'emissao' in col.lower():
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
            except:
                pass
    
    return cleaned_df

def get_sample_questions() -> list:
    """
    Retorna lista de perguntas exemplo para o sistema
    
    Returns:
        list: Lista de perguntas exemplo
    """
    return [
        "Qual é o fornecedor que teve maior montante recebido?",
        "Qual item teve maior volume entregue (em quantidade)?",
        "Quantas notas fiscais foram emitidas no total?",
        "Qual é a soma total de todos os valores das notas fiscais?",
        "Quais são os 5 fornecedores com maior valor total?",
        "Qual é a média de valor por item?",
        "Quantos itens diferentes foram comprados?",
        "Qual é o produto mais caro?",
        "Em que período foram emitidas as notas fiscais?",
        "Qual é a distribuição de valores por fornecedor?",
        "Qual é o CNPJ do fornecedor com maior volume de vendas?",
        "Quantos produtos únicos foram comprados?",
        "Qual é o valor médio das notas fiscais?",
        "Quais notas fiscais têm valor acima de R$ 10.000?",
        "Qual é a quantidade total de itens comprados?"
    ]