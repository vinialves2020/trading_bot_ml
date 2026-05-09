import sqlite3
import pandas as pd
import os

class DatabaseManager:
    """
    Gerencia a conexo e as operaes com o banco de dados SQLite local.
    """
    def __init__(self, db_path="data/trading_data.db"):
        # Garante que a pasta data/ existe
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

    def save_data(self, df, table_name, if_exists='append'):
        """
        Salva o DataFrame no banco de dados.
        if_exists: 'replace' (sobrescreve tudo) ou 'append' (adiciona novas linhas)
        """
        print(f" Salvando {len(df)} linhas na tabela '{table_name}'...")
        try:
            # Cria a conexo com o arquivo SQLite
            with sqlite3.connect(self.db_path) as conn:
                # O Pandas faz a mgica de criar a tabela e inserir os dados automaticamente
                df.to_sql(name=table_name, con=conn, if_exists=if_exists, index=True)
            print(" Dados salvos com sucesso no SQLite!")
        except Exception as e:
            print(f" Erro ao salvar no banco de dados: {e}")

    def execute_query(self, query, params=None):
        """Executa um comando SQL arbitrrio (INSERT, CREATE, etc.) com parmetros opcionais"""
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.conn.commit()
        except Exception as e:
            print(f" Erro ao executar query: {e}")

    def load_data(self, table_name=None, query=None):
        """Carrega dados de uma tabela ou executa uma query SQL customizada."""
        try:
            # Se houver uma query, usa ela. Se no, busca a tabela inteira.
            sql = query if query else f"SELECT * FROM {table_name}"

            import pandas as pd
            return pd.read_sql(sql, self.conn)
        except Exception as e:
            print(f" Erro ao carregar do banco de dados: {e}")
            return None
