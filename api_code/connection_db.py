import mysql.connector
import psycopg2
import pymongo
import streamlit as st

class DBConnector:
    def __init__(self, host, port, user, password, database, db_type):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.db_type = db_type

    def connect(self):
        try:
            if self.db_type == 'mysql':
                self.conn = mysql.connector.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    # charset='utf8mb4',
                    # collation='utf8mb4_unicode_ci',  # Add collation
                    database=self.database
                )
            elif self.db_type == 'mongodb':
                self.conn = pymongo.MongoClient(
                    host=self.host,
                    port=self.port,
                    username=self.user,
                    password=self.password,
                    authSource=self.database
                )
            elif self.db_type == 'postgresql':
                self.conn = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    dbname=self.database
                )
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
        except Exception as e:
            st.error(f"Error connecting to database: {e}")

        return self.conn
    def disconnect(self):
        if self.conn:
            self.conn.close()

    @staticmethod
    def db_config(st):
        if 'db_config' not in st.session_state:
            st.session_state['db_config'] = {
                'db_type': '',
                'user': '',
                'pwd': '',
                'host': '',
                'port': '',
                'db': '',
                'schema':'' 
        }

        config = st.session_state['db_config']
        if config['db_type'] == '':
            config['db_type'] = st.selectbox('Select database type', ['<select>','mysql', 'postgresql'], index=0)
        else:
            config['db_type'] = st.selectbox('Select database type', ['<select>','mysql', 'postgresql'], index=1 if config['db_type'] == 'mysql' else 2)
        if config['db_type'] != '<select>':
            config['user'] = st.text_input('Enter your user name', config['user'])
            config['pwd'] = st.text_input('Enter your password', config['pwd'], type='password')
            config['host'] = st.text_input('Enter your host', config['host'])
            config['port'] = st.text_input('Enter your port', config['port'])
            config['db'] = st.text_input('Enter your database', config['db'])
            if config['db_type'] == 'postgresql':
                config['schema'] = st.text_input('Enter your schema (optional)', config['schema'])

        return config['db_type'], config['user'], config['pwd'], config['host'], config['port'], config['db'], config['schema']
    
    