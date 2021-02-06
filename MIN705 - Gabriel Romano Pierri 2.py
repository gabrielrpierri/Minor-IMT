"""

2º Projeto de BIG DATA, DATA MINING E USO DE SPARK PARA MANIPULAÇÃO DE DADOS

Tema: Estudo de Línguas Estrangeiras com uso de Dados.

Descrição: A coleta de dados ocorre por meio da API do Twitter. 
A ideia do projeto é estudar idiomas baseado em dados. Ou seja, encontrar as palavras 
que mais aparecem em um idioma específico, ordenando por frequência de aparição e 
agrupando por classe gramátical (uso de Web Scraping nesta etapa), além de acréscimo de 
frases de exemplos para leitura e melhor absorção da palavra. Busca por tempos verbais 
mais ditos na língua para orientar também o estudo da gramática do Idioma, entre outros.

"""


"""

PARTE 1:
    
CÓDIGO API TWITTER QUE ENGLOBA TODAS KEYS DE UM USUÁRIO E FAZ REQUESTS EM TEMPOS DETERMINADOS
DE MODO A NÃO PUXAR O LIMITE MÁXIMO DE REQUESTS POR TEMPO DO TWITTER

LEMBRAR:
    
 lang='de', items(2200)
 lang='en', items(2650)

"""

########################################################################################
#                               IMPORTANDO BIBLIOTECAS                                 #
########################################################################################

!pip install pyspark
import tweepy as tw
import pandas as pd
import time
import os
import datetime
import random
import json
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import glob 

########################################################################################
#           SELEÇÃO DO DIRETÓRIO LOCAL QUE SALVARÁ OS DADOS DA API (PATH)              #
########################################################################################


# COMPUTADOR - GABRIEL
path = r'C:\Users\gabri\OneDrive\Documentos\Data Science\Twitter - DDL\Phrases.txt'
path1 = r'C:\Users\gabri\OneDrive\Documentos\Data Science\Twitter - DDL\Phrases '


########################################################################################################
        
APIs = []        

########################################################################################
#        LEITURA e AUTENTICAÇÃO DAS CHAVES DE ACESSO E TOKENS DA API DO TWITTER        #
########################################################################################

# GABRIEL
consumer_key = '' 
consumer_secret = ''
access_token = ''
access_token_secret = ''

# GABRIEL
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth)

APIs.append(api)


########################################################################################
#                                     API TWITTER                                      #
########################################################################################

#tweet._json: VER OS ATRIBUTOS DO TWEET SE NECESSÁRIO.

df_phrases = pd.read_csv(path) 
phrases = df_phrases['PHRASES'].to_list()

contador = 0
epoca = 0

for t in range(1000):
    
    if len(phrases) > (300000): # PEGANDO LIMITE DE LINHAS DO EXCEL

        Current_Date = datetime.datetime.today().strftime('%d-%b-%Y (%Hh%M)') # PEGANDO DATA E HORA ATUAIS
        os.rename(path,path1 + str(Current_Date) + '.txt') # SALVANDO ARQUIVO .txt COM A DATA E HORA ATUAIS

        phrases = []    
        
        for api in APIs:

            tweets = tw.Cursor(api.search,q='-filter:retweets AND -filter:replies',lang='de',tweet_mode='extended', truncated = False).items(2200)

            for tweet in tweets:
                contador+=1
                print('[{0}] {1}'.format(contador,tweet.created_at)) # CONTA CADA FRASE E PRINTA HORÁRIO DO TWEET
                print(tweet.full_text) # TEXTO DO TWEET
                phrases.append(tweet.full_text) # APPENDA FRASE NA LISTA PHRASES
            time.sleep(1000) 

        contador = 0   
        s_phrases = pd.Series(phrases)
        df_phrases = pd.DataFrame({'PHRASES':s_phrases})
        df_phrases.to_csv(path, index = False)
        epoca+=1 
        print(epoca)
        print(os.path.getsize(path), 'Bytes')
        print(len(phrases), 'Frases')
 
    else:
        
        for api in APIs:
            tweets = tw.Cursor(api.search,q='-filter:retweets AND -filter:replies',lang='de', tweet_mode='extended', truncated = False).items(2200)
    
            for tweet in tweets:
                contador+=1
                print('[{0}] {1}'.format(contador,tweet.created_at)) # CONTA CADA FRASE E PRINTA HORÁRIO DO TWEET
                print(tweet.full_text) # TEXTO DO TWEET
                phrases.append(tweet.full_text) # APPENDA FRASE NA LISTA PHRASES
            time.sleep(1000) 

        contador = 0   
        s_phrases = pd.Series(phrases)
        df_phrases = pd.DataFrame({'PHRASES':s_phrases})
        df_phrases.to_csv(path, index = False)
        epoca+=1 
        print(epoca)
        print(os.path.getsize(path), 'Bytes')
        print(len(phrases), 'Frases')




"""
PARTE 2:
    
PROCESSO EM PYSPARK DE LEITURA DOS DADOS OBTIDOS PELA PARTE 1 E PROCESSAMENTO DE SPLIT 
DAS FRASES EM PALAVRAS EM UMA COLUNA, JÁ REALIZANDO O GROUP BY E ORDENAÇÃO DELAS PARA ENCONTRO DE
PALAVRAS QUE SÃO DITAS COM MAIS FREQUÊNCIA NO IDIOMA.

"""

########################################################################################
#             CRIANDO UM PONTO DE PARTIDA PARA NOSSA APLICAÇÃO COM SPARK               #
########################################################################################

spark = SparkSession.builder.appName('NLP').getOrCreate()

########################################################################################
#  INICIANDO PROCESSO DE LEITURA E JUNÇÃO DOS FILES .txt OBTIDOS PELO API TWITTER.py   #
########################################################################################


# ENCONTRO DE TODOS CAMINHOS CONTENDO TODOS DADOS NECESSÁRIOS JUNTAR
all_files = glob.glob(r'C:\Users\gabri\OneDrive\Documentos\Data Science\Twitter - DDL\Phrases*.txt')


# SELECÃO DO DATAFRAME QUE IRÁ SER APPENDADO COM OS DADOS DOS OUTROS ARQUIVOS .txt
df = spark.read.text(r'C:\Users\gabri\OneDrive\Documentos\Data Science\Twitter - DDL\Phrases.txt')


# EXCLUSÃO OS DADOS PROVENIENTES DO FILE JÁ LIDO ACIMA EVITANDO DUPLICAÇÃO DE FRASES
if '/content/Phrases.txt' in all_files: 
  all_files.remove('/content/Phrases.txt')
else:
  'Phrases já removido'


# JUNÇÃO DOS OUTROS DADOS EM UM DATAFRAME ÚNICO (df)
for filename in all_files:
    dfs = spark.read.text(filename)
    df = df.unionAll(dfs)

df.show()
df.count() # 9.619.440 TWEETS OBTIDOS NA LINGUA ALEMÃ


########################################################################################
#       SPLITANDO TODAS FRASES EM PALAVRAS, CONTANDO E ORDENANDO DECRESCENTEMENTE      #
########################################################################################


# SPLIT POR ESPAÇOS NAS FRASES DOS TWEETS
df_words = df.select(f.explode(f.split('value',' ')))

df_words.show()
df_words.count() # 79.847.667 PALAVRAS OBTIDAS

# LIMPEZA DE CARACTERES ESPECIAIS E OUTRAS SUJEIRAS CONTIDAS NAS PALAVRAS
df_words = df_words.withColumn('col', f.lower('col'))\
    .withColumn('col',f.regexp_replace('col',r"[$“&+,:;=?@’#|\"'<>.-^*()%!]", ""))\
    .withColumn('col',f.regexp_replace('col',r"’", "'"))
    .cont
df_words.show()

# CONTAGEM E AGRUPAMENTO DAS PALAVRAS EM ORDEM DECRESCENTE E LIMPEZA DE 'SUJEIRA'
df_words = df_words.withColumn('value',f.lit(1)) # COLUNA CRIADA PARA AUXILIO NA CONTAGEM
df_frequency = df_words.groupby('col').count().orderBy('count', ascending = False)
df_frequency = df_frequency.filter(df_frequency['count'] > 10)
df_frequency.show()

# SALVANDO RESULTADO EM .csv LOCALMENTE
#df_frequency.toPandas().to_csv(r'C:\Users\gabri\OneDrive\Documentos\Data Science\Twitter - DDL\Sätze.csv', encoding = 'utf-8')

########################################################################################
#             ANÁLISE DA QUANTIDADE DE PALAVRAS QUE POSSUEM MESMO TAMANHO              #
########################################################################################

df_frequency.columns

df_frequency.createOrReplaceTempView('WORDS')
SQL = spark.sql('''SELECT COUNT(*) AS COUNT , CHARACTER_LENGTH(COL) AS LENGTH 
                FROM WORDS GROUP BY LENGTH ORDER BY COUNT DESC ''')
SQL.show()



"""
PARTE 3:
    
WEB SCRAPING, TRAZENDO INFORMAÇÕES GRAMATICAIS DAS PALAVRAS OBTIDAS PELO API DO TWITTER

"""


########################################################################################
#                       IMPORTS NECESSÁRIOS DO WEB SCRAPING                            #
########################################################################################



import pandas as pd
import numpy as np
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup
from  urllib.error import HTTPError
from urllib.error import URLError
import regex as re

########################################################################################
#       INICIO DO CÓDGO DO WEB SCRAPING PARA LÍNGUA INGLESA. ALEMÃ EM CONSTRUÇÃO       #
########################################################################################

Definicao = []
Nivel = []
Gramatica = []
Words = []
Exemplo = []
words = pd.read_csv(r'C:\Users\gabri\OneDrive\Documentos\Data Science\Twitter - DDL\Sätze.csv', encoding = 'utf-8')['col'].tolist() 
url = '' #ENDERÇO HTTP DO SITE

def getHTML(url,word):
    try:
        req = urllib.request.Request(url + word ,headers={'User-agent': 'Mozilla/5.0'})
        html = urlopen(req) # ABRE UM ARQUIVO DE REDE (HTTP) INDICADO POR UMA 'URL' PARA LEITURA.
        
    except HTTPError: # CASO DÊ UM ERRO DA PAGINA ESTAR FORA DO AR, É RETORNADO UM 'None'.
        Words.append(word) # ADD A PALAVRA QUE NÃO FOI ENCONTRADA NO SITE A LISTA DE WORDS MESMO ASSIM (VISUALIZAR DEPOIS QUAIS FORAM)
        Gramatica.append('HTTPError') # ADD NA LISTA DE GRAMATICA AO LADO DA PALAVRA QUE NÃO FOI ENCONTRADA O ERRO GERADO.
        
    except URLError: # CASO DÊ UM ERRO DA URL TER SIDO ESCRITA ERRADA, É RETORNADO UM 'None'. 
        Words.append(word)
        Gramatica.append('URLError')
        
    else:
        global bs
        bs = BeautifulSoup(html, 'html.parser') # ORGANIZA O CODIGO HTML PARA UMA FORMA MAIS FACIL DE NAVEGAR UTILIZANDO O PYTHON (OBJETO BEAUTIFULSOUP).
        return bs # SAI DA FUNÇÃO 'getHTML' RETORNANDO AO CODIGO PRINCIPAL O OBJETO BEUTIFULSOUP.

getHTML(url, 'in')
