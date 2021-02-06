
"""

1º PROJETO DE BIG DATA ENVOLVENDO MANIPULAÇÃO DE DADOS PELO SPARK

DADOS DISPONÍVEIS EM: https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store

ANÁLISES REALIZADAS:
1) QUANTIDADE DE USUÁRIOS DISTINTOS QUE ACESSARAM O SITE DE E-COMMERCE EM 2 MESES
2) QUANTIDADE DIFERENTE DE TIPOS DE EVENTOS QUE FORAM REALIZADOS NO SITE PELOS USUÁRIOS
3) QUANTIDADE DE USUÁRIOS, VALORES MÉDIOS, MÁXIMOS E MÍNIMOS DE PRODUTOS QUE OLHAM, ADICIONAM NO CARRINHO E COMPRAM.
4) VALOR TOTAL FATURADO PELO SITE E VALOR TOTAL POR USUÁRIO GASTO EM ORDEM DECRESCENTE
5) CATEGORIAS DOS PRODUTOS MAIS VISUALIZADOS, ADICIONADOS NO CARRINHO E COMPRADOS
6) MARCAS DOS PRODUTOS MAIS VISUALIZADOS, ADICIONADOS NO CARRINHO E COMPRADOS
7) MAIORES E MENORES PREÇOS PAGOS PARA CADA MARCA DE PRODUTOS MAIS COMPRADOS
8) HORÁRIOS QUE TIVERAM MAIOR NÚMEROS DE VISUALIZAÇÕES E COMPRAS DE PRODUTOS

"""
########################################################################################
#                               IMPORTANDO BIBLIOTECAS                                 #
########################################################################################


!pip install pyspark
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import glob 


########################################################################################
#             CRIANDO UM PONTO DE PARTIDA PARA NOSSA APLICAÇÃO COM SPARK               #
########################################################################################


spark = SparkSession.builder.appName('NLP').getOrCreate()


########################################################################################
#     INICIANDO PROCESSO DE LEITURA E JUNÇÃO DOS FILES .csv OBTIDOS PELO NO KAGGLE     #
########################################################################################


# SELECÃO DO DATAFRAME QUE IRÁ SER APPENDADO COM OS DADOS DOS OUTROS ARQUIVOS .txt
df_oct = spark.read.csv(r'C:\Users\gabri\OneDrive\Documentos\Minor - Ciência de Dados\MIN706 - Projetos em Ciência de Dados\2019-Oct.csv', inferSchema = True, header = True)
df_nov = spark.read.csv(r'C:\Users\gabri\OneDrive\Documentos\Minor - Ciência de Dados\MIN706 - Projetos em Ciência de Dados\2019-Nov.csv',inferSchema = True, header = True)
df = df_oct.unionAll(df_nov)

df_oct.count() # 42.448.764 ROWS
df_nov.count() # 67.501.979 ROWS
df.count() # 109.950.743 ROWS
#df.distinct().count() # 109.820.004 ROWS    NÃO RODAR ESTE DISTINCT. PROBLEMA DE MEMÓRIA RAM E DISCO

####################################################################################################
#                                         DATA ANALYSIS                                            #
####################################################################################################

"""

1) QUANTIDADE DE USUÁRIOS DISTINTOS QUE ACESSARAM O SITE DE E-COMMERCE EM 2 MESES
    
"""
df.select('user_id').distinct().count()

# HÁ 5.316.649 USUÁRIOS DISTINTOS QUE ACESSARAM O SITE DE E-COMMERCE EM 2 MESES (OUT E NOV)

"""

2) QUANTIDADE DIFERENTE DE TIPOS DE EVENTOS QUE FORAM REALIZADOS NO SITE PELOS USUÁRIOS
    
"""

df.select('event_type').groupby('event_type').count().show()

#+----------+-----------+
#|event_type|      count|
#+----------+-----------+
#|      view|104.335.509| 94.9% DOS REGISTROS REFEREM-SE A VISUALIZAÇÃO DE PRODUTOS
#|      cart|  3.955.446|  3.6% DOS REGISTROS REFEREM-SE A ADICÃO DE PRODUTOS NO CARRINHO 
#|  purchase|  1.659.788|  1.5% DOS REGISTROS REFEREM-SE A COMPRAS
#+----------+-----------+
#|     total|109.950.743| 100%
#+----------+-----------+


"""

3) QUANTIDADE DE USUÁRIOS, VALORES MÉDIOS, MÁXIMOS E MÍNIMOS DE PRODUTOS QUE OLHAM, ADICIONAM NO CARRINHO E COMPRAM.
    
"""

df.select('user_id','event_type').filter(df['event_type'] == 'view').groupby('user_id','event_type').count().describe(['count']).show()

#+-------+------------------+
#|summary|             count|
#+-------+------------------+
#|  count|         5.316.128| 5316128/5316649*100 = 99.99% DOS USUÁRIOS VISUALIZARAM PRODUTOS
#|   mean|19.626222130091676| DOS QUAIS CADA USUÁRIO VISUALIZOU EM MÉDIA 20 PRODUTOS
#|    min|                 1|
#|    max|            22.926|
#+-------+------------------+

df.select('user_id','event_type').filter(df['event_type'] == 'cart').groupby('user_id','event_type').count().describe(['count']).show()

#+-------+------------------+
#|summary|             count|
#+-------+------------------+
#|  count|         1.054.133| 1054133/5316649*100 = 19.8% DOS USUÁRIOS ADICIONARAM PRODUTOS NO CARRINHO
#|   mean|3.7523215761199014| DOS QUAIS CADA USUÁRIO ADICIONOU NO CARRINHO EM MÉDIA 4 PRODUTOS
#|    min|                 1|
#|    max|               720|
#+-------+------------------+

df.select('user_id','event_type').filter(df['event_type'] == 'purchase').groupby('user_id','event_type').count().describe(['count']).show()

#+-------+------------------+                                                    
#|summary|             count|
#+-------+------------------+
#|  count|           697.470| 697470/5316649*100 = 13.1% DOS USUÁRIOS FIZERAM COMPRAS 
#|   mean|2.3797267265975597| DOS QUAIS CADA USUÁRIO COMPROU EM MÉDIA 3 PRODUTOS
#|    min|                 1|
#|    max|               640|
#+-------+------------------+

"""

4) VALOR TOTAL FATURADO PELO SITE E VALOR TOTAL POR USUÁRIO GASTO EM ORDEM DECRESCENTE
    
"""

df.select('user_id','price').filter(df['event_type'] == 'purchase').groupby('user_id').sum('price').orderBy('sum(price)', ascending = False).show()

#+---------+------------------+
#|  user_id|        sum(price)|
#+---------+------------------+
#|512386086|344153.28999999986|
#|515384420|         303144.24|
#|513117637|         266331.24|
#|519267944|265569.51999999996|
#|518514099|203986.06999999998|
#|530834332|188820.87000000002|
#|549109608|184394.51999999996|
#|534545940|183309.02999999997|
#|532499743|          180755.4|
#|515715331|175747.25999999998|
#|519250600|174449.34999999998|
#|553431815|171603.13999999998|
#|564068124|163373.56999999998|
#|538216048|161467.83999999994|
#|513320236|         158312.75|
#|513784794|155013.74000000002|
#|555394812|         151984.59|
#|533074223|         128784.04|
#|545563258|124646.51999999999|
#|512842822|         121562.36|
#+---------+------------------+
#only showing top 20 rows


df.select('event_type','price').groupby('event_type').sum('price').show()

#+----------+--------------------+
#|event_type|          sum(price)|
#+----------+--------------------+
#|  purchase| 5.051523927699983E8| 
#|      view| 3.03727027572275E10|
#|      cart|1.1876080511299996E9|
#+----------+--------------------+


"""

5) CATEGORIAS DOS PRODUTOS MAIS VISUALIZADOS, ADICIONADOS NO CARRINHO E COMPRADOS
    
"""

df1 = df.select('category_code').filter(df['event_type'] == 'view').groupby('category_code').count().orderBy('count', ascending = False)
df1.withColumn('% count', df1['count'] / 104335509 * 100).show()    

#+--------------------+--------+------------------+
#|       category_code|   count|           % count|
#+--------------------+--------+------------------+
#|                null|34073918| 32.65802632927204|
#|electronics.smart...|25451835|24.394221338394008|
#|  electronics.clocks| 3267223|3.1314583417616717|
#|  computers.notebook| 3209430|  3.07606684508531|
#|electronics.video.tv| 3127266|2.9973170495578834|
#|electronics.audio...| 2663452|2.5527761598402705|
#|       apparel.shoes| 2596322| 2.488435648500071|
#|appliances.kitche...| 2225429|2.1329545629570847|
#|appliances.enviro...| 2217058|2.1249314075805197|
#|appliances.kitche...| 2145086|  2.05595009844635|
#|   computers.desktop| 1091654| 1.046291919656998|
#|auto.accessories....|  920930|0.8826621049982131|
#|  apparel.shoes.keds|  883559|0.8468440020741165|
#|furniture.bedroom...|  874801| 0.838449927914762|
#|furniture.living_...|  839506|0.8046215598564819|
#|  electronics.tablet|  766065|0.7342322928620592|
#|construction.tool...|  654868|0.6276559210536846|
#|furniture.living_...|  632899|0.6065998106167289|
#|electronics.audio...|  615955|0.5903598936772331|
#|electronics.telep...|  566178|0.5426513038815961|
#+--------------------+--------+------------------+
#only showing top 20 rows




df1 = df.select('category_code').filter(df['event_type'] == 'cart').groupby('category_code').count().orderBy('count', ascending = False)
df1.withColumn('% count', df1['count'] / 3955446 * 100).show()    

#+--------------------+-------+-------------------+
#|       category_code|  count|            % count|
#+--------------------+-------+-------------------+
#|electronics.smart...|1709731|  43.22473369627597|
#|                null| 932219|  23.56798702346082|
#|electronics.audio...| 182276|  4.608228755998692|
#|electronics.video.tv| 142691| 3.6074566559624377|
#|appliances.kitche...|  92264| 2.3325814585763527|
#|  electronics.clocks|  89633| 2.2660655713666675|
#|appliances.enviro...|  82099|  2.075594003811454|
#|  computers.notebook|  74724| 1.8891422105117857|
#|appliances.kitche...|  65228| 1.6490681455390872|
#|       apparel.shoes|  40074|  1.013134801991988|
#|  electronics.tablet|  31426| 0.7944995330488649|
#|appliances.kitche...|  20655| 0.5221914292345288|
#|electronics.telep...|  20174| 0.5100309800715267|
#|     appliances.iron|  18403| 0.4652572680805148|
#|appliances.kitche...|  17561|0.44397016164548825|
#|construction.tool...|  16952|0.42857366779877665|
#|auto.accessories....|  16129| 0.4077669117464882|
#|appliances.enviro...|  16103| 0.4071095901701098|
#|   computers.desktop|  16077| 0.4064522685937313|
#|appliances.enviro...|  15830|0.40020771361813556|
#+--------------------+-------+-------------------+
#only showing top 20 rows

df1 = df.select('category_code').filter(df['event_type'] == 'purchase').groupby('category_code').count().orderBy('count', ascending = False)
df1.withColumn('% count', df1['count'] / 1659788 * 100).show()    

#+--------------------+------+-------------------+
#|       category_code| count|            % count|
#+--------------------+------+-------------------+
#|electronics.smart...|720665| 43.419099306658445|
#|                null|407643| 24.559943800051574|
#|electronics.audio...| 71337| 4.2979585344634375|
#|electronics.video.tv| 51839|  3.123230195663542|
#|  electronics.clocks| 41143|  2.478810546889121|
#|appliances.kitche...| 35920|  2.164131804784707|
#|  computers.notebook| 34023| 2.0498401000609716|
#|appliances.enviro...| 30571| 1.8418617317392343|
#|appliances.kitche...| 24260|  1.461632449445351|
#|       apparel.shoes| 14395| 0.8672794356869672|
#|  electronics.tablet| 11741| 0.7073794966586093|
#|appliances.kitche...|  8855| 0.5335018689133793|
#|auto.accessories....|  8440| 0.5084986757344914|
#|     appliances.iron|  8303| 0.5002446095525452|
#|electronics.telep...|  8076| 0.4865681641269849|
#|appliances.kitche...|  7086| 0.4269219924472282|
#|   computers.desktop|  7013| 0.4225238403940745|
#|appliances.kitche...|  6542|0.39414672235249315|
#|electronics.audio...|  6478| 0.3902908082237008|
#|appliances.enviro...|  6156|0.37089074026321434|
#+--------------------+------+-------------------+
#only showing top 20 rows


"""

6) MARCAS DOS PRODUTOS MAIS VISUALIZADOS, ADICIONADOS NO CARRINHO E COMPRADOS
    
"""

df1 = df.select('brand').filter(df['event_type'] == 'view').groupby('brand').count().orderBy('count', ascending = False)
df1.withColumn('% count', df1['count'] / 104335509 * 100).show()    

#+--------+--------+------------------+
#|   brand|   count|           % count|
#+--------+--------+------------------+
#|    null|14922708|14.302616763004433|
#| samsung|11898628|11.404197970606536|
#|   apple| 9374247| 8.984713919400154|
#|  xiaomi| 7232401| 6.931869187507391|
#|  huawei| 2358235|2.2602420044742386|
#| lucente| 1775749|1.7019603556062586|
#|      lg| 1574848|1.5094075019080992|
#|   bosch| 1480771|1.4192397336174398|
#|    oppo| 1203440|1.1534328164345276|
#|    sony| 1193071|1.1434946850165844|
#|    acer| 1084065|1.0390182694177492|
#|  lenovo| 1030106|0.9873014564964647|
#| respect| 1011896|0.9698481463295492|
#|cordiant|  947097|0.9077417737042909|
#|   artel|  946693|0.9073545613315597|
#|      hp|  796697|0.7635914250439896|
#| redmond|  709878|0.6803800612119504|
#| philips|  705559|0.6762405309203026|
#| indesit|  683316|0.6549218061513459|
#|dauscher|  680259|0.6519918353012492|
#+--------+--------+------------------+
#only showing top 20 rows

df1 = df.select('brand').filter(df['event_type'] == 'cart').groupby('brand').count().orderBy('count', ascending = False)
df1.withColumn('% count', df1['count'] / 3955446 * 100).show()    

#+--------+------+------------------+
#|   brand| count|           % count|
#+--------+------+------------------+
#| samsung|900469|22.765296252306314|
#|   apple|698749|17.665492083572875|
#|  xiaomi|364516|  9.21554737443009|
#|    null|277048| 7.004216465096477|
#|  huawei|115892| 2.929935081909853|
#|cordiant| 65317|1.6513182078582287|
#|    oppo| 65174|1.6477029391881473|
#|      lg| 62940|1.5912238468177797|
#|    sony| 44992|1.1374697063238886|
#|   artel| 40400|1.0213766032958105|
#| lucente| 39050|0.9872464445223118|
#|   bosch| 37663|0.9521808665824284|
#|    acer| 29780| 0.752886020944288|
#|  nokian| 27892|0.7051543618595729|
#| indesit| 25437|0.6430880360899883|
#|  lenovo| 24262|0.6133821571574988|
#|   vitek| 23957|0.6056712694345973|
#| philips| 23293|0.5888842876378543|
#| redmond| 22767|0.5755861665157356|
#|  viatti| 21384| 0.540621714972218|
#+--------+------+------------------+
#only showing top 20 rows

df1 = df.select('brand').filter(df['event_type'] == 'purchase').groupby('brand').count().orderBy('count', ascending = False)
df1.withColumn('% count', df1['count'] / 1659788 * 100).show()    

#+--------+------+------------------+
#|   brand| count|           % count|
#+--------+------+------------------+
#| samsung|372923| 22.46811038518172|
#|   apple|308937|18.613039737605046|
#|    null|131487| 7.921915328945624|
#|  xiaomi|124908|7.5255394062374235|
#|  huawei| 47204| 2.843977664617409|
#|cordiant| 27534|1.6588865565963846|
#| lucente| 26137| 1.574719181003839|
#|    oppo| 25971| 1.564717903732284|
#|      lg| 21606| 1.301732510416993|
#|    sony| 17038| 1.026516639474439|
#|   artel| 15391|0.9272870993162982|
#|   bosch| 13715|0.8263103480685485|
#|    acer| 13284|0.8003431763574625|
#|  lenovo| 11125|0.6702663231689829|
#|  nokian| 10888|0.6559873911607988|
#|elenberg| 10845|0.6533966988555165|
#|triangle| 10576|0.6371898097829362|
#| indesit| 10211|0.6151990495171672|
#| philips| 10002|0.6026070799403298|
#|   vitek|  9530| 0.574169713240486|
#+--------+------+------------------+
#only showing top 20 rows


"""

7) MAIORES E MENORES PREÇOS PAGOS PARA CADA MARCA DE PRODUTOS MAIS COMPRADOS
    
"""

df_brands = df.select('brand').filter(df['event_type'] == 'purchase').groupby('brand').count().orderBy('count', ascending = False)
df_expensiver_price = df.select('brand','price').filter(df['event_type'] == 'purchase').groupby('brand').max().orderBy('max(price)', ascending = False)
df_cheaper_price = df.select('brand','price').filter(df['event_type'] == 'purchase').groupby('brand').min().orderBy('min(price)', ascending = False)

df_brands.join(df_expensiver_price, 'brand', 'left')\
    .join(df_cheaper_price, 'brand', 'left')\
        .select('brand','max(price)','min(price)').orderBy('count', ascending = False).show()

#+--------+----------+----------+
#|   brand|max(price)|min(price)|
#+--------+----------+----------+
#| samsung|   2574.04|      1.26|
#|   apple|   2574.04|      4.61|
#|    null|      null|      null|
#|  xiaomi|   2033.51|      1.29|
#|  huawei|    965.02|      4.09|
#|cordiant|    448.37|     26.51|
#| lucente|    971.97|      9.52|
#|    oppo|    952.38|    115.54|
#|      lg|   2574.04|      1.45|
#|    sony|   2574.04|      4.61|
#|   artel|   1714.07|     17.86|
#|   bosch|   1490.09|      2.24|
#|    acer|   2574.04|     12.07|
#|  lenovo|   2335.37|      2.24|
#|  nokian|    384.82|     26.94|
#|elenberg|    934.39|      4.35|
#|triangle|    319.18|     28.06|
#| indesit|     595.1|     16.19|
#| philips|   1698.81|      3.32|
#|   vitek|    231.64|      5.12|
#+--------+----------+----------+
#only showing top 20 rows

"""

8) HORÁRIOS QUE TIVERAM MAIOR NÚMEROS DE VISUALIZAÇÕES E COMPRAS DE PRODUTOS
    
"""

df = df.withColumn('date', df['event_time'].substr(0, 10))\
    .withColumn('time_hours', df['event_time'].substr(11, 3)).drop('event_time')

df.select('time_hours').filter(df['event_type'] == 'view').groupby('time_hours').count().orderBy('count', ascending = False).show(30)

#+----------+-------+
#|time_hours|  count|
#+----------+-------+
#|        16|7270950|
#|        15|7117965|
#|        17|6882199|
#|        14|6693970|
#|        13|5944610|
#|        08|5701665|
#|        09|5624290|
#|        07|5555772|
#|        10|5552230|
#|        06|5468082|
#|        11|5421757|
#|        12|5398504|
#|        05|5234702|
#|        18|5180650|
#|        04|4692790|
#|        03|3740363|
#|        19|3575999|
#|        02|2632243|
#|        20|2089835|
#|        01|1350441|
#|        21|1212983|
#|        00| 729308|
#|        22| 722995|
#|        23| 541206|
#+----------+-------+


df.select('time_hours').filter(df['event_type'] == 'purchase').groupby('time_hours').count().orderBy('count', ascending = False).show(30)

#+----------+------+
#|time_hours| count|
#+----------+------+
#|        09|126617|
#|        10|120946|
#|        08|120458|
#|        07|112111|
#|        11|111582|
#|        06|109432|
#|        12|103145|
#|        05|101432|
#|        13| 98809|
#|        14| 96837|
#|        15| 90231|
#|        04| 85235|
#|        16| 81993|
#|        17| 76215|
#|        03| 55970|
#|        18| 47967|
#|        19| 33811|
#|        02| 25124|
#|        20| 20045|
#|        21| 12597|
#|        01|  9741|
#|        22|  8126|
#|        00|  5771|
#|        23|  5593|
#+----------+------+
