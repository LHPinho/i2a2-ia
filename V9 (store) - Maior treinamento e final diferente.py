import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from IPython.display import display
from prophet import Prophet
import joblib
import csv
import os

def generate_timeframe(start_date, end_date):
    # Gerar a faixa de datas
    dates = pd.date_range(start=start_date, end=end_date)
    
    # Criar arrays para Store e Item
    stores = np.arange(1, 11)
    items = np.arange(1, 51)
    
    # Gerar todas as combinações possíveis de Date, Store e Item
    combinations = pd.MultiIndex.from_product([items, stores, dates], names=["item", "store", "ds"]).to_frame(index=False) #ds é a data
    
    # Adicionar coluna 'id'
    combinations['id'] = np.arange(len(combinations))

    # Adicionar coluna 'Forecast'
    combinations['forecast'] = None
    
    return combinations[['id', 'ds', 'store', 'item', 'forecast']]


# Exemplo de uso
start_date = '2018-01-01'
end_date = '2018-03-31'
timeframe_df = generate_timeframe(start_date, end_date)
print(timeframe_df)

#Guardar em arquivo
timeframe_df.to_csv(f'Predicoes\\timeframe.csv', index=False)

dict_timeframe_df = timeframe_df.to_dict(orient='records')




dados = pd.read_csv('demand-forecasting-kernels-only\\train.csv')
dados_pred = pd.read_csv('demand-forecasting-kernels-only\\test.csv')
#display(dados)

# Dividindo o dataset para treino e teste
#split_date = '2016-12-31'
dados_treino = dados.copy()
#dados_treino = dados.loc[dados.date <= split_date].copy()
#dados_teste = dados.loc[dados.date > split_date].copy()

#display(dados_treino)
#display(dados_teste)


#Printa os tipos de dados de cada coluna do dataset
#print(dados_treino.dtypes)
#print(dados_teste.dtypes)

# Prophet model expects the dataset to be named a specific way. We will rename our dataframe columns before feeding it into the model.
# Datetime column named: ds
# target : y

#Transforma o tipo de dado da coluna "date" de object para datetime, necessário para o Prophet
dados_treino['date'] = pd.to_datetime(dados_treino['date'])
#dados_teste['date'] = pd.to_datetime(dados_teste['date'])
dados_pred['date'] = pd.to_datetime(dados_pred['date'])


#Printa os tipos de dados de cada coluna do Data Frame (atualizado)
#print(dados_treino.dtypes)
#print(dados_teste.dtypes)


#Renomeando as colunas do dataset dados_treino e do dataset dados_teste, necessário para o Prophet
dados_treino = dados_treino.reset_index() \
    .rename(columns={'date':'ds', 'sales':'y'})
#dados_teste = dados_teste.reset_index() \
#    .rename(columns={'date':'ds', 'sales':'y'})
dados_pred = dados_pred.reset_index() \
    .rename(columns={'date':'ds'})


dados_pred['yhat'] = None #Adicionar coluna de predicao


#display(dados_treino)
#display(dados_teste)

# Definindo funções de treinar e salvar e função de carregar o modelo
def train_and_save_model(df, model_path='Modelos\\prophet_model.joblib'):
    model = Prophet(yearly_seasonality=True)
    model.fit(df)
    joblib.dump(model, model_path)
    return model

def load_model(model_path='Modelos\\prophet_model.joblib'):
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print("Modelo carregado com sucesso.")
            return model
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            return None
    else:
        print("Arquivo de modelo não encontrado.")
        return None


items = dados_treino['item'].unique()
stores = dados_treino['store'].unique()


for i in stores: #in items

    # Selecionando somente os dados relativos a store i para treinar o modelo
    dados_treino_store_i = dados_treino.loc[(dados_treino.store == i)].copy()
    if ((i == 1) or (i % 5 == 0)):
        print(f"Dados de treino da store {i:02}:")
        display(dados_treino_store_i)

    # Caminho e nome arquivo onde o modelo será salvo/carregado
    model_path = f'Modelos\\store{i:02}.joblib'

    # Carregar o modelo se ele já estiver salvo, caso contrário treinar e salvar
    model = load_model(model_path)
    if model is None:
        print(f"Treinando o modelo para a store {i:02}...")
        model = train_and_save_model(dados_treino_store_i, model_path)

    # Prevendo o Futuro e armazenando essas previsões em arquivos para cada store i
    #future = model.make_future_dataframe(periods=90, freq='D', include_history=True)
    forecast_path = f'Predicoes\\pred_store_{i:02}.csv'
    if not os.path.exists(forecast_path):
        print(f"Arquivo de predicao do store {i:02} sera gerado")
        dates = pd.date_range(start=start_date, end=end_date)
        future = pd.DataFrame(dates, columns=['ds'])
        forecast = model.predict(future)
        forecast['store'] = i #adicionando a coluna do store ao forecast
        forecast.to_csv(forecast_path, index=False)
    else:
        print(f"Arquivo de predicao da store {i:02} ja existe!")



    # Escrevendo previsões no arquivo consolidado
    # Abra o arquivo CSV em modo de leitura


    with open(f'Predicoes\\pred_store_{i:02}.csv', 'r') as pred_store_csv:

        # Crie um leitor de CSV com cabeçalho
        leitor_pred_store_csv = csv.DictReader(pred_store_csv)

        #pred_store_csv['ds'] = pd.to_datetime(pred_store_csv['ds'])           

        for linha_csv in leitor_pred_store_csv: # Iterando sobre as linhas do arquivo CSV
            for linha_timeframe_df in dict_timeframe_df:
                linha_csv['ds'] = pd.to_datetime(linha_csv['ds'])
                if (linha_timeframe_df['ds'] == linha_csv['ds']) and (linha_timeframe_df['store'] == i):
                    #print('linha_timeframe_df:')
                    #print(linha_timeframe_df['forecast'])
                    #print('linha_csv[yhat]:')
                    #print(linha_csv['yhat'])
                    #print('linha_timeframe_df[forecast] antes:')
                    #print(linha_timeframe_df['forecast'])
                    linha_timeframe_df['forecast'] = linha_csv['yhat']
                    #print('linha_timeframe_df[forecast] depois:')
                    #print(linha_timeframe_df['forecast'])

        #print("linha_timeframe_df['ds']:")
        #print(linha_timeframe_df['ds'])
        #print("linha_csv[ds]:")
        #print(linha_csv['ds'])
        #print("Tipo linha_timeframe_df['ds']")
        #print(type(linha_timeframe_df['ds']))
        #print("Tipo linha_csv['ds']")
        #print(type(linha_csv['ds']))

        """
        for linha_pred in dados_pred:
            #Linha_pred
            print(linha_pred) #linha_pred['yhat']
            print("Próxima")
            print(type(dados_pred))
        """


    """
        # Itere sobre as linhas do arquivo CSV
        for linha_csv in leitor_csv:
            # Cada linha é um dicionário onde as chaves são os nomes das colunas
            print(linha_csv['yhat'])
     """

#display(dict_timeframe_df)

timeframe_df_selecionado = pd.DataFrame(dict_timeframe_df)[['id', 'forecast']]
timeframe_df_selecionado = timeframe_df_selecionado.rename(columns={'forecast': 'sales'})
timeframe_df_selecionado.to_csv('submission.csv', index=False)


"""
dates = pd.date_range(start='2018-01-01', end='2018-03-31')
future = pd.DataFrame(dates, columns=['ds'])
display(future)
forecast = model.predict(future)
display(forecast)
forecast.to_csv(f'Predicoes\\pred.csv', index=False)
"""