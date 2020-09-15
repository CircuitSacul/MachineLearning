import pandas as pd
from ast import literal_eval

from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import download_model, download_bnpp_data
from cdqa.pipeline.cdqa_sklearn import QAPipeline

# Download data and models
#download_bnpp_data(dir='./data/bnpp_newsroom_v1.1/')

if input('Download model? Only do if you haven\'t already.').lower().startswith('y'):
    download_model(model='bert-squad_1.1', dir='./models')

#df = pd.read_csv('data/my_data/homework.csv', converters={'paragraphs': literal_eval})
#df = filter_paragraphs(df)

df = pd.DataFrame(columns=['title', 'paragraphs'])
paragraphs = input("Text to Analyze:\n").split('\n')
df = df.append({'title': 'Inputed Data', 'paragraphs': paragraphs}, ignore_index=True)

print(df)

cdqa_pipeline = QAPipeline(reader='models/bert_qa.joblib', min_df=1, max_df=1000)

cdqa_pipeline.fit_retriever(df=df)

while True:
    query = input('> ')
    prediction = cdqa_pipeline.predict(query=query)

    #if prediction[3] < -2:
    #    print("cdQA: Sorry, I don't know.")
    #else:
        #print('query: {}\n'.format(query))
    print('cdQA: {}'.format(prediction[0]))
        #print('title: {}\n'.format(prediction[1]))
        #print('paragraph: {}\n'.format(prediction[2]))
        #print('confidence: {}'.format(prediction[3]))