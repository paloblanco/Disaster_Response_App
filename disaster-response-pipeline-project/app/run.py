import json
from numpy import vectorize
import plotly
import pandas as pd
from typing import List

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords','omw-1.4'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap, Scatter
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

# this is defined globally so it does not need to be compiled repeatedly in a function
PATTERN_REMOVE_STOPWORDS = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')

def tokenize(text: str) -> List[str]:
    """
    Tokenizer for messages. Returns a list of relevant words. 
    Intended for usage in a sklearn pipeline.

    INPUTS:
        text: str - string corresponding to a message

    OUTPUTS:
        clean_tokens: List[str] - list of tokens (relevant words) in supplied text
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    text = PATTERN_REMOVE_STOPWORDS.sub('',text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for tok in word_tokenize(text):
        clean_tok = lemmatizer.lemmatize(tok)#.lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_classified', engine)
df_svd = pd.read_sql_table('messages_svd', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    category_counts = df.iloc[:,4:].sum() / len(df)
    category_names = list(df.columns)[4:]

    category_corr = df.corr()

    # cv = CountVectorizer(tokenizer=tokenize)
    # tf = TfidfTransformer()
    # X = tf.fit_transform(cv.fit_transform(df.message))
    # pca=TruncatedSVD(n_components=2)
    # xplot = pca.fit_transform(X)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    z=category_corr,
                    y=category_names,
                    x=category_names
                )
            ],

            'layout': {
                'title': 'Correlation of Message Categories',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Scatter(
                    x=df_svd['x'],
                    y=df_svd['y'],
                    mode='markers',
                    text=df_svd['message'] + r"<br>" + df_svd["labels"] 
                )
            ],

            'layout': {
                'title': 'First two principal components of comments ',
                'yaxis': {
                    'title': "Component 1"
                },
                'xaxis': {
                    'title': "Component 2"
                }
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()