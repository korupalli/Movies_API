import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import re

df = pd.read_csv('movies_metadata.csv')

credits = pd.read_csv('credits.csv')

#dropping unnecessary columns
df.drop(['belongs_to_collection','homepage','poster_path','tagline','overview','status','title'],axis=1,inplace=True)

df = df.query("adult == 'False'").drop(['adult', 'video'], 'columns')

# df = df[~df['id'].str.contains('-')]
df['id'] = df['id'].apply(lambda x:int(x))

#converting release date to month and year
def func1(x):
    x = str(x)
    if x == 'nan':
        return (np.nan,np.nan)
    return tuple(x.split('-')[: 2])
df['year'], df['month'] = zip(*df['release_date'].map(func1))
df.drop(['release_date'],axis=1,inplace=True)

#converting zero values in butget and revenue to NaN
df['budget'] = df['budget'].apply(lambda x:x if x!=0 else np.nan)
df['revenue'] = df['revenue'].apply(lambda x:x if x!=0.0 else np.nan)

#getting movie genre
df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])




#calculating movie rating basing on imdb formula
vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()

#considering movies with a min m number of votes
m = vote_counts.quantile(0.95)

qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['original_title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False)

#print(qualified['original_title'].head(15))

#replicating movies with genre
s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = df.drop('genres', axis=1).join(s)

def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['original_title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified

build_chart('Romance').head(15)

links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

df = df[df['id'].isin(links_small)]
#credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

df = df.merge(credits, on='id')
df = df.merge(keywords, on='id')

df['cast'] = df['cast'].apply(literal_eval)
df['keywords'] = df['keywords'].apply(literal_eval)

df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['cast'] = df['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)

df['keywords'] = df['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


def func(x):
    string = [i for i in x.split('}') if 'Directing' in i]
    if len(string) ==0:
        return 'Unknown'
    #print(string)
    string = string[0]
    if 'name\': \'' in string:
        name = string[re.search('name\': \'',string).span()[1]:].split('\'')[0]
        return re.sub('\s+',' ',re.sub('\W',' ',name))
    else:
        name = string[re.search('name',string).span()[1]+5:].split('\'')[0]
        return re.sub('\s+',' ',re.sub('\W',' ',name))
df['Director'] = df['crew'].apply(func)


#converting all to lower case
# df['cast'] = df['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

#replicating director name to give it more emphasis
# df['Director'] = df['Director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
df['Director'] = df['Director'].apply(lambda x: [x,x, x])



#obtaining keyword counts
s = df.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s = s[s > 1]

stemmer = SnowballStemmer('english')

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

#reducing the keyword for uniformity
df['keywords'] = df['keywords'].apply(filter_keywords)
df['keywords'] = df['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
# df['keywords'] = df['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


df['soup'] = df['keywords'] + df['cast'] + df['Director'] + df['genres']
df['soup'] = df['soup'].apply(lambda x: ' '.join(x))

count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(df['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

df = df.reset_index()
titles = df['original_title']
indices = pd.Series(df.index, index=df['original_title'])


def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = df.iloc[movie_indices][['original_title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified
#print(improved_recommendations('The Dark Knight'))


df.drop(['index', 'id'], 'columns', inplace = True)


import json
from flask import Flask, request
from flask_restplus import Resource, Api, fields, inputs, reqparse


app = Flask(__name__)
api = Api(app)

film_model = api.model('Movie', {'budget': fields.Integer,
                                 'genres': fields.List(fields.String),
                                 'imdb_id': fields.String,
                                 'original_language': fields.String,
                                 'original_title': fields.String,
                                 'popularity': fields.Float,
                                 'production_companies': fields.List(fields.String),
                                 'production_countries': fields.List(fields.String),
                                 'revenue': fields.Integer,
                                 'runtime': fields.Integer,
                                 'spoken_languages': fields.List(fields.String),
                                 'vote_average': fields.Float,
                                 'vote_count': fields.Integer,
                                 'year': fields.String,
                                 'month': fields.String,
                                 'cast': fields.List(fields.String),
                                 'crew': fields.List(fields.String),
                                 'keywords': fields.List(fields.String),
                                 'Director': fields.List(fields.String),
                                 'soup': fields.String
                                }
                      )

parser = reqparse.RequestParser()
parser.add_argument('order', choices = list(column for column in film_model.keys()))
parser.add_argument('ascending', type = inputs.boolean)

@api.route('/movies')
class MovieList(Resource):
    def get(self):
        args = parser.parse_args()
        order_by = args.get('order')
        ascending = args.get('ascending', True)
        if order_by:
            df.sort_values(by = order_by, inplace = True, ascending = ascending)
        ds = json.loads(df.to_json(orient = 'index'))
        return [ds[idx] for idx in ds]

    @api.expect(film_model, validate = True)
    def post(self):
        movie = request.json
        id = max(df.index) + 1
        for key in movie:
            if key not in film_model.keys():
                return {'message': 'Property {} is invalid'.format(key)}, 400
            df.at[id, key] = movie[key]
        return {'message': 'Movie {} has been created'.format(id)}, 201

@api.route('/movies/<int:id>')
class Movies(Resource):
    def get(self, id):
        if id not in df.index:
            api.abort(404, "Movie {} doesn't exist".format(id))
        return dict(df.loc[id])
    
    def delete(self, id):
        if id not in df.index:
            api.abort(404, "Movie {} doesn't exist".format(id))
        df.drop(id, inplace = True)
        return {'message': 'Movie {} has been removed.'.format(id)}, 200
    
    @api.expect(film_model)
    def put(self, id):
        if id not in df.index:
            api.abort(404, "Movie {} doesn't exist".format(id))
        movie = request.json
        for key in movie:
            if key not in film_model.keys():
                return {'message': 'Property {} is invalid'.format(key)}, 400
            df.at[id, key] = movie[key]
        return {'message': 'Movie {} has been successfully updated'.format(id)}, 200



if __name__ == '__main__':
    app.run(debug=True)








