# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import ensemble
import dill as pickle

def input_predict(pred_model,genre, critics=0, duration=0, df=0,a3=0, a1=0, content_rating=0, budget=0, a2=0, gross=0):
    input = [critics, duration, df, a3, a1, content_rating, budget, a2, gross] + [0]*20
    genre = genre.lower().split(',')
    genre_labels = {'action': 0, 'adventure': 1, 'animation': 2, 'biography': 3, 'comedy': 4, 'crime': 5, 
                    'documentary': 6, 'drama': 7, 'family': 8, 'fantasy': 9, 'film-noir': 10, 'game-show': 11, 
                    'history': 12, 'horror': 13, 'music': 14, 'musical': 15, 'mystery': 16, 'romance': 17, 'sci-fi': 18, 
                    'thriller': 19, 'western': 20}
    input[10] = genre_labels[genre[0]]
    pred_genres = ['action', 'adventure', 'fantasy', 'sci-fi', 'thriller', 'romance', 'animation', 'comedy', 'family', 'musical', 
                     'mystery', 'drama', 'history', 'sport', 'crime', 'horror', 'biography', 'music', 'documentary']
    for i in range(len(pred_genres)):
        if pred_genres[i] in genre:
            input[i+10] = 1
    return round(pred_model.predict(pd.DataFrame(input).T)[0],1)


filename = 'model2.pk'
with open(filename ,'rb') as f:
    loaded_model = pickle.load(f)

print(input_predict(loaded_model,'thriller',2000,2.5))