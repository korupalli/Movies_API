#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:36:47 2019

@author: apple
"""
import pandas as pd
from sklearn import ensemble
import dill as pickle
from collections import defaultdict
import requests
from flask import Flask, render_template,request,redirect,flash,url_for, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager,current_user,UserMixin
from flask_login import login_user, logout_user, login_required
import jwt
import datetime


app = Flask(__name__)
app.config['SECRET_KEY'] = 'thisismysecretkeydonotstealit'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']  = False


db=SQLAlchemy(app)

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))

def input_predict(pred_model,genre, critics=0, duration=0, df=0,a3=0, a1=0, content_rating=0, budget=0, a2=0, gross=0):
    duration=duration/60
    input = [critics, duration, df, a3, a1, content_rating, budget, a2, gross] + [0]*20
    genre = genre.lower().split(',')
    genre_labels = {'action': 0, 'adventure': 1, 'animation': 2, 'biography': 3, 'comedy': 4, 'crime': 5, 
                    'documentary': 6, 'drama': 7, 'family': 8, 'fantasy': 9, 'film-noir': 10, 'game-show': 11, 
                    'history': 12, 'horror': 13, 'music': 14, 'musical': 15, 'mystery': 16, 'romance': 17, 'sci-fi': 18, 
                    'thriller': 19, 'western': 20}
    if(genre[0] in genre_labels):
        input[10] = genre_labels[genre[0]]
    pred_genres = ['action', 'adventure', 'fantasy', 'sci-fi', 'thriller', 'romance', 'animation', 'comedy', 'family', 'musical', 
                     'mystery', 'drama', 'history', 'sport', 'crime', 'horror', 'biography', 'music', 'documentary']
    for i in range(len(pred_genres)):
        if pred_genres[i] in genre:
            input[i+10] = 1
    return round(pred_model.predict(pd.DataFrame(input).T)[0],1)



def get_movies_by_title(title):
    r = requests.get('http://127.0.0.1:5000/movies', {'order': 'original_title', 'ascending': True})
    print('Status Code:', str(r.status_code))
    movies = r.json()
    d=defaultdict(list)
    for movie in movies:
        e=[]
        if title in movie['original_title']:
            d['Movie Name'].append(movie['original_title'])
            d['Director'].append(movie['Director'][0])
            e=movie['genres']
            d['Genres'].append(', '.join([str(elem) for elem in sorted(e)]))
            d['Rating'].append(movie['vote_average'])
            d['Year of Release'].append(movie['year'])
    

    
    x1=d['Movie Name']
    x2=d['Director']
    x3=d['Genres']
    x4=d['Rating']
    x5=d['Year of Release']
    
   
    z1 = [x for _,x in sorted(zip(x4,x1),reverse=True)]
    z2 = [x for _,x in sorted(zip(x4,x2),reverse=True)]
    z3 = [x for _,x in sorted(zip(x4,x3),reverse=True)]
    z4 = [x for _,x in sorted(zip(x4,x4),reverse=True)]
    z5 = [x for _,x in sorted(zip(x4,x5),reverse=True)]
    
    
    
    d['Movie Name']=z1
    d['Director']=z2
    d['Genres']=z3
    d['Rating']=z4
    d['Year of Release']=z5
    
    return d
 
@login_manager.user_loader
def load_user(user_id):
        return User.query.get(int(user_id))

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']

        if not token:
            return jsonify({'message' : 'Token is missing!'}), 401

        try: 
            data = jwt.decode(token, app.config['SECRET_KEY'])
            current_user = User.query.filter_by(name=data['name']).first()
        except:
            return jsonify({'message' : 'Token is invalid!'}), 401

        return f(current_user, *args, **kwargs)

    return decorated

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False

    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password):
        flash('Please check your login details and try again.')
        return redirect(url_for('login'))
    token=jwt.encode({'name' : user.name, 'exp' : datetime.datetime.utcnow() + datetime.timedelta(minutes=30)}, app.config['SECRET_KEY'])
    
    login_user(user, remember=remember)
    return redirect(url_for('home'))

@app.route('/signup')
def signup():
    return render_template('signup.html')



#@app.route('/profile')
#@login_required
#def profile():
#    return render_template('profile.html', name=current_user.name)

@app.route('/signup', methods=['POST'])
def signup_post():
    email = request.form.get('email')
    name = request.form.get('name')
    password = request.form.get('password')

    user = User.query.filter_by(email=email).first()

    if user:
        flash('Email address already exists.')
        return redirect(url_for('signup'))

    new_user = User(email=email, name=name, password=generate_password_hash(password, method='sha256'))

    db.session.add(new_user)
    db.session.commit()

    return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

#app.secret_key='dont tell anyone.'
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/ratings/')
@login_required
def ratings():
    return render_template('ratings.html')
    
@app.route('/ratings/', methods=["GET","POST"])
@login_required
def getvalue():
    if request.method == "POST":
        
        
        num_critic_for_reviews=request.form.get('num_critic_for_reviews')
        duration=request.form.get('duration')
        director_facebook_likes=request.form.get('director_facebook_likes')
        actor_1_facebook_likes=request.form.get('actor_1_facebook_likes')
        actor_2_facebook_likes=request.form.get('actor_2_facebook_likes')
        actor_3_facebook_likes=request.form.get('actor_3_facebook_likes')
        #gross=request.form.get('gross')
        budget=request.form.get('budget')
        genre=request.form.get('genre')
        content_rating=request.form.get('content_rating')
        
        data={}
        data['Expected no. of Critic Reviews']=int(num_critic_for_reviews)
        data['Duration Time (minutes)']=int(duration)
        data['No. of Director FB likes']=int(director_facebook_likes)
        data['No. of Actor 1 FB likes']=int(actor_1_facebook_likes)
        data['No. of Actor 2 FB likes']=int(actor_2_facebook_likes)
        data['No. of Actor 3 FB likes']=int(actor_3_facebook_likes)
        #data['Revenue Generated (dollars)']=int(gross)
        data['Budget of Movie (dollars)']=int(budget)
        data['Genre (csv format)']=genre
        data['Content Rating']=int(content_rating)
        
     
        content_rating_list=['Approved','G','GP','M','NC-17','Not Rated','PG','PG-13','Passed','R','TV-14',
                             'TV-G','TV-MA','TV-PG','TV-Y','TV-Y7','Unrated','X']
        
        data1={}
        data1['Expected no. of Critic Reviews']=int(num_critic_for_reviews)
        data1['Duration Time (minutes)']=int(duration)
        data1['No. of Director FB likes']=int(director_facebook_likes)
        data1['No. of Actor 1 FB likes']=int(actor_1_facebook_likes)
        data1['No. of Actor 2 FB likes']=int(actor_2_facebook_likes)
        data1['No. of Actor 3 FB likes']=int(actor_3_facebook_likes)
        #data['Revenue Generated (dollars)']=int(gross)
        data1['Budget of Movie (dollars)']=int(budget)
        data1['Genre (csv format)']=genre
        data1['Content Rating']=content_rating_list[int(content_rating)]
        
        
        filename = 'model2.pk'
        with open(filename ,'rb') as f:
            loaded_model = pickle.load(f)
        #(pred_model,genre, critics=0, duration=0, df=0,a3=0, a1=0, content_rating=0, budget=0, a2=0, gross=0)
        
        x=input_predict(loaded_model,data['Genre (csv format)'],data['Expected no. of Critic Reviews'],data['Duration Time (minutes)'],
                      data['No. of Director FB likes'], data['No. of Actor 3 FB likes'],  data['No. of Actor 1 FB likes'],                 
                      data['Content Rating'],data['Budget of Movie (dollars)'],data['No. of Actor 2 FB likes'],
                      )
                                          
        #return genres
        if x!=None:
            #data['Movie Rating']=x
            return render_template('ratingfinal.html',result=data1,m_rating=x)
        else:
            return 'sorry'
    
    

@app.route('/moviedetail/')
@login_required
def moviedetail():
    return render_template('moviedetail.html')
    
@app.route('/moviedetail/', methods=["GET","POST"])
@login_required
def getmovievalue():
    if request.method == "POST":
        #genre=request.form.get('genre')
        moviename=request.form.get('moviename')
        
        if len(get_movies_by_title(moviename)['Movie Name'])>0:
            return render_template('moviedetails.html',result=get_movies_by_title(moviename),length=len(get_movies_by_title(moviename)['Movie Name']))
        else:
            return render_template('error.html',result=moviename)



            

#print(get_movies_by_title('Inception'))
            


if __name__ == '__main__':
    app.run(debug=True,host='127.0.0.1',port='5001')