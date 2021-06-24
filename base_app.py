"""

    Simple Streamlit webserver application for serving developed classification
    models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/
"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
#from nlppreprocess import NLP # pip install nlppreprocess
#import en_core_web_sm
from nltk import pos_tag

import seaborn as sns
import re

from nlppreprocess import NLP
nlp = NLP()

def data_cleaning(tweet):
   
    # Convert to lowercase
    tweet = tweet.lower() 
    
    # Remove mentions or twitter handles   
    tweet = re.sub('@[\w]*','',tweet)  
    
    # Remove url's
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    
    # Remove hashtags
    tweet = re.sub(r'#\w*', '', tweet)    
    
    # Remove numbers
    tweet = re.sub(r'\d+', '', tweet)  
    
    # Remove punctuation
    tweet = re.sub(r"[,.;':@#?!\&/$]+\ *", ' ', tweet)                  
    
    return tweet


# Vectorizer
news_vectorizer = open("resources/cv.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) 

# Loading raw data
raw = pd.read_csv("resources/train.csv")

def main():
    """Tweet Classifier App with Streamlit """
    import streamlit as st
    st.title("Climate Change Tweet Classifer")
    st.subheader("Climate Change Belief Analysis Based on Tweets")
 
    # Creating sidebar with selection purpose
    # you can create multiple pages this way
    pages = ["About Project", "Data Visuals", "Predict Model", "About Us"]
    selection = st.sidebar.selectbox("Go to page", pages)

    col1, mid, col2 = st.beta_columns([1,4,20])
    with col1:
        st.image('resources/2.PNG', width=600)
    with col2:
        st.write(' ')

    if selection == "About Project":
        st.info("Info")
        st.markdown("""Group AE6 has deployed Machine Learning Algorithms that are able to classify twitter sentiments, based on novel tweet data. 
        This plotform provides accurate and robust sentimental analysis using Machine Learning models and Algorithms. The models classifies tweets or text based on consumer's sentiment towards climate change.
        The analysis offers organisations an access to consumer sentiment spanning from wide demographic and geographic categories.
        Do you have a compiled list of tweets or text that you would like to classify?
        Then this is the platform to explore and find out what the consumers think""")

    st.subheader("Climate Change  Belief Analysis Based on Tweets")
    if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page

    # Creating the EDA page
    if selection == "Data Visuals":	
        st.info("Graphs and charts created from the raw data. These graphs and charts show the distributions of tweets sentiments")	

        # Number of Messages Per Sentiment
        st.write('Distribution of the sentiments')
        # Labeling the target
        raw['sentiment'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1] for x in raw['sentiment']]
        
        # checking the distribution
        st.write('Distribution')
        sns.countplot(x='sentiment' ,data = raw, palette='PRGn')
        plt.ylabel('Count')
        plt.xlabel('Sentiment')
        plt.title('Dataset labels distribution')
        st.pyplot()



        #Create a violinplot of the dataset
        st.write('Violin')
        raw['message_length'] = raw['message'].apply(len)
        plt.figure(figsize=(8,5)) #Set the figsize to 8 and 5 respectively
        plt.title('Sentiments vs. Length of tweets') #Add the title of the violin plot
        sns.violinplot(x='sentiment', y='message_length', data=raw,scale='count') #Add the dimentions of the violin plot
        plt.ylabel("Length of the tweets") #Y_lable of the plot
        plt.xlabel("Sentiment Class") 
        st.pyplot()


        # checking the distribution
        st.write('Dataset labels distribution')
        values = raw['sentiment'].value_counts()/raw.shape[0]
        labels = (raw['sentiment'].value_counts()/raw.shape[0]).index
        colors = ['lightgreen', 'blue', 'purple', 'lightsteelblue']
        plt.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=90, explode= (0.04, 0, 0, 0), colors=colors)
        st.pyplot()

        # Popular Tags
        st.write('Popular tags found in the tweets')
        raw['users'] = [''.join(re.findall(r'@\w{,}', line)) if '@' in line else np.nan for line in raw.message]
        sns.countplot(y="users", hue="sentiment", data=raw,
                    order=raw.users.value_counts().iloc[:20].index, palette='PRGn') 
        plt.ylabel('User')
        plt.xlabel('Number of Tags')
        plt.title('Top 20 Most Popular Tags')
        st.pyplot()

        # Tweet lengths
        st.write('The length of the sentiments')
        st.write('The average Length of Messages in all Sentiments is 100 which is of no surprise as tweets have a limit of 140 characters.')

        # Repeated tags
        
        # Generating Counts of users
        st.write("Analysis of hashtags in the messages")
        counts = raw[['message', 'users']].groupby('users', as_index=False).count().sort_values(by='message', ascending=False)
        values = [sum(np.array(counts['message']) == 1)/len(counts['message']), sum(np.array(counts['message']) != 1)/len(counts['message'])]
        labels = ['First Time Tags', 'Repeated Tags']
        colors = ['lightsteelblue', "purple"]
        plt.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=90, explode= (0.04, 0), colors=colors)
        st.pyplot()

        # Popular hashtags
        st.write("The Amount of popular hashtags")
        repeated_tags_rate = round(sum(np.array(counts['message']) > 1)*100/len(counts['message']), 1)
        print(f"{repeated_tags_rate} percent of the data are from repeated tags")
        sns.countplot(y="users", hue="sentiment", data=raw, palette='PRGn',
              order=raw.users.value_counts().iloc[:20].index) 
        plt.ylabel('User')
        plt.xlabel('Number of Tags')
        plt.title('Top 20 Most Popular Tags')
        st.pyplot()

        st.markdown("Now that we've had a look at the tweets themselves as well as the users, we now analyse the hastags:")

        # Generating graphs for the tags
        st.write('Analysis of most popular tags, sorted by populariy')
        # Analysis of most popular tags, sorted by populariy
        sns.countplot(x="users", data=raw[raw['sentiment'] == 'Positive'],
                    order=raw[raw['sentiment'] == 'Positive'].users.value_counts().iloc[:20].index) 

        plt.xlabel('User')
        plt.ylabel('Number of Tags')
        plt.title('Top 20 Positive Tags')
        plt.xticks(rotation=85)
        st.pyplot()

        # Analysis of most popular tags, sorted by populariy
        st.write("Analysis of most popular tags, sorted by populariy")
        sns.countplot(x="users", data=raw[raw['sentiment'] == 'Negative'],
                    order=raw[raw['sentiment'] == 'Negative'].users.value_counts().iloc[:20].index) 

        plt.xlabel('User')
        plt.ylabel('Number of Tags')
        plt.title('Top 20 Negative Tags')
        plt.xticks(rotation=85)
        st.pyplot()


        st.write("Analysis of most popular tags, sorted by populariy")
        # Analysis of most popular tags, sorted by populariy
        sns.countplot(x="users", data=raw[raw['sentiment'] == 'News'],
                    order=raw[raw['sentiment'] == 'News'].users.value_counts().iloc[:20].index) 

        plt.xlabel('User')
        plt.ylabel('Number of Tags')
        plt.title('Top 20 News Tags')
        plt.xticks(rotation=85)
        st.pyplot()


    # Building out the predication page
    if selection == 'Predict Model':

        st.info('Make Predictions of your Tweet(s) using our ML Model')

        data_source = ['Select option', 'Single text', 'Dataset'] ## differentiating between a single text and a dataset inpit

        source_selection = st.selectbox('What to classify?', data_source)

        # Load Our Models
        def load_prediction_models(model_file):
            loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
            return loaded_models

        # Getting the predictions
        def get_keys(val,my_dict):
            for key,value in my_dict.items():
                if val == value:
                    return key


        if source_selection == 'Single text':
            ### Classifying one tweet ###
            st.subheader('Single tweet classification')

            input_text = st.text_area('Tweet Here:') 
            all_ml_models = ["Log-Regression","Log-Regression2","SVC-Linear1","SVC-Linear2","SVC-Non_linear2","SVC-Non_linear2"]
            model_choice = st.selectbox("Select a Model",all_ml_models)


            prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
            if st.button('Classify'):

                st.text("Original test ::\n{}".format(input_text))
                text1 = data_cleaning(input_text) 
                #y = raw[['sentiment']]
                vect_text = tweet_cv.transform([text1]).toarray()
                if model_choice == 'Log-Regression':
                    predictor = load_prediction_models("resources/model.pkl")
                    prediction = predictor.predict(vect_text)
                    st.write(prediction)
                elif model_choice == 'Log-Regression2':
                    predictor = load_prediction_models("resources/model1.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'SVC-Linear1':
                    predictor = load_prediction_models("resources/model2.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'SVC-linear2':
                    predictor = load_prediction_models("resources/model1.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'Non-Linear_SVC1':
                    predictor = load_prediction_models("resources/model1.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'Non-Linear_SVC2':
                    predictor = load_prediction_models("resources/model5.pkl")
                    prediction = predictor.predict(vect_text)
				# st.write(prediction)

                final_result = get_keys(prediction,prediction_labels)
                st.success("Tweet Categorized as:: {}".format(final_result))

        if source_selection == 'Dataset':
            ### DATASET CLASSIFICATION ###
            st.subheader('Dataset tweet classification')

            all_ml_models = ["Log-Regression","Log-Regression2","SVC-Linear1","SVC-Linear2", "Non-Linear_SVC1", "Non-Linear_SVC2"]
            model_choice = st.selectbox("Select a Model",all_ml_models)



            prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
            text_input = st.file_uploader("Choose a CSV file", type="csv")
            if text_input is not None:
                text_input = pd.read_csv(text_input)

            #X = text_input.drop(columns='tweetid', axis = 1, inplace = True)   

            uploaded_dataset = st.checkbox('See uploaded dataset')
            if uploaded_dataset:
                st.dataframe(text_input.head(25))

            col = st.text_area('Enter column to classify')

            
            if st.button('Classify'):

                st.text("Original test ::\n{}".format(text_input))
                X1 = text_input[col].apply(cleaner)
                vect_text = tweet_cv.transform([X1]).toarray()
                if model_choice == 'Log-Regression':
                    predictor = load_prediction_models("resources/model.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'Log-Regression2':
                    predictor = load_prediction_models("resources/model1.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'SVC-Linear1':
                    predictor = load_prediction_models("resources/model2.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'SVC-Linear2':
                    predictor = load_prediction_models("resources/model3.pkl")
                    prediction = predictor.predict(vect_text)

                elif model_choice == 'Non-Linear_SVC1':
                    predictor = load_prediction_models("resources/model4.pkl")
                    prediction = predictor.predict(vect_text)

                elif model_choice == 'Non-Linear_SVC2':
                    predictor = load_prediction_models("resources/model5.pkl")
                    prediction = predictor.predict(vect_text)

                
				# st.write(prediction)
                text_input['sentiment'] = prediction
                final_result = get_keys(prediction,prediction_labels)
                st.success("Tweets Categorized as:: {}".format(final_result))

                
                csv = text_input.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode() 
                href = f'<a href="data:file/csv;base64,{b64}">Upload csv file</a>'

                st.markdown(href, unsafe_allow_html=True)


 
    ##About Us
    if selection == 'About Us':

        st.info('Project Members')
        st.write('Khuliso Muleka: khuliso.muleka@gmail.com')
        st.write('Shanice Pillay: pillay.shanice18@gmail.com')
        st.write('Seromo Podile: seromopodile@gmail.com')
        st.write('Sbusiso Phakhade: phaks323@gmail.com')
        st.write('Ofentse Makeketlane')
        st.write('Maureen Matshitisho: maureen.mashitisho@gmail.com')
	
        # Footer 


if __name__ == '__main__':
    main()
