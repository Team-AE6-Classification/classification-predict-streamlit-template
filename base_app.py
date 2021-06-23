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

def cleaner(line):

    # Removes RT, url and trailing white spaces
    line = re.sub(r'^RT ','', re.sub(r'https://t.co/\w+', '', line).strip()) 

    # Removes puctuation
    punctuation = re.compile("[.;:!\'’‘“”?,\"()\[\]]")
    tweet = punctuation.sub("", line.lower()) 

    # Removes stopwords
    nlp_for_stopwords = NLP(replace_words=True, remove_stopwords=True, remove_numbers=True, remove_punctuations=False) 
    tweet = nlp_for_stopwords.process(tweet) # This will remove stops words that are not necessary. The idea is to keep words like [is, not, was]
    # https://towardsdatascience.com/why-you-should-avoid-removing-stopwords-aa7a353d2a52

    # tokenisation
    # We used the split method instead of the word_tokenise library because our tweet is already clean at this point
    # and the twitter data is not complicated
    tweet = tweet.split() 

    # POS 
    pos = pos_tag(tweet)


    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tweet = ' '.join([lemmatizer.lemmatize(word, po[0].lower()) if po[0].lower() in ['n', 'r', 'v', 'a'] else word for word, po in pos])
    # tweet = ' '.join([lemmatizer.lemmatize(word, 'v') for word in tweet])

    return tweet


# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Loading raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Climate Change Tweet Classifer")
    st.subheader("Climate Change  Belief Analysis Based on Tweets")
 
    # Creating sidebar with selection purpose
    # you can create multiple pages this way
    pages = ["Information", "EDA/Visuals", "Prediction", "App Developers Contacts"]
    selection = st.sidebar.selectbox("Go to page", pages)

    ##creating a sidebar for selection purposes
    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        st.write('Explorers Explore !!!!!!!!!')
        # You can read a markdown file from supporting resources folder
        st.markdown("""Group AE6 has   deployed Machine Learning Algorithms that are able to classify twitter sentiments, based on novel tweet data. 
        Like any data lovers, these are robust solutions to that can provide access to a 
        broad base of consumer sentiment, spanning multiple demographic and geographic categories. 
        So, do you have a Twitter API and ready to scrap? or just have some tweets off the top of your head? 
        Do explore the rest of this app's buttons.""")

    st.subheader("Climate Change  Belief Analysis Based on Tweets")
    if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page

    # Building out the EDA page
    if selection == "EDA/Visuals":	
        st.info("Graphs and charts created from the raw data. Some of the text is too long and may cut off, feel free to right click on the chart and either save it or open it in a new window to see it properly")	

        # Number of Messages Per Sentiment
        st.write('Distribution of the sentiments')
        # Labeling the target
        raw['sentiment'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1] for x in raw['sentiment']]
        
        # checking the distribution
        st.write('The numerical proportion of the sentiments')
        values = raw['sentiment'].value_counts()/raw.shape[0]
        labels = (raw['sentiment'].value_counts()/raw.shape[0]).index
        colors = ['lightgreen', 'blue', 'purple', 'lightsteelblue']
        plt.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=90, explode= (0.04, 0, 0, 0), colors=colors)
        st.pyplot()
        
        # checking the distribution
        sns.countplot(x='sentiment' ,data = raw, palette='PRGn')
        plt.ylabel('Count')
        plt.xlabel('Sentiment')
        plt.title('Number of Messages Per Sentiment')
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
    if selection == 'Prediction':

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
            ### SINGLE TWEET CLASSIFICATION ###
            st.subheader('Single tweet classification')

            input_text = st.text_area('Enter Text (max. 140 characters):') ##user entering a single text to classify and predict
            all_ml_models = ["LR","NB","RFOREST","DECISION_TREE"]
            model_choice = st.selectbox("Choose ML Model",all_ml_models)

            st.info('for more information on the above ML Models please visit: https://datakeen.co/en/8-machine-learning-algorithms-explained-in-human-language/')


            prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
            if st.button('Classify'):

                st.text("Original test ::\n{}".format(input_text))
                text1 = cleaner(input_text) ###passing the text through the 'cleaner' function
                vect_text = tweet_cv.transform([text1]).toarray()
                if model_choice == 'LR':
                    predictor = load_prediction_models("resources/Logistic_regression.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'RFOREST':
                    predictor = load_prediction_models("resources/RFOREST_model.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'NB':
                    predictor = load_prediction_models("resources/NB_model.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'DECISION_TREE':
                    predictor = load_prediction_models("resources/DTrees_model.pkl")
                    prediction = predictor.predict(vect_text)
				# st.write(prediction)

                final_result = get_keys(prediction,prediction_labels)
                st.success("Tweet Categorized as:: {}".format(final_result))

        if source_selection == 'Dataset':
            ### DATASET CLASSIFICATION ###
            st.subheader('Dataset tweet classification')

            all_ml_models = ["LR","NB","RFOREST","SupportVectorMachine", "MLR", "LDA"]
            model_choice = st.selectbox("Choose ML Model",all_ml_models)

            st.info('for more information on the above ML Models please visit: https://datakeen.co/en/8-machine-learning-algorithms-explained-in-human-language/')


            prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
            text_input = st.file_uploader("Choose a CSV file", type="csv")
            if text_input is not None:
                text_input = pd.read_csv(text_input)

            #X = text_input.drop(columns='tweetid', axis = 1, inplace = True)   

            uploaded_dataset = st.checkbox('See uploaded dataset')
            if uploaded_dataset:
                st.dataframe(text_input.head(25))

            col = st.text_area('Enter column to classify')

            #col_list = list(text_input[col])

            #low_col[item.lower() for item in tweet]
            #X = text_input[col]

            #col_class = text_input[col]
            
            if st.button('Classify'):

                st.text("Original test ::\n{}".format(text_input))
                X1 = text_input[col].apply(cleaner) ###passing the text through the 'cleaner' function
                vect_text = tweet_cv.transform([X1]).toarray()
                if model_choice == 'LR':
                    predictor = load_prediction_models("resources/Logistic_regression.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'RFOREST':
                    predictor = load_prediction_models("resources/Random_model.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'NB':
                    predictor = load_prediction_models("resources/NB_model.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'SupportVectorMachine':
                    predictor = load_prediction_models("resources/svm_model.pkl")
                    prediction = predictor.predict(vect_text)

                elif model_choice == 'MLR':
                    predictor = load_prediction_models("resources/mlr_model.pkl")
                    prediction = predictor.predict(vect_text)

                elif model_choice == 'SupportVectorMachine':
                    predictor = load_prediction_models("resources/simple_lda_model.pkl")
                    prediction = predictor.predict(vect_text)

                
				# st.write(prediction)
                text_input['sentiment'] = prediction
                final_result = get_keys(prediction,prediction_labels)
                st.success("Tweets Categorized as:: {}".format(final_result))

                
                csv = text_input.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

                st.markdown(href, unsafe_allow_html=True)


 
    ##contact page
    if selection == 'App Developers Contacts':

        st.info('Contact details in case you any query or would like to know more of our designs:')
        st.write('Khuliso Muleka: khuliso.muleka@gmail.com')
        st.write('Shanice Pillay: pillay.shanice18@gmail.com')
        st.write('Seromo Podile: seromopodile@gmail.com')
        st.write('Sbusiso Phakhade: phaks323@gmail.com')
        st.write('Ofentse Makeketlane')
        st.write('Maureen Matshitisho: maureen.mashitisho@gmail.com')
	
        # Footer 
        image = Image.open('resources/imgs/EDSA_logo.png')

        st.image(image, caption='Team-SS4-Johannesbrug', use_column_width=True)




# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
