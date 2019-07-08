# Run in python console
# Download stopwords from NLTK for text pre-processing
import nltk; nltk.download('stopwords')
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
from pprint import pprint
import pandas
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# spacy for lemmatization
import spacy
import os
import logging
import warnings
# NLTK Stop words
from nltk.corpus import stopwords
import json

os.environ.update({'MALLET_HOME':r'C:/Users/Administrator/Desktop/NLP-l1/mallet-2.0.8/'})

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
# Enable logging for gensim - optional

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


warnings.filterwarnings("ignore",category=DeprecationWarning)


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

###########################################################################
###########################################################################

class TopicModelBugs:

    """
    Generate a topic model on project's methods, then find the methods that most related to bugs

    Parameters:
    ----------
    project_name : the name of the project
    project_path : path to json file that contain the commits for the project
    min_amount_topics: the minimum amount of topics for the model
    max_amount_topics: the maximum amount of topics for the model
    """
    def __init__(self, project_name,project_path, min_amount_topics, max_amount_topics ):
        self.project_name = project_name
        self.project_path = project_path
        self.min_amoumt_topics = min_amount_topics
        self.max_amount_topics = max_amount_topics


    def createTheModel(self):
        with open(self.project_path +'.json', 'r') as f:

            data2 = json.load(f)
            values = []

            for k, v in data2.items():
                values.append(v)

        data = values
        bugs =[]
        filepath = self.project_name +"_bugs.txt"
        with open(filepath,encoding="utf-8" ) as fp:
           line = fp.readline()
           bugs.append(line)
           while line:
               line = fp.readline()
               bugs.append(line)

        data.extend(bugs)
        # Remove new line characters
        data = [re.sub('\s+', ' ', sent) for sent in data]

        # Remove distracting single quotes
        data = [re.sub("\'", "", sent) for sent in data]


        # Tokenize words and Clean-up text
        def sent_to_words(sentences):
            for sentence in sentences:
                yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

        data_words = list(sent_to_words(data))





        #  Creating Bigram and Trigram Models
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # See trigram example
        print(trigram_mod[bigram_mod[data_words[0]]])


        # Remove Stopwords, Make Bigrams and Lemmatize
        # Define functions for stopwords, bigrams, trigrams and lemmatization
        def remove_stopwords(texts):
            return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

        def make_bigrams(texts):
            return [bigram_mod[doc] for doc in texts]

        def make_trigrams(texts):
            return [trigram_mod[bigram_mod[doc]] for doc in texts]

        def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
            """https://spacy.io/api/annotation"""
            texts_out = []
            for sent in texts:
                doc = nlp(" ".join(sent))
                texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            return texts_out

        # Remove Stop Words
        data_words_nostops = remove_stopwords(data_words)

        # Form Bigrams
        data_words_bigrams = make_bigrams(data_words_nostops)

        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        #python3 -m spacy download en
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        # Do lemmatization keeping only noun, adj, vb, adv
        # Lemmatization is converting a word to its root word.
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        print(data_lemmatized[:1])


        #  Create the Dictionary and Corpus needed for Topic Modeling
        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized
        # Create the dictionary and corpus needed for topic modeling
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]


        # Human readable format of corpus (term-frequency)
        # Gensim creates a unique id for each word in the document. The produced corpus shown above is a mapping of (word_id, word_frequency).
        [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

        number_topics =self.min_amoumt_topics


        mallet_path = 'C:\\Users\\Administrator\\Desktop\\NLP-l1\\mallet-2.0.8\\bin\\mallet' # update this path

        #Trains multiple LDA models and provides the models and their corresponding coherence scores.
        def compute_coherence_values(dictionary, corpus, texts, limit, start=5, step=1):
            """
            Compute c_v coherence for various number of topics

            Parameters:
            ----------
            dictionary : Gensim dictionary
            corpus : Gensim corpus
            texts : List of input texts
            limit : Max num of topics

            Returns:
            -------
            model_list : List of LDA topic models
            coherence_values : Coherence values corresponding to the LDA model with respective number of topics
            """
            coherence_values = []
            model_list = []
            for num_topics in range(start, limit, step):
                model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
                model_list.append(model)
                coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                coherence_values.append(coherencemodel.get_coherence())

            return model_list, coherence_values

        # Can take a long time to run.
        model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=self.min_amoumt_topics, limit=self.max_amount_topics, step=1)
        # Show graph
        # Begin with 5 topics, and each step, increas the topic by 1.
        # Stop when the number of topic is 40
        limit=self.max_amount_topics; start=self.min_amoumt_topics; step=1;
        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')

        #Save the coherence score diagram
        plt.savefig(self.project_name + '.png')

#
        counter = 0
        max_coh = -100
        max_model = 0
        for m, cv in zip(x, coherence_values):
            print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

            if max_coh < float(round(cv,4)):
                max_coh = float(round(cv,4))
                max_model = counter
            counter = counter + 1
        # Select the model  with maximum coherence and print the topics
        optimal_model = model_list[max_model]
        model_topics = optimal_model.show_topics(formatted=False)
        pprint(optimal_model.print_topics(num_words=10))
        # write the topics of the created model
        f = open(self.project_name + "_topics_list.csv" ,"w")
        for top, valu in model_topics:
            f.write(str(top))
            f.write(",")
            f.write(str(valu))
            f.write("\n")
        f.close()



        def format_topics_sentences(ldamodel, corpus=corpus, texts=data):
            # Init output
            sent_topics_df = pd.DataFrame()

            # Get main topic in each document
            for i, row in enumerate(ldamodel[corpus]):
                row = sorted(row, key=lambda x: (x[1]), reverse=True)
                topic_keywords = ", ".join([str(word)+" "+str(prop) for word, prop in row])
                sent_topics_df = sent_topics_df.append(pd.Series([topic_keywords]),
                                                       ignore_index=True)
            sent_topics_df.columns = ['Topics']

            # Add original text to the end of the output
            contents = pd.Series(texts)
            sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
            return(sent_topics_df)


        df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)
        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Topics', 'Text']

        #Write the topics for each document
        df_dominant_topic.to_csv(self.project_name+"methods_topics.csv" , encoding='utf-8')

        ###################################Handle Bugs###############################

        #Clean the bugs description from stop words etc.
        data = bugs
        # Remove new line characters
        data = [re.sub('\s+', ' ', sent) for sent in data]

        # Remove distracting single quotes
        data = [re.sub("\'", "", sent) for sent in data]

        data_words = list(sent_to_words(data))


        # #  Creating Bigram and Trigram Models
        # # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

        # # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # # Remove Stop Words
        data_words_nostops = remove_stopwords(data_words)

        # # Form Bigrams
        data_words_bigrams = make_bigrams(data_words_nostops)

        # # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        # #
        # # # Create Corpus
        # # texts = data_lemmatized

        df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Bug_No', 'Topics', 'Text']

        #Write the topics for each bugs
        df_dominant_topic.to_csv(self.project_name+"_bugs_topics.csv" , encoding='utf-8')


t = TopicModelBugs("lang", "methods_lang_clean",5, 40)
t.createTheModel()