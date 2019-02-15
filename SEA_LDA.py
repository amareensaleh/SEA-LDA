import warnings

warnings.simplefilter("ignore")
import html

import re
import nltk
from xlrd import open_workbook
from nltk.stem.snowball import SnowballStemmer
from gensim import corpora, models
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LdaModel as LMSingle
from gensim.models.coherencemodel import CoherenceModel
import argparse
import pyLDAvis.gensim as ldvis
import pyLDAvis
import pandas as pd
import  random
import  gensim

# create English stop words list

my_stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
                'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such',
                'into',
                'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each',
                'the',
                'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
                'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'could', 'would', 'our', 'their', 'while',
                'above', 'both',
                'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and',
                'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',
                'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too',
                'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my',
                'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'xa', 'use']

domain_terms = ['monero', 'bitcoin', 'blockchain', 'ethereum', 'xmr', 'btc', 'eth', 'block',
    'coin','bitcoins','blocks','ether','ethers']

stemmer = SnowballStemmer("english")

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def replace_bigram(texts):

    bigram=gensim.models.Phrases(texts,min_count=20, threshold=10)
    #print(bigram.vocab.items())
    mod=[ bigram[sent] for sent in texts]
    return mod



def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def remove_stopwords(tokens):
    stopped_tokens = [i for i in tokens if not i in my_stopwords]
    return stopped_tokens


def remove_domainterms(tokens):
    newtokens = [i for i in tokens if not i in domain_terms]  # remove domain terms
    return newtokens


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [w.lower() for w in tokens if w.isalpha() == True or '_' in w]  # lower case, remove number, punctuation
    return tokens


url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def remove_url(s):
    return url_regex.sub(" ", s)


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext = html.unescape(cleantext)
    return cleantext


def cleanup_text(text):
    # comments=text
    text = str(text).encode('ascii', 'ignore').decode('ascii')
    text = cleanhtml(text)  # clean html
    text = remove_url(text)  # remove url
    return text


def preprocess_text(text):
    # comments=text
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = remove_domainterms(tokens)
    stems = stem_tokens(tokens)
    return  stems

def get_random_number():
    return random.randint(0,50000)

class LDADocument:
    def __init__(self, id, posttype, body):
        self.id = id
        self.posttype = posttype
        self.body = body


class SEALDAModel:
    def __init__(self, training_data=None, num_topics=10, fileprefix='bitcoin',
                 use_multicore=True, coherence=0.6, core=24, iterations=100):
        self.num_topics = num_topics
        self.fileprefix = fileprefix
        self.use_multicore=use_multicore
        self.target_coherence=coherence
        self.workers=core
        self.iterations=iterations


        if (training_data is None):
            self.training_data = self.read_data_from_oracle()
        else:
            self.training_data = training_data

        self.model = self.create_model_from_training_data()
        coherence=self.compute_coherence()
        while(coherence<self.target_coherence):
            print("Random seed: " + str(self.seed))
            print("Coherence score: "+str(coherence))
            self.model=self.prepare_model(self.num_topics)
            coherence=self.compute_coherence()

    def get_model(self):
        return self.model

    def visualize(self):
        lda_display = ldvis.prepare(self.model, self.corpus, self.dictionary)

        pyLDAvis.save_html(lda_display, self.fileprefix + ".html")
        pyLDAvis.display(lda_display)

    def print_topics(self, count=5):
        print(self.model.print_topics(num_topics=20, num_words=10))

    def create_model_from_training_data(self):
        training_documents = []
        document_ids = []
        print("Training classifier model..")
        for document in self.training_data:
            doc = cleanup_text(document.body)
            training_documents.append(doc)
            document_ids.append(document.id)
        self.document_ids = document_ids


        doc_collection = []
        for text in training_documents:
            collection = preprocess_text(text)
            doc_collection.append(collection)

        self.token_collection = replace_bigram(doc_collection)

        self.dictionary = corpora.Dictionary(self.token_collection)
        self.dictionary.filter_extremes(no_below=20, no_above=0.2, keep_n=20000)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.token_collection]
        return self.prepare_model(self.num_topics)

    def prepare_model(self, topics_count):
        self.seed = get_random_number()
        if(self.use_multicore):
            ldamodel = LdaMulticore(self.corpus, num_topics=topics_count, id2word=self.dictionary,
                                passes=50, workers=self.workers, alpha='symmetric', random_state=self.seed,
                                    eta='auto', iterations=self.iterations)
            return ldamodel
        else:
            ldamodel = LMSingle(corpus =self.corpus, num_topics=topics_count, id2word=self.dictionary,
                                random_state=self.seed, passes=50, alpha='auto', eta='auto', iterations=self.iterations)
            return ldamodel

    def compute_coherence(self):
        coherencemodel = CoherenceModel(model=self.model, dictionary=self.dictionary, texts=self.token_collection, topn=10,
                                            coherence='c_v')
        value = coherencemodel.get_coherence()
        return value

    def read_data_from_oracle(self):
        workbook = open_workbook(self.fileprefix+"-posts.xlsx")
        sheet = workbook.sheet_by_index(0)
        model_data = []
        print("Reading data from oracle..")
        for cell_num in range(1, sheet.nrows):
            comments = LDADocument(sheet.cell(cell_num, 0).value, sheet.cell(cell_num, 1).value,
                                   sheet.cell(cell_num, 3).value)
            model_data.append(comments)
        return model_data

    def view_dominant_topics(self):
        df_topic_sents_keywords = self.format_topics_sentences()

        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Original_id']

        # Show
        df_dominant_topic.to_csv(self.fileprefix+"-document-to-topic.csv")


    def format_topics_sentences(self):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(self.model[self.corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original ids to the end of the output
        contents = pd.Series(self.document_ids)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return (sent_topics_df)

    def get_topic(self, text):
        comment = preprocess_text(text)
        feature_vector = self.vectorizer.transform([comment]).toarray()
        sentiment_class = self.model.predict(feature_vector)
        return sentiment_class

    def get_topic_collection(self, texts):
        predictions = []
        for text in texts:
            comment = preprocess_text(text)
            feature_vector = self.vectorizer.transform([comment]).toarray()
            sentiment_class = self.model.predict(feature_vector)
            predictions.append(sentiment_class)

        return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LDA Model')

    parser.add_argument('--file', type=str,
                        help='File prefix', default="bitcoin")

    parser.add_argument('--multicore', type=bool,
                        help='Iteration count', default=True)

    parser.add_argument('--topic', type=int,
                        help='Iteration count', default=10)
    parser.add_argument('--coherence', type=float,
                        help='Target coherence', default=0.6)
    parser.add_argument('--core', type=int,
                        help='CPU Threads', default=24)
    parser.add_argument('--iteration', type=int,
                        help='Number of iterations', default=100)

    args = parser.parse_args()
    project = args.file
    multi_core=args.multicore
    topics = args.topic
    t_coherence=args.coherence
    num_core=args.core
    iter=args.iteration

    mymodel = SEALDAModel(num_topics=topics, fileprefix=project, use_multicore=multi_core,
                          coherence=t_coherence, core=num_core, iterations=iter)

    mymodel.print_topics()
    print(mymodel.compute_coherence())
    mymodel.view_dominant_topics()
    mymodel.visualize()
