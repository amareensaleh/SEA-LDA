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
import pyLDAvis
import pyLDAvis.gensim_models as ldvis
import pandas as pd
import random
import gensim
from gensim.models import EnsembleLda

# create English stop words list

stemmer = SnowballStemmer("english")

my_stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
                'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such',
                'into', 'however', 'every', 'like', 'want', 'fine', 'one', 'two', 'make', 'thing', 'every', 'able'
                'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each',
                'the', 'work', 'set', 'get', 'similar', 'change', 'must', 'above', 'both', 'need',
                'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
                'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'could', 'would', 'our', 'their', 'while',
                'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and',
                'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',
                'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too',
                'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my',
                'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'xa', 'use']

stemmed_stopwords = []

for i in my_stopwords:
    stemmed_stopwords.append(stemmer.stem(i))

domain_terms = ['graphql', 'apollo', 'application', 'app', 'service', 'code', 'gql', 'data', 'object', 'project',
                'schema', 'return', 'name', 'run', 'implement', 'call', 'api', 'file', 'write', 'follow', 'new',
                'update', 'generate', 'class', 'user']

stemmed_domain_terms = []
for i in domain_terms:
    stemmed_domain_terms.append(stemmer.stem(i))



def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def replace_bigram(texts):
    bigram = gensim.models.Phrases(texts, min_count=20, threshold=10)
    mod = [bigram[sent] for sent in texts]
    return mod


def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def remove_stopwords(tokens):
    stopped_tokens = [i for i in tokens if not i in stemmed_stopwords]
    return stopped_tokens


def remove_domainterms(tokens):
    newtokens = [i for i in tokens if not i in stemmed_domain_terms]  # remove domain terms
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
    tokens = stem_tokens(tokens)
    tokens = remove_stopwords(tokens)
    tokens = remove_domainterms(tokens)
    return tokens


def get_random_number():
    return random.randint(0, 50000)


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
        self.use_multicore = use_multicore
        self.target_coherence = coherence
        self.workers = core
        self.iterations = iterations

        if (training_data is None):
            self.training_data = self.read_data_from_oracle()
        else:
            self.training_data = training_data

        self.model = self.create_model_from_training_data()

        print(len(self.model.ttda))
        print(len(self.model.get_topics()))

        #
        # NUMBER_OF_TOPICS = [12]
        # NUMBER_OF_ITERATIONS = [1000]
        # highest_coherence = 0
        # best_topics = -1
        # best_iterations = -1
        #
        # for aNumberOfTopics in NUMBER_OF_TOPICS:
        #     for aNumberOfIterations in NUMBER_OF_ITERATIONS:
        #
        #         print("----------------------------------------------------------------\nAttempting with #ofTopics: "
        #               + str(aNumberOfTopics) + " and #ofIterations: " + str(aNumberOfIterations))
        #
        #         self.model = self.prepare_model(aNumberOfTopics, aNumberOfIterations)
        #         coherence = self.compute_coherence()
        #         print("Coherence score: " + str(coherence))
        #         print("Previous highest coherence score: " + str(highest_coherence))
        #
        #         if coherence > highest_coherence:
        #             highest_coherence = coherence
        #             best_topics = aNumberOfTopics
        #             best_iterations = aNumberOfIterations
        #             print("Best #ofTopics became: " + str(aNumberOfTopics))
        #             print("Best #ofTopics iterations became: " + str(aNumberOfIterations))
        #
        # print("***************************************************************")
        # print("Best #ofTopics: " + str(best_topics))
        # print("Best #ofTopics iterations: " + str(best_iterations))
        # print("Best coherence: " + str(highest_coherence))

    def get_model(self):
        return self.model

    def visualize(self):
        lda_display = ldvis.prepare(self.model, self.corpus, self.dictionary)
        pyLDAvis.save_html(lda_display, self.fileprefix + ".html")
        #pyLDAvis.display(lda_display)

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

        return self.prepare_model()

    def prepare_model(self):
        if (self.use_multicore):
            print('LDA MultiCore')
            # ldamodel = LdaMulticore(self.corpus,
            #                         num_topics=self.num_topics,
            #                         id2word=self.dictionary,
            #                         passes=50,
            #                         workers=self.workers,
            #                         alpha='symmetric',
            #                         random_state=get_random_number(),
            #                         eta='auto',
            #                         iterations=self.iterations)
            return EnsembleLda(
                epsilon=0.35,
                min_samples=5,
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=10,
            num_models=8,
            topic_model_class='ldamulticore',
            ensemble_workers=self.workers,
            distance_workers=self.workers)

        else:
            ldamodel = LMSingle(corpus=self.corpus,
                                num_topics=self.num_topics,
                                id2word=self.dictionary,
                                random_state=get_random_number(),
                                passes=50,
                                alpha='auto',
                                eta='auto',
                                iterations=self.iterations)
            return ldamodel

    def compute_coherence(self):
        coherencemodel = CoherenceModel(model=self.model, dictionary=self.dictionary, texts=self.token_collection,
                                        topn=10,
                                        coherence='c_v')
        value = coherencemodel.get_coherence()
        return value

    def read_data_from_oracle(self):
        workbook = open_workbook(self.fileprefix + "-posts.xls")
        sheet = workbook.sheet_by_index(0)
        model_data = []
        print("Reading data from oracle..")
        for cell_num in range(1, sheet.nrows):
            id = sheet.cell(cell_num, 0).value
            postType = sheet.cell(cell_num, 1).value
            body = None
            if (self.fileprefix.__eq__("graphql")):
                body = sheet.cell(cell_num, 6).value
            else:
                body = sheet.cell(cell_num, 3).value
            comments = LDADocument(id, postType, body)
            model_data.append(comments)
        return model_data

    def view_dominant_topics(self):
        df_topic_sents_keywords = self.format_topics_sentences()

        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Original_id']

        # Show
        df_dominant_topic.to_csv(self.fileprefix + "-document-to-topic.csv")

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
    parser.add_argument('--file', type=str, help='File prefix', default="bitcoin")
    parser.add_argument('--multicore', type=bool, help='Is Multicore', default=True)
    parser.add_argument('--topic', type=int, help='Number of Topics', default=10)
    parser.add_argument('--coherence', type=float, help='Target coherence', default=0.6)
    parser.add_argument('--core', type=int, help='CPU Threads', default=24)
    parser.add_argument('--iteration', type=int, help='Number of iterations', default=100)

    args = parser.parse_args()
    print("args: " + args.__str__())

    project = args.file
    multi_core = args.multicore
    topics = args.topic
    t_coherence = args.coherence
    num_core = args.core
    iter = args.iteration

    mymodel = SEALDAModel(num_topics=topics, fileprefix=project, use_multicore=multi_core, coherence=t_coherence,
                          core=num_core, iterations=iter)

    mymodel.print_topics()
    print(mymodel.compute_coherence())
    mymodel.view_dominant_topics()
    mymodel.visualize()
