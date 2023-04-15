from flask import Flask
from flask_bootstrap import Bootstrap5
from flask_session import Session
import logging
import psutil
import time
print('Importing SentenceTransformer. %.3f%% memory usage' % psutil.virtual_memory().percent)
_t0 = time.time()
from sentence_transformers import SentenceTransformer
_t1 = time.time()
print('\tTook %.3f seconds ' % (_t1-_t0))
print('Loading Specific SentenceTransformer. %.3f%% memory usage' % psutil.virtual_memory().percent)
_t0 = time.time()
HansardSentenceTransformer = SentenceTransformer('sentence-transformers/average_word_embeddings_komninos')
_t1 = time.time()
print('\tTook %.3f seconds ' % (_t1-_t0))
#HansardSentenceTransformer = SentenceTransformer('nreimers/MiniLM-L6-H384-uncased')
print('Importing bertopic. %.3f%% memory usage' % psutil.virtual_memory().percent)
_t0 = time.time()
from bertopic import BERTopic 
_t1 = time.time()
print('\tTook %.3f seconds ' % (_t1-_t0))
print('Importing UMAP. %.3f%% memory usage' % psutil.virtual_memory().percent)
_t0 = time.time()
from umap import UMAP
_t1 = time.time()
print('\tTook %.3f seconds ' % (_t1-_t0))
print('Importing nltk ngrams and WordNetLemmatizer. %.3f%% memory usage' % psutil.virtual_memory().percent)
_t0 = time.time()
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
_ = lemmatizer.lemmatize('cat') #Force it to do initial setup work now
_t1 = time.time()
print('\tTook %.3f seconds ' % (_t1-_t0))
    
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s : %(message)s", datefmt='%m/%d/%Y %I/%M:%S %p')
log = logging.getLogger(__file__)


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.logger.info('Starting create_app')
    app.config.from_mapping(
        SECRET_KEY = 'dev',
    )

    
    app.logger.info('Importing main')
       
    from . main import bp
    app.logger.info('Registering Main')
    app.register_blueprint(bp)

    app.logger.info('Bootstrap5')
    bootstrap = Bootstrap5(app)
    app.logger.info('Completed create_app')
    app.config["SESSION_PERMANENT"] = False
    app.config["SESSION_TYPE"] = "filesystem"
    Session(app)
    app.logger.info('Completed session_app')
    app.jinja_env.cache = {}
    return app
