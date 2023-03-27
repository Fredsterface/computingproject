import os
from flask import Flask
from flask_bootstrap import Bootstrap5
from flask import session
from flask_session import Session
import logging
import psutil
print('Importing SentenceTransformer. %.3f%% memory usage' % psutil.virtual_memory().percent)
from sentence_transformers import SentenceTransformer
print('Loading Specific SentenceTransformer. %.3f%% memory usage' % psutil.virtual_memory().percent)
HansardSentenceTransformer = SentenceTransformer('sentence-transformers/average_word_embeddings_komninos')
print('Importing bertopic. %.3f%% memory usage' % psutil.virtual_memory().percent)
from bertopic import BERTopic 
print('Importing UMAP. %.3f%% memory usage' % psutil.virtual_memory().percent)
from umap import UMAP
print('Importing nltk ngrams. %.3f%% memory usage' % psutil.virtual_memory().percent)
from nltk import ngrams


class LogFilter(logging.Filter):
    def filer(self, record):
        record.memory = psutil.virtual_memory().percent
        return True
    
logging.basicConfig(level=logging.INFO, format="%(asctime)s ?? memory %(levelname)s : %(message)s", datefmt='%m/%d/%Y %I/%M:%S %p')
log = logging.getLogger(__file__)
log.addFilter(LogFilter())




def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.logger.addFilter(LogFilter())
    logging.getLogger('werkzeug').addFilter(LogFilter())
    app.logger.info('Starting create_app')
    app.config.from_mapping(
        SECRET_KEY = 'dev',
    )
#    if test_config is None:
#        app.logger.info('test_config is None')
#        app.config.from_pyfile('config.py', silent=True)
#    else:
#        app.logger.info('test_config from mapping')
#        app.config.from_mapping(test_config)
#    try:
#        app.logger.info('Trying to make dirs')
#        os.makedirs(app.instance_path)
#        app.logger.info('Completed make dirs')
#    except OSError:
#        pass

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

    return app
