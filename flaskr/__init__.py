import os
from flask import Flask
from flask_bootstrap import Bootstrap5
import logging

def create_app(test_config=None):
    logging.basicConfig(level=logging.INFO)
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY = 'dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )
    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
        
    from . import main
    app.register_blueprint(main.bp)

    bootstrap = Bootstrap5(app)
    app.logger.info('Completed create_app')
    return app