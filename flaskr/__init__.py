import os
from flask import Flask
from flask_bootstrap import Bootstrap5

def create_app(test_config=None):
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
        
    #app.add_url_rule('/', endpoint='index')
    from . import dropdown
    app.register_blueprint(dropdown.bp)

    bootstrap = Bootstrap5(app)
    return app