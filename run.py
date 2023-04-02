import argparse
from flaskr import create_app

import argparse

# create an argument parser object
parser = argparse.ArgumentParser(description='Specify a port number')
# add an argument for the port number
parser.add_argument('--port', type=int, default=8081, help='the port number to use')
# add an argument for the debug flag
parser.add_argument('--debug', action='store_true', help='enable debug mode')


app = create_app()

if __name__ == '__main__':
    # parse the arguments from the command line
    args = parser.parse_args()
    # access the port number entered by the user (or the default value)
    port_number = args.port
    # access the debug flag entered by the user (or False by default)
    debug_mode = args.debug
    if debug_mode:
        app.run(debug=True, use_reloader=True, port=port_number)
    else:
        app.run(host='0.0.0.0', port=port_number, threaded=True)
