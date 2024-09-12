import logging
import sys


def create_logger(level='debug'):
    # Create a logger object
    logger = logging.getLogger('stdout_logger')
    if level=='info':
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
    
    if not logger.hasHandlers():

    # Create a handler that writes to stdout
        console_handler = logging.StreamHandler(sys.stdout)
        if level=='info':
            console_handler.setLevel(logging.INFO)
        else:
            console_handler.setLevel(logging.DEBUG)

        # Create a formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)
    return logger