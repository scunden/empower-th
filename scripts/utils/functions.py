import logging
import sys


def create_logger():
    # Create a logger object
    logger = logging.getLogger('stdout_logger')
    logger.setLevel(logging.DEBUG)  # Set the logging level
    
    if not logger.hasHandlers():

    # Create a handler that writes to stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)  # Set the handler level

        # Create a formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)
    return logger