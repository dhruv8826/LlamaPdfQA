import logging

filename = None


def set_log_filename(file_loc):
    global filename
    filename = file_loc


# Create and configure logger
logging.basicConfig(filename=filename,
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Set logging level
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def log_update(message, level=0):
    if level == 0:
        pre_post: str = ""
    else:
        a = int(10 / level)
        pre_post = ' ' + ('-' * a) + ' '
    final_message = pre_post + message + pre_post
    # adding to logs
    logger.info(final_message)
    # printing on console
    # print(final_message)
