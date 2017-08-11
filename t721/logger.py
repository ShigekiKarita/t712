import logging


default_level = logging.DEBUG
default_name = "t721"

logger = logging.getLogger(default_name)
logger.setLevel(default_level)

fh = logging.FileHandler("/tmp/t721.log")
fh.setLevel(default_level)
ch = logging.StreamHandler()
ch.setLevel(default_level)
formatter = logging.Formatter(
    '[%(name)s.%(module)s.%(funcName)s/%(filename)s:L%(lineno)d] %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def get_logger(level=default_level, name=default_name):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
