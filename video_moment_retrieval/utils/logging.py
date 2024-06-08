import logging
import sys
import json



LOG_FORMAT =  "%(asctime)s - %(levelname)s %(name)s - %(filename)s:%(lineno)d - %(message)s"
logger = logging.getLogger("video_moment_retrieval")

def init_logging(level: int = logging.DEBUG, root_level: int = logging.ERROR):
    global logger
    root_logger = logging.getLogger("root")
    loggers : list[logging.Logger] = [logger, root_logger]
    levels: list[int] = [level, root_level]
    for logger_, level_ in zip(loggers, levels):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setLevel(level_)
        handler.setFormatter(formatter)
        logger_.handlers = [handler]
        logger_.setLevel(level_)
    
    logger.propagate = False