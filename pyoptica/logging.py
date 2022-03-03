import logging

LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class LoggerMixin(object):
    """A mixin class to add logger. It is expected to be used in all classes
    developed withing pyoptica. It is expected to be used only internally!

    **Example**

    >>> import pyoptica.logging as pol
    >>>
    >>> class MyClass(pol.LoggerMixin):
    >>>     pass

    >>> my_instance = MyClass()
    >>> my_instance.logger.info('I am printing a log.')
    yyy-mm-dd hh:mm:ss,sss - module_name.MyClass - INFO - I am printing a log.
    """

    def __init_subclass__(cls, **kwargs):
        name = '.'.join([cls.__module__, cls.__name__])
        logger = get_standard_logger(name)
        cls.logger = logger


def get_standard_logger(name, level=None):
    """ Gets the logger with provided name` with default values for the project

    :param name: name of the logger
    :type name: str
    :param level: level of logging (default logging.DEBUG)
    :type level: int
    :return: logger
    :rtype: logging.Logger

    **Example**

    >>> import pyoptica.logging as pol
    >>>
    >>> pol.get_standard_logger('my_name')
    <Logger my_name (DEBUG)>
    """
    level = level or LOGGING_LEVEL
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter(LOGGING_FORMAT)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
