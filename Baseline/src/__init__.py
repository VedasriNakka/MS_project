from __future__ import absolute_import, division, print_function

__version__ = "0.3.0"


def _setup_logger():
    import logging

    logging_fmt = "%(asctime)s (%(module)s:%(lineno)d)" "%(levelname)s: %(message)s"
    logger = logging.getLogger("main")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(logging_fmt))
    logger.addHandler(handler)
    # print("HERE!")
    logger.propagate = True


_setup_logger()
