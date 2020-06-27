import logging
import datetime


def get_logger(log_name="app", log_level="INFO", log_format=None, file_name=None):
    logging.basicConfig(level=log_level, format=log_format)

    if file_name is None:
        log_date = datetime.datetime.today().strftime("%Y%m%d")
        full_file_name = f"./log/app_{log_date}.log"
    else:
        full_file_name = file_name

    if log_format is None:
        log_format = "%(asctime)s\t[%(levelname)s] %(name)s:\t%(message)s"

    fh = logging.FileHandler(full_file_name)
    fh.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)

    logger = logging.getLogger(log_name)
    logger.addHandler(fh)

    return logger
