import logging
import sys

def setup_logger(log_file='run.log', log_level=logging.DEBUG):
    """
    setup logger
    :param log_file: 日志文件路径
    :param log_level: 日志级别
    """
    # Create a basic logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='w'
    )
    
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Create a error logger
    error_logger = logging.getLogger('error_logger')
    error_logger.setLevel(logging.ERROR)
    error_logger.propagate = False  # Prevent double logging
    error_file_handler = logging.FileHandler('logs/error.log', mode='a', encoding='utf-8')
    error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    error_file_handler.setFormatter(error_formatter)
    if not error_logger.hasHandlers():
        error_logger.addHandler(error_file_handler)
    
    return logging.getLogger('')
