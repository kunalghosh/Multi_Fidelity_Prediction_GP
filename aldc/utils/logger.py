import os
import logging



class Logger
    def __init__(self, file_name:str, app_name:str, log_folder="."):
        super(Logger, self).__init__()
        self.file_name = file_name
        self.app_name = app_name
        self.log_folder = log_folder
        self.logger = None
        self._init_logger()

    def _init_logger(self):
        logger = logging.getLogger(self.app_name)
        logger.setLevel(logging.DEBUG)

        global fh
        global ch

        if fh is not None:
            fh.flush()
            fh.close()
            print("Flushed and closed file handler.")

        fh = logging.FileHandler(logfolder + os.sep + fname)
        fh.setLevel(logging.DEBUG)

        if ch is  None:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            print("Created new stream Handler.")

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        self.logger = logger

    def log(message:str):
        logger.info(message)
