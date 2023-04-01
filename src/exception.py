import sys
import logging
import logger

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="error occured file [{0}, line np. [{1}] , error message [{2}]]".format(file_name,exc_tb.tb_lineno,str(error))
    return error_message




class CustomException(Exception):
    def __init__(self,error,error_detail:sys):
        self.error=error_message_detail(error,error_detail)
    def __str__(self):
        return logging.error(self.error)


