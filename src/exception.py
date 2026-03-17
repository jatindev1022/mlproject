import sys 

def error_message_detail(error, error_detail: sys):
    _, _, exe_tb = error_detail.exc_info()
    
    file_name = exe_tb.tb_frame.f_code.co_filename if exe_tb else "Unknown"
    line_number = exe_tb.tb_lineno if exe_tb else "Unknown"

    error_message = 'Error occurred in python script name [{0}] line [{1}] error message [{2}]'.format(
        file_name,
        line_number,
        str(error)
    )

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message