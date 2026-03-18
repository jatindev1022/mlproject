import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    _, _, exe_tb = error_detail.exc_info()
    
    file_name = exe_tb.tb_frame.f_code.co_filename if exe_tb else "Unknown"
    line_number = exe_tb.tb_lineno if exe_tb else "Unknown"

    return f"Error occurred in [{file_name}] line [{line_number}] message [{str(error)}]"


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


# if __name__ == "__main__":
#     try:
#         a = 1 / 0
#     except Exception as e:
#         logging.error("Exception occurred", exc_info=True)
#         raise CustomException(e, sys)