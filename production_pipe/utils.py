import os
from datetime import date

log_save_path = '/Workspace/Users/609399@mgmresorts.com/Tier_Imminent/production_pipe/log'

def write_log(message):
    file_path = log_save_path+f'/{date.today().strftime("%Y-%m-%d")}.txt'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as log_file:
        log_file.write(f"{message}\n")