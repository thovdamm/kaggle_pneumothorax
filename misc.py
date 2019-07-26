import os

def new_logdir(log_path):
    try:
        n = max([int(i) for i in os.listdir(log_path)])+1
        path = os.path.join(log_path,f'{n:0.03d}')
    except:
        path = os.path.join(log_path,'000')
    os.mkdir(path)
    return path

