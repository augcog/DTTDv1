from datetime import datetime

time_format = '%Y-%m-%d-%H-%M-%S'

def current_time_str():
    time = datetime.now()
    time_str = time.strftime(time_format)
    return time_str

def time_of_str(s):
    time = datetime.strptime(s, time_format)
    return time
