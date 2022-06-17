from datetime import datetime

time_format = '%Y-%m-%d-%H-%M-%S'

def current_time_str():
    time = datetime.now()
    time_str = time.strftime(time_format)
    return time_str

def time_of_str(s):
    time = datetime.strptime(s, time_format)
    return time

def get_latest_str_from_str_time_list(strs):
    times = []
    for s in strs:
        try:
            time = time_of_str(s)
            times.append(time)
        except:
            continue
    times.sort()

    if len(times) == 0:
        print("ERROR! No valid extrinsic scenes")
        exit(-1)

    return times[-1].strftime(time_format)