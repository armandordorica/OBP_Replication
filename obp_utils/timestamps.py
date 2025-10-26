import datetime
from datetime import timedelta
import math

# def convert_unix_to_timestamp(input_unix_timestamp, gmt_zone_int=-5): 
#     date = datetime.datetime.utcfromtimestamp(timestamp_int / 1e3)
#     return (date + timedelta(hours=gmt_zone_int)).strftime('%d/%m/%Y %H:%M:%S')


def convert_unix_to_timestamp(input_unix_timestamp, gmt_zone_int=-5):
    if len(str(input_unix_timestamp))==10: 
        divisor = 1 
    if len(str(input_unix_timestamp))==13:
        divisor = 1e3
    if len(str(input_unix_timestamp))==19:
        divisor = 1e9
    date = datetime.datetime.utcfromtimestamp(input_unix_timestamp/divisor)
    # (date + timedelta(hours=gmt_zone_int)).strftime('%d/%m/%Y %H:%M:%S.%f')
    return (date).strftime('%Y-%m-%d %H:%M:%S.%f')



def get_duration_in_mins(x):
    return x.total_seconds()/60


def nanoseconds_to_seconds(nanoseconds):
    """
    Convert nanoseconds to seconds.

    Args:
        nanoseconds (int): The number of nanoseconds.

    Returns:
        float: The number of seconds.
    """
    return nanoseconds / 1_000_000_000


def milliseconds_to_human_readable(milliseconds):
    seconds = milliseconds / 1000
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return int(days), int(hours), int(minutes), int(seconds)


def milliseconds_to_minutes(milliseconds):
    seconds = milliseconds / 1000  # convert milliseconds to seconds
    minutes = seconds / 60  # convert seconds to minutes
    return minutes

def milliseconds_to_hours(milliseconds):
    seconds = milliseconds / 1000  # convert milliseconds to seconds
    minutes = seconds / 60  # convert seconds to minutes
    hours = minutes / 60  # convert minutes to hours
    return hours


def calculate_session_duration(x):  
    if not math.isnan(x['next_session_start_time_ms']):  
        return x['next_fv_start_time_stamp_ms_datetime'] - x['a_fv_start_time_stamp_ms_datetime']  
    else:  
        return None  

