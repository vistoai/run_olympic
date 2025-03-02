from datetime import datetime

# Get the current time with microseconds
current_time = datetime.now()

# Format the time to include milliseconds (3 decimal places of microseconds)
current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

# Convert the formatted time string to an integer (milliseconds since epoch)
current_time_milliseconds = int(datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000)

print("Current time in milliseconds:", current_time_milliseconds)