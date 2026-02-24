import math
import datetime

def greet():
    """Prints a hello message with the current date and time."""
    current_time = datetime.datetime.now()
    print(f"Hello, World! Current date and time: {current_time}")

if __name__ == "__main__":
    greet()

