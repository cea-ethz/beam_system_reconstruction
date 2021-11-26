import time

times = []
timer_name = None


def start(name):
    global times
    global timer_name
    times.append(time.time())
    timer_name = name


def end():
    global timer_name
    global times

    times.append(time.time())

    if len(times) % 2 != 0:
        print("Timer : Bad time list count")

    total = 0

    for i, j in zip(times[0::2], times[1::2]):
        total += (j - i)

    print("{} completed in {} seconds".format(timer_name, round(total, 3)))
    timer_name = None
    times = []


def pause():
    times.append(time.time())


def unpause():
    times.append(time.time())
