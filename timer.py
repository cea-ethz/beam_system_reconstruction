import time

timers = {}


def start(name):
    timers[name] = []
    timers[name].append(time.time())


def end(name):
    timers[name].append(time.time())

    if len(timers[name]) % 2 != 0:
        print("Timer : Bad time list count")

    total = 0

    for i, j in zip(timers[name][0::2], timers[name][1::2]):
        total += (j - i)

    print("{} completed in {} seconds".format(name, round(total, 3)))

    del timers[name]


def pause(name):
    timers[name].append(time.time())


def unpause(name):
    timers[name].append(time.time())


def check_for_orphans():
    for name in timers:
        print("Note : Timer {} left open".format(name))
