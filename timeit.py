from datetime import datetime


def timeit(func, args=None, kwargs=None):
    start = datetime.now()
    if args is None:
        args = []
    if kwargs is None:
        kwargs = dict()
    result = func(*args, **kwargs)
    end = datetime.now()
    return result, (end - start).total_seconds()


