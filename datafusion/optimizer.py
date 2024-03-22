from ._internal import optimizer


def __getattr__(name):
    return getattr(optimizer, name)
