from ._internal import builder


def __getattr__(name):
    return getattr(builder, name)
