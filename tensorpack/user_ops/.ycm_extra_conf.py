import tensorflow as tf

flags = [
    '-Wall',
    '-Wextra',
    '-Werror',
    '-Wno-long-long',
    '-Wno-variadic-macros',
    '-fexceptions',
    '-std=c++11',
    '-x',
    'c++',

    '-isystem',
    tf.sysconfig.get_include()
]


def FlagsForFile(filename, **kwargs):
    return {
        'flags': flags,
        'do_cache': True
    }
