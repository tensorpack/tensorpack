# common flags among multiple files go here
import absl

from tensorflow.python.platform import flags

# in the case of running unit tests, this is imported more than once
# TODO: there has got to be a more elegant way of doing this

try:
    flags.DEFINE_string("input", "", """TensorFlow 'GraphDef' file to load.""")
except absl.flags._exceptions.DuplicateFlagError:
    pass

try:
    flags.DEFINE_boolean("input_binary", True,
                         """Input graph binary or text.""")
except absl.flags._exceptions.DuplicateFlagError:
    pass

try:
    flags.DEFINE_string("output", "", """File to save the output graph to.""")
except absl.flags._exceptions.DuplicateFlagError:
    pass

try:
    flags.DEFINE_boolean("output_binary", True,
                         """Output graph binary or text.""")
except absl.flags._exceptions.DuplicateFlagError:
    pass
