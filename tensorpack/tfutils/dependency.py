
import tensorflow as tf

from ..utils.argtools import graph_memoized

"""
Utils about parsing dependencies in the graph.
"""

__all__ = [
    'dependency_of_targets', 'dependency_of_fetches'
]


@graph_memoized
def dependency_of_targets(targets, op):
    """
    Check that op is in the subgraph induced by the dependencies of targets.
    The result is memoized.

    This is useful if some SessionRunHooks should be run only together with certain ops.

    Args:
        targets: a tuple of ops or tensors. The targets to find dependencies of.
        op (tf.Operation or tf.Tensor):

    Returns:
        bool: True if any one of `targets` depend on `op`.
    """
    # TODO tensorarray? sparsetensor?
    if isinstance(op, tf.Tensor):
        op = op.op
    assert isinstance(op, tf.Operation), op

    try:
        from tensorflow.contrib.graph_editor import get_backward_walk_ops  # deprecated
    except ImportError:
        from tensorflow.python.ops.op_selector import get_backward_walk_ops
    # alternative implementation can use graph_util.extract_sub_graph
    dependent_ops = get_backward_walk_ops(targets, control_inputs=True)
    return op in dependent_ops


def dependency_of_fetches(fetches, op):
    """
    Check that op is in the subgraph induced by the dependencies of fetches.
    fetches may have more general structure.

    Args:
        fetches: An argument to `sess.run`. Nested structure will affect performance.
        op (tf.Operation or tf.Tensor):

    Returns:
        bool: True if any of `fetches` depend on `op`.
    """
    try:
        from tensorflow.python.client.session import _FetchHandler as FetchHandler
        # use the graph of the op, so that this function can be called without being under a default graph
        handler = FetchHandler(op.graph, fetches, {})
        targets = tuple(handler.fetches() + handler.targets())
    except ImportError:
        if isinstance(fetches, list):
            targets = tuple(fetches)
        elif isinstance(fetches, dict):
            raise ValueError("Don't know how to parse dictionary to fetch list! "
                             "This is a bug of tensorpack.")
        else:
            targets = (fetches, )
    return dependency_of_targets(targets, op)


if __name__ == '__main__':
    a = tf.random_normal(shape=[3, 3])
    b = tf.random_normal(shape=[3, 3])
    print(dependency_of_fetches(a, a))
    print(dependency_of_fetches([a, b], a))
