from . import leaf, optimizer, initializer, logger

try:
    import theano
    from theano.gpuarray import ContextNotDefined
    from theano.gpuarray.dnn import dnn_available
    assert dnn_available(None)
    assert theano.config.dnn.enabled != "False"
except (ImportError, AssertionError, ContextNotDefined):
    logger.logger.warning("cuDNN is unavailable")
