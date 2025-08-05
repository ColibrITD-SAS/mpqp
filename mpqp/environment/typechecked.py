"""
This module provides a decorator for conditionally applying runtime type checking
to functions based on the value of the ``MPQP_TYPECHECK`` environment variable.
by default, type checking is disabled to avoid performance overhead in production environments.
to enable type checking, set the environment variable to "true".

.. code-block:: python

    >>> from mpqp.environment import save_env_variable
    >>> save_env_variable("MPQP_TYPECHECK", "True") # doctest: +SKIP

"""

from typeguard import typechecked

from mpqp.environment.var_cache import is_typecheck_enabled


def conditional_typechecked(func):  # pyright: ignore[reportMissingParameterType]
    """
    Decorator that conditionally applies the ``typechecked`` decorator to a function
    based on the mpqp environment variable ``MPQP_TYPECHECK``.

    If type checking is enabled, the function is wrapped with ``typechecked`` to
    enforce runtime type checking. Otherwise, the original function is returned
    unmodified.

    Args:
        func : The function to be conditionally type-checked.

    Returns:
        The original or type-checked function, depending on
        whether type checking is enabled.

    """
    if is_typecheck_enabled():
        return typechecked(func)
    return func
