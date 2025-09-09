"""
This module provides a decorator for conditionally applying runtime type checking
to functions based on the value of the ``MPQP_TYPECHECK`` environment variable.
By default, type checking is disabled to avoid performance overhead in production environments.
To enable type checking, set the environment variable to "True".

.. code-block:: python

    >>> from mpqp.environment import enable_typecheck
    >>> enable_typecheck(True) # doctest: +SKIP

"""

from typeguard import typechecked

from mpqp.environment.var_cache import is_typecheck_enabled


def enable_typecheck(enable: bool):
    """
    Enable or disable MPQP's runtime type checking, then reload the entire `mpqp` package.

    This function updates the `MPQP_TYPECHECK` environment variable and clears the
    cached result of `is_typecheck_enabled()`. It then removes all `mpqp`-related
    modules from `sys.modules` so that the next import will load fresh code reflecting
    the new type checking setting.

    Args:
        enable:
            If True, type checking is enabled by setting `MPQP_TYPECHECK` to "True".
            If False, type checking is disabled by setting it to "False".

    """
    from mpqp.environment.env_manager import save_env_variable

    save_env_variable("MPQP_TYPECHECK", "True" if enable else "False")
    is_typecheck_enabled.cache_clear()

    # Refresh all mpqp
    import importlib
    import sys

    to_remove = [m for m in sys.modules if m == "mpqp" or m.startswith("mpqp" + ".")]
    for mod in to_remove:
        del sys.modules[mod]
    importlib.import_module("mpqp")


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
