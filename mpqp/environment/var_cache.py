from functools import lru_cache

from mpqp.environment.env_manager import get_env_variable

@lru_cache(maxsize=1)
def is_translation_warning() -> bool:
    return not get_env_variable("MPQP_TRANSLATION_WARNING").lower() == "false"

@lru_cache(maxsize=1)
def is_typecheck_enabled() -> bool:
    return get_env_variable("MPQP_TYPECHECK").lower() == "true"