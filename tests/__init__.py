from mpqp.environment import enable_typecheck
from mpqp.environment.var_cache import is_typecheck_enabled

TMP_TYPECHECK = is_typecheck_enabled()
enable_typecheck(True)
