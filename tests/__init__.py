from mpqp.environment import enable_typecheck
from mpqp.environment.typechecked import is_typecheck_enabled

TMP_TYPECHECK = is_typecheck_enabled()
enable_typecheck(True)

import importlib
import sys

to_remove = [m for m in sys.modules if m == "mpqp" or m.startswith("mpqp" + ".")]
for mod in to_remove:
    del sys.modules[mod]
importlib.import_module("mpqp")
