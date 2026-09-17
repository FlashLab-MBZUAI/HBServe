"""Optional native SGLang frontend for the persistent HBFSim engine.

SGLang owns scheduling, prefix reuse and KV allocation. This package reuses
HBServe's semantic compiler and maps the native slots without reallocating them.
Importing it does not import the optional SGLang runtime.
"""

SGLANG_REVISION = "13d593b6cf885c5c4d50eea88c82b9e28cf5941e"
