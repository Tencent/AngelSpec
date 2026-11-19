import sys

from angelspec.transfer.mooncake import eagle_store

# One module object under both names, so patches and isinstance checks agree.
sys.modules[__name__] = eagle_store
