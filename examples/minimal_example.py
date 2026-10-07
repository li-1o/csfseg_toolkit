"""Show the default CSFSeg configuration.

Most users should start with the command-line tools. This tiny example only
checks that the package imports and shows the default model/preprocessing
settings.
"""

from csfseg.config import DEFAULT_CONFIG

print(DEFAULT_CONFIG)
