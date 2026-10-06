import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Tests must never touch the user's real application settings
# (QSettings "ComicLabs"/"ComicTranslate" stores API keys, credentials and
# preferences). Redirect HOME/XDG_CONFIG_HOME so QSettings writes into a
# throwaway directory instead of ~/.config/ComicLabs/ComicTranslate.conf.
# Must happen before Qt reads the environment.
_test_home = tempfile.mkdtemp(prefix="ct_test_home_")
os.environ.setdefault("XDG_CONFIG_HOME", os.path.join(_test_home, ".config"))
os.environ["HOME"] = _test_home
