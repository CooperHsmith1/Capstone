"""Make the package importable without triggering the mjlab task registry.

``whiteboard/__init__.py`` imports the G1 config package, which imports mjlab.
The state-estimation tests deliberately do not need mjlab, so they import
``mdp.state_estimation.*`` directly. Putting the package root on ``sys.path``
here lets that work from any working directory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
