"""
HYDROTWIN-AI Application Entry Point
Forwards execution to the advanced animated digital twin simulator in hydrotwin_twin.py.
"""

import os
import runpy

if __name__ == "__main__":
    twin_path = os.path.join(os.path.dirname(__file__), "hydrotwin_twin.py")
    runpy.run_path(twin_path, run_name="__main__")
