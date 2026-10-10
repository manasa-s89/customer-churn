import os
import sys

# Ensure customer churn application directory is in Python module search path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app_dir = os.path.join(base_dir, "customer churn")

if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

