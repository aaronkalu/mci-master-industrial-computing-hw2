import os
import tempfile

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="order_network_tests_")
os.environ.setdefault("LITELLM_API", "http://127.0.0.1:9/v1")
