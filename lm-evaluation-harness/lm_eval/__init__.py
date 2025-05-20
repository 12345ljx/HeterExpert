import logging
import os
os.environ["http_proxy"] = "http://10.129.202.92:7900"
os.environ["https_proxy"] = os.environ["http_proxy"]


from .evaluator import evaluate, simple_evaluate


__version__ = "0.4.8"
