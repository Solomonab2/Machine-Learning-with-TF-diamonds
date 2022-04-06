""" get_dir.py  -  for getting current and parent directories """
import os

def getCurrDir():
    return os.getcwd()

def getParentDir(path):
    return os.path.abspath(os.path.join(path, os.pardir))