import os

"""
Renames the filenames within the same directory to lowercase.
"""

path =  os.getcwd()
filenames = os.listdir(path)

for filename in filenames:
    os.rename(filename, filename.lower())