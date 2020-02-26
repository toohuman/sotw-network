import argparse
import lzma
import pickle
import sys

result_directory = "../../results/test_results/sotw-network/"

# Parse the arguments of the program, e.g., agents, states, random init.
parser = argparse.ArgumentParser(description="Distributed decision-making\
    in a multi-agent environment in which agents must reach a consensus\
        about the true state of the world.")
parser.add_argument("states", type=int)

try:
    with lzma.open(result_directory + file_name, "rb") as file:
        data = pickle.load(file)

except FileNotFoundError:
    print("MISSING: " + file_name)