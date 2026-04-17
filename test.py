import os
import subprocess

def env_list():
    command = "conda env list"

    result = subprocess.run(command, stdout=subprocess.PIPE, text=True, shell=True)

    print(result.stdout)

a = env_list()

print(a)