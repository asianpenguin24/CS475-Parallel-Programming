import os

print(os.getcwd())
for size in [1000, 4000, 10000, 40000, 100000, 400000, 1000000, 4000000, 8000000]:
    os.system("PROJ_0.5.exe " + str(size))

os.system("pause")