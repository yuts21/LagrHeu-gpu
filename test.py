import os
import subprocess

file = []
for root, dirs, files in os.walk("./data"):
    for f in files:
        a, _ = os.path.splitext(os.path.join(root, f))
        file.append(a)
file = sorted(list(set(file)))
f = open("./testLog.txt", "w")
devNull = open(os.devnull, 'w')
for i in file:
    s = "solving {}".format(i)
    print(s)
    f.write(s)

    cmd = "./LagrHeu {}".format(i)
    f = open("{}.bound".format(i), "r")
    std = int(f.read().strip())
    f.close()
    result = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=devNull, universal_newlines=True)
    result.wait()
    res = int(result.stdout.read().strip())
    if(res != std):
        err = "mismatch on {}, std={}, find={}".format(i, std, res)
        print(err)
        f.write(err)
        f.flush()

f.close()
