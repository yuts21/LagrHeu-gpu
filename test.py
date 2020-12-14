import os
import subprocess
import json

file = []
for root, dirs, files in os.walk("./data"):
    for f in files:
        a, _ = os.path.splitext(os.path.join(root, f))
        if(ord(f[0])-ord('a')>3):file.append(a)
file = sorted(list(set(file)))
f = open("./testLog.txt", "w")
devNull = open(os.devnull, 'w')
for i in file:
    f2 = open("{}.bound".format(i), "r")
    std = int(f2.read().strip())
    f2.close()

    cmd = "./LagrHeu {}".format(i)
    result = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    result.wait()
    res1 = int(result.stdout.read().strip())
    t1 = float(result.stderr.read().strip())

    cmd = "./serial/LagrHeu {}".format(i)
    result = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    result.wait()
    res2 = int(result.stdout.read().strip())
    t2 = float(result.stderr.read().strip())

    js = json.dumps({"case": i, "cuda res": res1, "cuda time": t1,
                     "serial res": res2, "serial time": t2, "std": std})
    print(js)
    f.write("{}\n".format(js))
    f.flush()

f.close()
