import os
import subprocess
import time
import json
file = []
for root, dirs, files in os.walk("./data"):
    for f in files:
        a, _ = os.path.splitext(os.path.join(root, f))
        file.append(a)
file = sorted(list(set(file)))
devNull = open(os.devnull, 'w')
f = open("./benchLog.txt", "w")
for i in file:
    s1 = time.time()
    cmd = "./LagrHeu {}".format(i)
    result = subprocess.Popen(cmd, shell=True, stdout=devNull, stderr=devNull)
    result.wait()
    s1 = time.time()-s1

    s2 = time.time()
    cmd = "./serial/LagrHeu --verbose 0 --inst {} --alpha 2.546 --alphastep 0.936 --minalpha 0.01 --innerIter 98 --maxiter 619 --algoType 2 --seed 505".format(
        i)
    result = subprocess.Popen(cmd, shell=True, stdout=devNull, stderr=devNull)
    result.wait()
    s2 = time.time()-s2

    js = json.dumps({"case": i, "cuda": s1, "serial": s2})
    print(js)
    f.write(js)
    f.flush()
f.close()
