import torch
import sys
import subprocess

argslist = list(sys.argv)[1:]
world_size = torch.cuda.device_count()
device_ids = None
if '--world-size' in argslist:
    argslist[argslist.index('--world-size') + 1] = str(world_size)
else:
    argslist.append('--world-size')
    argslist.append(str(world_size))

if '--device-ids' in argslist:  # Manually specified GPU IDs
    device_ids = argslist[argslist.index('--device-ids') + 1].strip().split(',')
    world_size = len(device_ids)
    # Remove GPU IDs since these are not for the training script
    argslist.pop(argslist.index('--device-ids') + 1)
    argslist.pop(argslist.index('--device-ids'))
else:
    device_ids = list(range(world_size))

workers = []

for i, did in enumerate(device_ids):
    if '--rank' in argslist:
        argslist[argslist.index('--rank') + 1] = str(i)
    else:
        argslist.append('--rank')
        argslist.append(str(i))
    if '--gpu-rank' in argslist:
        argslist[argslist.index('--gpu-rank') + 1] = str(did)
    else:
        argslist.append('--gpu-rank')
        argslist.append(str(did))
    stdout = None if i == 0 else open("GPU_" + str(did) + ".log", "w")
    print(argslist)
    p = subprocess.Popen([str(sys.executable)] + argslist, stdout=stdout, stderr=stdout)
    workers.append(p)

for p in workers:
    p.wait()
