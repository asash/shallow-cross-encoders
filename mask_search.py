from datetime import datetime
import subprocess
import os
import shlex

masks = [0, 0.05, 0.1, 0.2, 0.4, 0.8]
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = f"output/mask_search_{time}"
os.mkdir(out_dir)


for mask in masks:
    cmd = f"python train_shallow_crossencoder.py --mask {mask:.2f} --output-dir {out_dir}"
    print(cmd)
    subprocess.check_call(shlex.split(cmd))