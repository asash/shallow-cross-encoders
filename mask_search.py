from datetime import datetime
import subprocess
import os
import shlex
import sys
from pathlib import Path

base_output_dir = Path(sys.argv[1])

masks = [0, 0.05, 0.1, 0.2, 0.4, 0.8]
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_id = "mask_search_" + time
out_dir = base_output_dir / experiment_id / "output"
tb_dir = base_output_dir / experiment_id / "tensorboard"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(tb_dir, exist_ok=True)


for mask in masks:
    cmd = f"python train_shallow_crossencoder.py --mask {mask:.2f} --output-dir {out_dir} --tensorboard-dir {tb_dir}"
    print(cmd)
    subprocess.check_call(shlex.split(cmd))