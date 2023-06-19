#!/bin/bash
#SBATCH -p compute                 #specify partition
#SBATCH -N 1                       #specify number of nodes
#SBATCH --cpus-per-task=1          #specify number of cpus
#SBATCH -t 120:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200060                     # Specify project name
#SBATCH -J cap_tb                      # Specify job name

#SBATCH --error=error_tb.txt
#SBATCH --output=output_tb.txt

module purge
module load Miniconda3/22.11.1-1
conda activate palm_search


port=$(shuf -i 6000-9999 -n 1)
USER=$(whoami)
node=$(hostname -s)

echo -e "
Jupyter server is running on: $(hostname)
Job starts at: $(date)
Copy/Paste this in your local terminal to ssh tunnel with remote
-----------------------------------------------------------------
ssh -L $port:$node:$port $USER@lanta.nstda.or.th
-----------------------------------------------------------------
Then open a browser on your local machine to the following address
------------------------------------------------------------------
http://localhost:${port}/
------------------------------------------------------------------
"

tensorboard --logdir ./runs --port $port --host $node
