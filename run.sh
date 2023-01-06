#!/bin/bash
#SBATCH --job-name=serial_job_test    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=email@ufl.edu     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=1gb                     # Job memory request
#SBATCH --output=serial_test_%j.log   # Standard output and error log
pwd; hostname; date

module load python

python train_grid_search.py

date
