#!/bin/sh

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --time 06:00:00
#SBATCH -p newnodes
#SBATCH -J experiment_design

module add engaging/python/3.6.0
source venv/bin/activate

