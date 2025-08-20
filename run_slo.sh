#!/bin/bash
#SBATCH --job-name=slo_extract
#SBATCH --mail-type=NONE
#SBATCH --mail-user=NONE
#SBATCH --ntasks=1                 # Number of MPI tasks (i.e. processes)
#SBATCH --cpus-per-task=32           # Number of cores per MPI task 
#SBATCH --nodes=1                    # Maximum number of nodes to be allocated
#SBATCH --mem=64g
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out

pwd; hostname; date

module load conda
conda activate oct-analysis

DATA_CSV=$1
OUTPUT_CSV=$2
Robust=$3

python3 usage_slo.py --analysis_csv $DATA_CSV --output_directory $OUTPUT_CSV --robust_run $Robust