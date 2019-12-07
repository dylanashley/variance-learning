#!/bin/sh

#SBATCH --mail-user=dashley@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-sutton
#SBATCH --time=03-00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000M

module load python/3.6.3
module load scipy-stack/2017b

./src/run.sh "$@"
