#!/bin/bash                                                                     
#SBATCH --time=72:00:00 # Time limit for the job (REQUIRED).                    
#SBATCH --job-name=explore-data # Job name                                       
#SBATCH --ntasks=1 # Number of cores for the job. Same as SBA TCH -n 1          
#SBATCH --partition=P4V16_HAS16M128_L # Partition/queue to run the job in. (REQUIRED)                                                                          
#SBATCH -e explore-data.err # Error file for this job.                              
#SBATCH -o explore-data.out # Output file for this job.                             
#SBATCH -A gol_hkh232_uksr # Project allocation account name (REQUIRED)         
#SBATCH --gres=gpu:1                                                            

module load ccs/Miniconda3
eval "$(conda shell.bash hook)"
source activate myenv
python src/explore-data.py