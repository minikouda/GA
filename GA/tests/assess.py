"""Assessment script for GA package using the diabetes dataset."""

from pathlib import Path
import sys
import time
import argparse 

from setdata import X_diab, y_diab

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import GA

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GA feature selection assessment on the diabetes dataset."
    )
    parser.add_argument(
        '--crossover',
        type=str,
        default='uniform', 
        choices=['uniform', 'single', 'kpoints'], 
        help='The type of crossover operation to use (e.g., uniform, single, kpoints).'
    )
    parser.add_argument(
        '--k-points',
        type=int,
        default=2,
        help='The number of cut points for k-point crossover.'
    )
    parser.add_argument(
        '--mutation-type',          
        type=str,
        default='bitflip',
        choices=['bitflip', 'swap'], 
        help='The type of mutation operation to use (bitflip, swap).'
    )
    parser.add_argument(
        '--n-gen',
        type=int,
        default=50,
        help='Number of generations for the GA run (n_gen).'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    ## Diabetes example
    print("Performance on diabetes dataset. This set of best covariates is based on an exhaustive all subsets search using AIC as the fitness score (rather than CV-based prediction error), with the R^2 simply the in-sample R^2 (and therefore likely a bit too high) based on that set of predictors. Your predictor set might differ a bit and your CV R^2 might be a bit lower. For this task, if your time reaches into the 10s or 100s of seconds, that would be concerning.\n")

    print(f"AIC-based best predictors (0-based indexing): 1,2,3,4,5,8 (sex+bmi+bp+s1+s2+s5)") 
    print(f"In-sample R^2: 0.515")


    t0 = time.time()
    results = GA.select(
        X_diab, 
        y_diab,
        crossover_type=args.crossover, 
        k_points=args.k_points,
        mutation_type=args.mutation_type,
        n_gen=args.n_gen, 
    )
    fulltime = time.time() - t0

    print(f"\n--- GA Run Details ---")
    print(f"Crossover Type Used: {args.crossover}")
    print(f"Selected predictors {results['selected']}.")
    print(f"CV R2: {results['R2']}.")
    print(f"Time taken: {fulltime}.")