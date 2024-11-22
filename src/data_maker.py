from typing import List, Tuple

import numpy as np
import os
import pickle
import PyMiniSolvers.minisolvers as minisolvers

def verify_solution(n: int, formula: List[List[int]], assignment: List[int]) -> bool:
  solver = minisolvers.MinisatSolver()
  for _ in range(n):
    solver.new_var(dvar=True)
  for clause in formula:
    solver.add_clause(clause)
  
  # add the assignment
  for var in assignment:
    solver.add_clause([var])

  return solver.solve()

def is_sat(n: int, formula: List[List[int]]) -> bool:
  solver = minisolvers.MinisatSolver()
  for _ in range(n):
    solver.new_var(dvar=True)
  for clause in formula:
    solver.add_clause(clause)

  return solver.solve()


def pickle_dump_in_chunks(data, path, chunk_size=10_000):
  formulas_chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

  for i, chunk in enumerate(formulas_chunks):
    chunk_path = os.path.join(path, f'dataset_part_{i}.pkl')
    print(f"Writing {len(chunk)} formulas to {chunk_path}...")
    with open(chunk_path, 'wb') as f_dump:
        pickle.dump(chunk, f_dump)

def pickle_load_from_chunks(datadir_path):
  chunks = []
  i = 0
  while True:
      chunk_path = os.path.join(datadir_path, f'dataset_part_{i}.pkl')
      if not os.path.exists(chunk_path):
        break
      print(f"Importing {i}th dataset part from {chunk_path}...")
      with open(chunk_path, 'rb') as f:
        chunk = pickle.load(f)
        chunks.extend(chunk)
      i += 1
  return chunks

class DataMaker:
  def __init__(self, seed):
    self._seed = seed
    self._rng = np.random.default_rng(seed)

  def test_randomness(self):
    random_nums: List[str] = []
    for _ in range(10):
      random_nums.append(str(self._rng.random()))
    print(', '.join(random_nums))

  # Generate clause with k literals for a n-variables formula
  def generate_k_clause(self, n: int, k: int) -> List[int]:
    vs: np.ndarray[int] = self._rng.choice(n, size=min(n, k), replace=False)
    return [v + 1 if self._rng.random() < 0.5 else -(v + 1) for v in vs]

  # Generate DIMACS string of the formula
  @staticmethod
  def to_dimacs(n_vars: int, formula):
    txt = f'p cnf {n_vars} {len(formula)}\n'
    for clause in formula:
      txt += ' '.join(map(str, clause)) + ' 0\n'

    return txt
  

class DataMaker_kHorn(DataMaker):
  # Generates random k-Horn-SAT(m) formulas. Each formula has:
  # - n variables
  # - m clauses
  # - k literals per clause, and one of them must be positive
  def __init__(self, seed, k):
    super().__init__(seed)
    self._k = k

  def generate(self, data_dir: str, ns: List[int], alpha_min: float, alpha_max: float, alpha_inc: float, N: int,
               dimacs: bool) -> List[Tuple[int, str, int, int, List[List[int]]]]:
    log_path: str = os.path.join(data_dir, 'log.txt')
    logfile = open(log_path, 'w')

    tot_formulas: int = np.floor(((alpha_max - alpha_min) / alpha_inc) + 1) * N * len(ns)
    formulas: List[Tuple[int, str, int, int, List[List[int]]]] = []
    formulas_idx: int = 1

    # Generate formulas with n variables for each n in ns
    alpha_inc = round(alpha_inc, 3)
    for n in ns:
      alpha = round(alpha_min, 3)
      while alpha <= alpha_max:
        # print("Alpha: %s" % alpha)
        for N_idx in range(N):
          # Generate N formulas for each alpha
          formula: List[List[int]] = []
          # Genrate prefix, a conjunction of: 1 negative literal, and 0.2*n positive literals
          formula.append([-1])
          for lit in self._rng.choice(n-1, size=int(round(0.5*n, 0)), replace=False):
            formula.append([lit+2])
          m = int(round(alpha * n, 0))  # int() alone could introduce errors, i.e. 122.9999999 becomes 122
          assert (m/n == alpha), "Size '%s' for formulas leads to wrong alpha (%s instead of %s)" % (n, m/n, alpha)
          for _ in range(m):
            formula.append(self.generate_k_clause(n, self._k))

          log_details: List[str] = ["%d vars" % n, "%d clauses" % len(formula)]

          dimacs_path: str = None
          if dimacs:
            dimacs_path = f"dimacs/{self._k}-horn_{n}_{m}_{N_idx}.cnf"
            with open(os.path.join(data_dir, dimacs_path), 'w') as dimacs_file:
              dimacs_file.write(self.to_dimacs(n, formula))
            log_details.append(dimacs_path)

          formulas.append((formulas_idx, n, m, formula, is_sat(n, formula), dimacs_path))

          log_str: str = "%d/%d formula generated (%s)" % (formulas_idx, tot_formulas, ', '.join(log_details))
          print(log_str)
          print(log_str, file=logfile, flush=True)

          formulas_idx += 1

        alpha += alpha_inc
        alpha = round(alpha, 3)
    
    return formulas
  
  def generate_k_clause(self, n, k):
    vs: np.ndarray[int] = self._rng.choice(n, size=min(n, k), replace=False)
    vs = [-(v + 1) for v in vs]
    vs[0] = -vs[0] # if self._rng.random() < 0.5 else vs[0]
    return vs

# UNFINISHED CODE (might be useful for future work)
# class DataMaker_n_k_Horn(DataMaker):
#   # Generates H^k_{n,d} formulas such that, for a finite k > 0 and a vector d 
#   # of k nonnegative real numbers d1,d2,...,dk, d1 < 1, let the random 
#   # Horn-SAT formula H^k_{n,d} be the conjunction of:
#   # - a single negative literal ¬x1,
#   # - d1*n positive literals chosen uniformly without replacement from x2, . . . , xn, and
#   # - for each 2 ≤ j ≤ k, dj*n clauses chosen uniformly from the j*\binom{n}{j} possible Horn clauses with j variables where one literal is positive.
#   # [Moore et al. "A continuous-discontinuous second-order transition in the satisfiability of random Horn-SAT formulas." (2007)]
#   def __init__(self, seed, k):
#     super().__init__(seed)
#     self._k = k

#   def generate(self, data_dir: str, n: int, ds: List[float], N: int, dimacs: bool) -> List[Tuple[int, str, int, int, List[List[int]]]]:
#     log_path: str = os.path.join(data_dir, 'log.txt')
#     logfile = open(log_path, 'w')

#     tot_formulas: int = N
#     formulas: List[Tuple[int, str, int, int, List[List[int]]]] = []
#     formulas_idx: int = 1

#     for _ in range(N):
#       formula = self.generate_formula(n, ds)

#       log_details: List[str] = ["%d vars" % n, "ds=%s" % ', '.join(map(str, ds))]

#       dimacs_path: str = None
#       if dimacs:
#         dimacs_path = f"dimacs/horn_{self._k}_{n}_{'_'.join([f"d{i}={str(d)}" for i,d in enumerate(ds)])}_{formulas_idx}.cnf"
#         with open(os.path.join(data_dir, dimacs_path), 'w') as dimacs_file:
#           dimacs_file.write(self.to_dimacs(n, formula))
#         log_details.append(dimacs_path)

#       formulas.append((formulas_idx, dimacs_path, n, len(formula), ds[0], ds[1], formula))

#       log_str: str = "%d/%d formula generated (%s)" % (formulas_idx, tot_formulas, ', '.join(log_details))
#       print(log_str)
#       print(log_str, file=logfile, flush=True)

#       formulas_idx += 1
    
#     return formulas
  
#   def generate_formula(self, n: int, d: float) -> List[List[int]]:
#     formula: List[List[int]] = []
#     # single negative literal
#     formula.append([-1])

#     # d1*n positive literals chosen uniformly without replacement from x2, . . . , xn
#     n_pos = int(np.floor(d * n))
#     pos_literals = list(range(2, n+1))
#     self._rng.shuffle(pos_literals)
#     formula.append(pos_literals[:n_pos])

#     # for each 2 ≤ j ≤ k, dj*n clauses chosen uniformly from the j*\binom{n}{j} possible Horn clauses with j variables where one literal is positive.
#     for j in range(2, self._k+1):
#       n_clauses = int(np.floor(d * n))
#       for _ in range(n_clauses):
#         clause = self.generate_horn_clause(n, j)
#         formula.append(clause)

#     return formula
  
#   def generate_horn_clause(self, n: int, j: int) -> List[int]:
#     vs: np.ndarray[int] = self._rng.choice(n, size=j, replace=False)
#     pos = self._rng.integers(0, j)
#     return [v + 1 if i == pos else -(v + 1) for i, v in enumerate(vs)]

  

class DataMakerCNFk(DataMaker):
  # Generates formulas from k-CNF(m,n) where
  # - k is the number of variables per clause
  # - m is the number of clauses
  # - n is the numebr of variables
  def __init__(self, seed, k):
    super().__init__(seed)
    self._k = k

  def generate(self, data_dir: str, ns: List[int], alpha_min: float, alpha_max: float, alpha_inc: float, N: int,
               dimacs: bool) -> List[Tuple[int, str, int, int, List[List[int]]]]:
    log_path: str = os.path.join(data_dir, 'log.txt')
    logfile = open(log_path, 'w')

    tot_formulas: int = np.floor(((alpha_max - alpha_min) / alpha_inc) + 1) * N * len(ns)
    formulas: List[Tuple[int, str, int, int, List[List[int]]]] = []
    formulas_idx: int = 1

    # Generate formulas with n variables for each n in ns
    alpha_inc = round(alpha_inc, 3)
    for n in ns:
      alpha = round(alpha_min, 3)
      while alpha <= alpha_max:
        # print("Alpha: %s" % alpha)
        for N_idx in range(N):
          # Generate N formulas for each alpha
          formula: List[List[int]] = []
          m = int(round(alpha * n, 0))  # int() alone could introduce errors, i.e. 122.9999999 becomes 122
          assert (m/n == alpha), "Size '%s' for formulas leads to wrong alpha (%s instead of %s)" % (n, m/n, alpha)
          for _ in range(m):
            formula.append(self.generate_k_clause(n, self._k))

          log_details: List[str] = ["%d vars" % n, "%d clauses" % m]

          dimacs_path: str = None
          if dimacs:
            dimacs_path = f"dimacs/cnf{self._k}_{n}_{m}_{N_idx}.cnf"
            with open(os.path.join(data_dir, dimacs_path), 'w') as dimacs_file:
              dimacs_file.write(self.to_dimacs(n, formula))
            log_details.append(dimacs_path)

          formulas.append((formulas_idx, n, m, formula, is_sat(n, formula), dimacs_path))

          log_str: str = "%d/%d formula generated (%s)" % (formulas_idx, tot_formulas, ', '.join(log_details))
          print(log_str)
          print(log_str, file=logfile, flush=True)

          formulas_idx += 1

        alpha += alpha_inc
        alpha = round(alpha, 3)
    
    return formulas
  
  # function to merge datasets in pickle format from a given directory into a 
  # single dataset, and save it in the same directory as 'merged_dataset.pkl'
  @staticmethod
  def merge_datasets(dir: str):
    import glob
    import pickle
    import os

    datasets = glob.glob(os.path.join(dir, '*.pkl'))
    print(datasets)

    merged_dataset = []
    total_index: int = 1
    for dataset in datasets:
      with open(dataset, 'rb') as f:
        formulas = pickle.load(f)

        # go over each formula and update index with total index
        for formula in formulas:
          formula = (total_index,) + formula[1:]
          total_index += 1
          merged_dataset.append(formula)

    with open(os.path.join(dir, 'merged_dataset.pkl'), 'wb') as f:
      pickle.dump(merged_dataset, f)

  # function that takes in input the path to a pickle dataset and removes all 
  # the formulas that have alpha (m/n) not integer, then save the new dataset 
  # in the same directory of the original dataset as 'dataset_float_alpha.pkl'
  @staticmethod
  def remove_float_alpha(dataset_path: str):
    import pickle
    import os

    dir = os.path.dirname(dataset_path)
    with open(dataset_path, 'rb') as f:
      formulas = pickle.load(f)

      formulas = [formula for formula in formulas if formula[2] % formula[1] != 0]

      with open(os.path.join(dir, 'dataset_float_alpha.pkl'), 'wb') as f:
        pickle.dump(formulas, f)


class DataMakerSR(DataMaker):
  # SR(n) family of formulas is described at the beginning of Section 4 in https://arxiv.org/pdf/1802.03685.pdf
  def __init__(self, seed):
    super().__init__(seed)

  def generate(self, data_dir: str, n_pairs: int, min_n: int, max_n: int, dimacs: bool, iid: bool) -> List[Tuple[int, str, int, int, List[List[int]], bool]]:
    log_path: str = os.path.join(data_dir, 'log.txt')
    logfile = open(log_path, 'w')

    tot_formulas: int = n_pairs if iid else n_pairs * 2
    formulas: List[Tuple[int, str, int, int, List[List[int]], bool]] = []
    formulas_idx: int = 1
    pair_idx: int = 1

    while pair_idx <= n_pairs:
      n_vars: int = self._rng.integers(min_n, max_n, endpoint=True)
      subformula, clause_unsat, clause_sat = self.gen_formula_pair(n_vars)

      n_clauses: int = len(subformula) + 1
      
      formula_pair: List[Tuple[bool,List[int]]] = [(False, clause_unsat), (True, clause_sat)]

      if iid: # select a single formula from the pair UNSAT/SAT
        formula_pair = [formula_pair[self._rng.integers(0, 1, endpoint=True)]]
        
      for is_sat, last_clause in formula_pair:
        formula: List[List[int]] = subformula + [last_clause]

        log_details: List[str] = ["%d vars" % n_vars, "%d clauses" % n_clauses, "SAT" if is_sat else "UNSAT"]

        dimacs_path: str = None
        if dimacs:
          dimacs_path = f"dimacs/sr_n={n_vars}_p={pair_idx}_sat={int(is_sat)}.cnf"
          with open(os.path.join(data_dir, dimacs_path), 'w') as dimacs_file:
            dimacs_file.write(self.to_dimacs(n_vars, formula))
          log_details.append(dimacs_path)

        formulas.append((formulas_idx, dimacs_path, n_vars, n_clauses, formula, is_sat))

        log_str: str = "%d/%d formula generated (%s)" % (formulas_idx, tot_formulas, ', '.join(log_details))
        print(log_str)
        print(log_str, file=logfile, flush=True)

        formulas_idx += 1
      pair_idx += 1
    
    return formulas

  # Generate pair of formulas, one sat and one unsat, differing only for one literal.
  # - subformula is SAT
  # - subformula /\ clause_sat is SAT
  # - subformula /\ clause_unsat is UNSAT
  # clause_sat and clause_unsat differ only for the first literal which is negated
  def gen_formula_pair(self, n) -> Tuple[List[List[int]], List[int], List[int]]:
    solver = minisolvers.MinisatSolver()
    for _ in range(n):
      solver.new_var(dvar=True)
    subformula: List[List[int]] = []

    while True:
      # from the paper:
      # clause length ~ 1 + Bernoulli(0.7) + Geometric(0.4)
      k_base: int = 1 if self._rng.random() < 0.3 else 2
      k: int = k_base + self._rng.geometric(0.4)
      clause: List[int] = self.generate_k_clause(n, k)
      solver.add_clause(clause)
      is_sat: bool = solver.solve()
      if is_sat:
        subformula.append(clause)
      else:
        break

    clause_unsat = clause
    clause_sat = [- clause_unsat[0] ] + clause_unsat[1:]

    return subformula, clause_unsat, clause_sat


if __name__ == "__main__":
  
  import argparse

  parser = argparse.ArgumentParser()

  task_parsers = parser.add_subparsers(dest='task', help='Generating procedures')

  parser.add_argument('data_dir', type=str, help="Data directory")
  parser.add_argument('seed', type=int, help="Seed number")
  parser.add_argument('--dimacs', action='store_true', default=False, help="Dump DIMACS files (def. False)")

  sr_parser = task_parsers.add_parser('sr')

  sr_parser.add_argument('n_pairs', type=int, help="Number of IID pairs")
  sr_parser.add_argument('min_n', type=int, help="Min. number of variables")
  sr_parser.add_argument('max_n', type=int, help="Max. number of variables")
  sr_parser.add_argument('--iid', action='store_true', default=False, 
                      help="Select a single formula from each pair UNSAT/SAT (def. False)")
  
  cnf_parser = task_parsers.add_parser('cnf')

  cnf_parser.add_argument('-k', type=int, default=3, help="Size of the clauses (def. 3)")
  cnf_parser.add_argument('--ns', type=lambda s: [int(item) for item in s.split(',')], 
                          help="Number of variables (def. [5,10,15,20,30,40])")
  cnf_parser.add_argument('alpha_min', type=float, help="Minimum alpha")
  cnf_parser.add_argument('alpha_max', type=float, help="Minimum alpha")
  cnf_parser.add_argument('--alpha_inc', type=float, default=0.2, help="Minimum alpha (def. 0.2)")
  cnf_parser.add_argument('-N', type=int, default=200, help="Number of formulas per alpha value (def. 200)")

  merge_parser = task_parsers.add_parser('merge_cnf')
  merge_parser = task_parsers.add_parser('remove_int_alpha_cnf')

  horn_parser = task_parsers.add_parser('horn')
  horn_parser.add_argument('-k', type=int, default=3, help="Size of the clauses (def. 3)")
  horn_parser.add_argument('--ns', type=lambda s: [int(item) for item in s.split(',')], 
                          help="Number of variables (def. [5,10,15,20,30,40])")
  horn_parser.add_argument('alpha_min', type=float, help="Minimum alpha")
  horn_parser.add_argument('alpha_max', type=float, help="Minimum alpha")
  horn_parser.add_argument('--alpha_inc', type=float, default=0.2, help="Minimum alpha (def. 0.2)")
  horn_parser.add_argument('-N', type=int, default=200, help="Number of formulas per alpha value (def. 200)")

  args = parser.parse_args()

  # create directory
  os.makedirs(args.data_dir, exist_ok=True)
  if args.dimacs:
    os.makedirs(os.path.join(args.data_dir, 'dimacs/'), exist_ok=True)

  if args.task == 'sr':
    sr = DataMakerSR(args.seed)

    formulas = sr.generate(args.data_dir,
                      args.n_pairs,
                      args.min_n,
                      args.max_n,
                      args.dimacs,
                      args.iid)
    
  elif args.task == 'cnf':
    cnf = DataMakerCNFk(args.seed, args.k)
    ns = [5,10,15,20,30,40]
    if args.ns:
      ns = args.ns

    formulas = cnf.generate(args.data_dir,
                            ns,
                            args.alpha_min,
                            args.alpha_max,
                            args.alpha_inc,
                            args.N,
                            args.dimacs)
  elif args.task == 'horn':
    horn = DataMaker_kHorn(args.seed, args.k)
    ns = [5,10,15,20,30,40]
    if args.ns:
      ns = args.ns

    formulas = horn.generate(args.data_dir,
                            ns,
                            args.alpha_min,
                            args.alpha_max,
                            args.alpha_inc,
                            args.N,
                            args.dimacs)
  elif args.task == 'merge_cnf':
    DataMakerCNFk.merge_datasets(args.data_dir)
  elif args.task == 'remove_int_alpha_cnf':
    DataMakerCNFk.remove_float_alpha(os.path.join(args.data_dir, 'dataset.pkl'))
  else:
    raise NotImplementedError("Unsupported task")

  pickle_path = os.path.join(args.data_dir, 'dataset.pkl')

  if args.task != 'merge_cnf' and args.task != 'remove_int_alpha_cnf':
    print("Writing %d formulas to %s..." % (len(formulas), pickle_path))
    with open(pickle_path, 'wb') as f_dump:
      pickle.dump(formulas, f_dump)
    # pickle_dump_in_chunks(formulas, args.data_dir)
