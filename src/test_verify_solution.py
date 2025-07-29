from sat_solve import verify_solution
import PyMiniSolvers.minisolvers as minisolvers

if __name__ == "__main__":
    n = 3

    formula_sat = [[1, 2, 3], [-1, 2, 3]]  # (A ∨ B ∨ C) ∧ (¬A ∨ B ∨ C)

    formula_unsat = [  # (A ∨ B ∨ C) ∧ (¬A ∨ B ∨ C) ∧ ... ∧ (¬A ∨ ¬B ∨ ¬C)
        [1, 2, 3],
        [-1, 2, 3],
        [1, -2, 3],
        [-1, -2, 3],
        [1, 2, -3],
        [-1, 2, -3],
        [1, -2, -3],
        [-1, -2, -3],
    ]

    assignment = [1, 2, 3]

    tmpsolver = minisolvers.MinisatSolver()
    for _ in range(n):
        tmpsolver.new_var(dvar=True)
    for clause in formula_sat:
        tmpsolver.add_clause(clause)
    print(f"The satisfiable formula is {'SAT' if tmpsolver.solve() else 'UNSAT'}")

    tmpsolver = minisolvers.MinisatSolver()
    for _ in range(n):
        tmpsolver.new_var(dvar=True)
    for clause in formula_unsat:
        tmpsolver.add_clause(clause)
    print(f"The unsatisfiable formula is {'SAT' if tmpsolver.solve() else 'UNSAT'}")

    print(
        f"The assignment is a model of the satisfiable formula: {verify_solution(n, formula_sat, assignment)}"
    )
    print(
        f"The assignment is a model of the satisfiable formula: {verify_solution(n, formula_unsat, assignment)}"
    )
