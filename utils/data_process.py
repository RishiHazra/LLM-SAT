import pickle as pkl

data = pkl.load(open('../dataset.pkl', 'rb'))
for sample in data:
    num_var, num_clauses, formula = sample[1:4]
    if num_var != 5 or num_clauses != 30:
        continue
    cnf_form = '(' + ')^('.join(['V'.join(map(str, clause)) for clause in formula]) + ')'
    print(cnf_form)
print(len(data))
