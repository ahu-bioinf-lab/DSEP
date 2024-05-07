import torch

MAX_DRUG_LEN = 200
def pack(atoms, adjs, labels, device='cuda:0'):
    atoms_len = 0
    N = len(atoms)

    for atom in atoms:
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    if atoms_len > MAX_DRUG_LEN: atoms_len = MAX_DRUG_LEN
    atoms_new = torch.zeros((N, atoms_len, 34), device='cuda:0')
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        if a_len > atoms_len: a_len = atoms_len
        atoms_new[i, :a_len, :] = atom[:a_len, :]
        i += 1

    adjs_new = torch.zeros((N, atoms_len, atoms_len), device='cuda:0')
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        if a_len > atoms_len: a_len = atoms_len
        adjs_new[i, :a_len, :a_len] = adj[:a_len, :a_len]
        i += 1

    labels_new = torch.zeros(N, dtype=torch.long, device='cuda:0')
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1
    return (atoms_new, adjs_new, labels_new)

def pack_combo(adjs1, atoms1, adjs2, atoms2, labels, device='cuda:0'):
    atoms1_len = 0
    atoms2_len = 0
    N = len(atoms1)

    for atom1 in atoms1:
        if atom1.shape[0] >= atoms1_len:
            atoms1_len = atom1.shape[0]

    if atoms1_len > MAX_DRUG_LEN: atoms1_len = MAX_DRUG_LEN
    atoms1_new = torch.zeros((N, atoms1_len, 34), device='cuda:0')

    i = 0
    for atom1 in atoms1:
        a_len = atom1.shape[0]
        if a_len > atoms1_len: a_len = atoms1_len
        atoms1_new[i, :a_len, :] = atom1[:a_len, :]
        i += 1
    adjs1_new = torch.zeros((N, atoms1_len, atoms1_len), device='cuda:0')
    i = 0
    for adj1 in adjs1:
        a1_len = adj1.shape[0]
        adj1 = adj1 + torch.eye(a1_len)
        if a1_len > atoms1_len: a1_len = atoms1_len
        adjs1_new[i, :a1_len, :a1_len] = adj1[:a1_len, :a1_len]
        i += 1

    for atom2 in atoms2:
        if atom2.shape[0] >= atoms2_len:
            atoms2_len = atom2.shape[0]
    if atoms2_len > MAX_DRUG_LEN: atoms2_len = MAX_DRUG_LEN
    atoms2_new = torch.zeros((N, atoms2_len, 34), device='cuda:0')
    i = 0
    for atom2 in atoms2:
        a2_len = atom2.shape[0]
        if a2_len > atoms2_len: a2_len = atoms2_len
        atoms2_new[i, :a2_len, :] = atom2[:a2_len, :]
        i += 1
    adjs2_new = torch.zeros((N, atoms2_len, atoms2_len), device='cuda:0')
    i = 0
    for adj2 in adjs2:
        a2_len = adj2.shape[0]
        adj2 = adj2 + torch.eye(a2_len)
        if a2_len > atoms2_len: a2_len = atoms2_len
        adjs2_new[i, :a2_len, :a2_len] = adj2[:a2_len, :a2_len]
        i += 1

    labels_new = torch.zeros(N, dtype=torch.long, device='cuda:0')
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1
    return (adjs1_new, atoms1_new, adjs2_new, atoms2_new, labels_new)