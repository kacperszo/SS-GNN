import argparse
import os
import pickle
from multiprocessing import Pool

from rdkit import Chem
from tqdm import tqdm

from utils import get_gnn_features


def process(args):
    pdb_id, root, out_dir, threshold = args

    pdb_path = os.path.join(root, pdb_id, pdb_id + '_protein.pdb')
    mol2_path = os.path.join(root, pdb_id, pdb_id + '_ligand.mol2')

    protein = Chem.MolFromPDBFile(pdb_path)
    if protein is None:
        protein = Chem.MolFromPDBFile(os.path.join(root, pdb_id, pdb_id + '_pocket.pdb'))
    ligand = Chem.MolFromMol2File(mol2_path)
    if ligand is None:
        suppl = Chem.SDMolSupplier(os.path.join(root, pdb_id, pdb_id + '_ligand.sdf'))
        mols = [m for m in suppl if m]
        if mols:
            ligand = mols[0]

    if protein is None or ligand is None:
        return pdb_id, 'none'

    try:
        x, edge_index, edge_attr = get_gnn_features(protein, ligand, threshhold=threshold)
        if x is None or edge_index is None or edge_attr is None:
            return pdb_id, 'none'
        with open(os.path.join(out_dir, pdb_id + '.pkl'), 'wb') as f:
            pickle.dump((x, edge_index, edge_attr), f)
        return pdb_id, 'ok'
    except Exception:
        return pdb_id, 'exception'


def main():
    parser = argparse.ArgumentParser(description='Preprocess CASF-2016 core set into graph features')
    parser.add_argument('--coreset', default='data/CASF-2016/coreset',
                        help='Path to CASF-2016 coreset directory (default: data/CASF-2016/coreset)')
    parser.add_argument('--out', default='data/processed/coreset_graphs',
                        help='Output directory for graph .pkl files (default: data/processed/coreset_graphs)')
    parser.add_argument('--threshold', type=float, default=5.0,
                        help='Distance threshold in Ångströms (default: 5.0)')
    parser.add_argument('--workers', type=int, default=os.cpu_count() - 1,
                        help='Number of worker processes (default: cpu_count - 1)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    already_done = set(f.split('.')[0] for f in os.listdir(args.out))
    all_ids = [d for d in os.listdir(args.coreset)
               if len(d) == 4 and d not in already_done]

    print(f'Processing {len(all_ids)} core set complexes '
          f'(skipping {len(already_done)} already done)')

    tasks = [(pdb_id, args.coreset, args.out, args.threshold) for pdb_id in all_ids]

    with Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(process, tasks), total=len(tasks)))

    n_ok = sum(1 for _, s in results if s == 'ok')
    n_none = sum(1 for _, s in results if s == 'none')
    n_exc = sum(1 for _, s in results if s == 'exception')
    print(f'\nDone: {n_ok} ok, {n_none} missing mol/protein, {n_exc} exceptions')


if __name__ == '__main__':
    main()
