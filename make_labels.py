import argparse
import os
import pickle


def main():
    parser = argparse.ArgumentParser(description='Build labels.pkl from PDBbind index file')
    parser.add_argument('--index', default='data/v2019/index/INDEX_general_PL_data.2019',
                        help='Path to PDBbind INDEX_general_PL_data file')
    parser.add_argument('--out', default='data/processed/labels.pkl',
                        help='Output path for labels pickle (default: data/processed/labels.pkl)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    labels = {}
    with open(args.index) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            labels[parts[0]] = float(parts[3])

    with open(args.out, 'wb') as f:
        pickle.dump(labels, f)

    print(f'Saved {len(labels)} labels to {args.out}')


if __name__ == '__main__':
    main()
