"""E17: repeat E16 on the cached positive-preserving 1M MS MARCO diagnostic, no training."""
from common import REPO
from e16_quantization_pilot import main

if __name__ == '__main__':
    main(dataset='msmarco1m', measurement='E17', label='exploratory second-scale replication',
         out_path=REPO / 'results/m15_e17_quantization_replication.json',
         run_dir=REPO / 'work/m15/e17-quantization-replication', script=__file__)
