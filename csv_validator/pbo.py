from itertools import combinations

import numpy as np


def generate_cpcv_splits(n_obs, n_groups=6, n_test_groups=2, embargo=5):
    """Generate (train_idx, tes_idx) pairs for all C(n_groups, n_test_groups) combinations."""
    group_bounds = np.linspace(0, n_obs, n_groups + 1, dtype=int)
    groups = [np.arange(group_bounds[i], group_bounds[i + 1]) for i in range(n_groups)]

    splits = []
    for test_groups_ids in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[g] for g in test_groups_ids])
        train_idx = np.concatenate(
            [groups[g] for g in range(n_groups) if g not in test_groups_ids]
        )

        # purge: remove train observations within 'embargo' of any test boundary
        purged_train = []
        for idx in train_idx:
            near_test = False
            for t in test_idx:
                if abs(idx - t) <= embargo:
                    near_test = True
                    break
            if not near_test:
                purged_train.append(idx)

        splits.append((np.array(purged_train), test_idx))
    return splits
