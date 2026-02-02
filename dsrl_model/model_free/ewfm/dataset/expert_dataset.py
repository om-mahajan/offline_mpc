# Minimal expert_dataset for compatibility
# Only used in Memory.load() which we don't use

class ExpertDataset:
    """Placeholder for expert dataset - not used when using load_from_data()."""
    def __init__(self, path, num_trajs, sample_freq, seed):
        raise NotImplementedError("Use Memory.load_from_data() instead")
    
    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        raise NotImplementedError("Use Memory.load_from_data() instead")
