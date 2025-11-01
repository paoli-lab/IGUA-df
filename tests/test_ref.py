# tests/test_golden.py
import pandas as pd
import pytest

test_cases = ["secretion_small", "ecoli", "emerge"]

@pytest.mark.parametrize("ref_data_id", test_cases)
def test_secretion_dataset_regression(ref_data_id):
    """Compare output against known good reference"""
    # using defaults (see makefile):
    # igua -i ../secretion_systems_metadata_short.tsv --output gcfs.tsv --compositions compositions.npz --features features.fa
    # igua -i ../strains_metadata.tsv --output gcfs.tsv --compositions compositions.npz --features features.fa
    # igua -i ../emerge_sample_15_metadata.tsv --output gcfs.tsv --compositions compositions.npz --features features.fa
    gcfs = pd.read_csv(f"tests/data/{ref_data_id}_gcfs.tsv", sep="\t")
    current_gcfs = pd.read_csv(f"test_output/{ref_data_id}_gcfs.tsv", sep="\t")

    # same shape and number of unique GCFs
    assert len(gcfs) == len(current_gcfs)
    assert gcfs["gcf_id"].nunique() == current_gcfs["gcf_id"].nunique()

    # clustering should be deterministic (if using same parameters)
    pd.testing.assert_frame_equal(
        gcfs.sort_values("cluster_id").reset_index(drop=True),
        current_gcfs.sort_values("cluster_id").reset_index(drop=True)
    )
