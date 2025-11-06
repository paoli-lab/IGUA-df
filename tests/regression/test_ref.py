import pandas as pd
import anndata
import pytest



test_cases = ["ecoli", "secretion_small", "emerge"]

@pytest.mark.parametrize("ref_data_id", test_cases)
def test_regression_gcfs(ref_data_id):
    """Compare output against known good reference"""
    # using defaults (see makefile):
    # igua -i ../secretion_systems_metadata_short.tsv --output gcfs.tsv --compositions compositions.npz --features features.fa
    # igua -i ../strains_metadata.tsv --output gcfs.tsv --compositions compositions.npz --features features.fa
    # igua -i ../emerge_sample_15_metadata.tsv --output gcfs.tsv --compositions compositions.npz --features features.fa
    gcfs = pd.read_csv(f"tests/fixtures/{ref_data_id}_gcfs.tsv", sep="\t")
    current_gcfs = pd.read_csv(f"tests/test_output/{ref_data_id}_gcfs.tsv", sep="\t")

    # same shape and number of unique GCFs
    assert len(gcfs) == len(current_gcfs)
    assert gcfs["gcf_id"].nunique() == current_gcfs["gcf_id"].nunique()

    # clustering should be deterministic (if using same parameters)
    # therefore the number of unique values per column should be the same
    pd.testing.assert_series_equal(
        current_gcfs.nunique(),
        gcfs.nunique()
    )
    # and the output gcfs should be identical
    pd.testing.assert_frame_equal(
        gcfs.sort_values("cluster_id").reset_index(drop=True),
        current_gcfs.sort_values("cluster_id").reset_index(drop=True)
    )


@pytest.mark.parametrize("ref_data_id", test_cases)
def test_regression_compositions(ref_data_id):
    """Test compositions output structure"""
    comp = anndata.read_h5ad(f"tests/fixtures/{ref_data_id}_compositions.h5ad")
    current_comp = anndata.read_h5ad(f"tests/test_output/{ref_data_id}_compositions.h5ad")

    assert comp.shape == current_comp.shape
    assert all(comp.var_names == current_comp.var_names)
    assert all(comp.obs_names == current_comp.obs_names)


