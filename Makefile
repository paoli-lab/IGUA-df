test:
	pytest tests/ -v

run-igua: 
	mkdir -p test_output
	igua -i tests/data/strains_metadata.tsv --output test_output/ecoli_gcfs.tsv --compositions test_output/ecoli_compositions.h5ad --features test_output/ecoli_features.fa
	igua -i tests/data/secretion_systems_metadata_short.tsv --output test_output/secretion_small_gcfs.tsv --compositions test_output/secretion_small_compositions.h5ad --features test_output/secretion_small_features.fa
	igua -i tests/data/emerge_sample_15_metadata.tsv --output test_output/emerge_gcfs.tsv --compositions test_output/emerge_compositions.h5ad --features test_output/emerge_features.fa


test-regression:
	pytest tests/test_output_regression.py -q --tb=line

test-ref:
	pytest tests/test_ref.py -q --tb=line


clean:
	rm -rf test_output/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf igua/__pycache__/
	rm -rf tests/__pycache__/
	rm -f defense_extraction_errors.log


all: clean run-igua tests

tests: test-regression test-ref

update-ref:
	# after confirming new behaviour is correct
	cp -rf test_output/* tests/data/
