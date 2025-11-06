test:
	pytest tests/ -v

run-igua: 
	mkdir -p tests/test_output
	igua -i tests/fixtures/ecoli_input_metadata.tsv --output tests/test_output/ecoli_gcfs.tsv --compositions tests/test_output/ecoli_compositions.h5ad --features tests/test_output/ecoli_features.fa
	igua -i tests/fixtures/secretion_systems_input_metadata_sample.tsv --output tests/test_output/secretion_small_gcfs.tsv --compositions tests/test_output/secretion_small_compositions.h5ad --features tests/test_output/secretion_small_features.fa
	igua -i tests/fixtures/emerge_input_metadata_sample15.tsv --output tests/test_output/emerge_gcfs.tsv --compositions tests/test_output/emerge_compositions.h5ad --features tests/test_output/emerge_features.fa


test-regression:
	pytest tests/regression/test_output_regression.py -q --tb=line

test-ref:
	pytest tests/regression/test_ref.py -q --tb=line


clean:
	rm -rf tests/test_output/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf igua/__pycache__/
	rm -rf tests/__pycache__/

all: clean run-igua tests

tests: test-regression test-ref

update-ref:
	# after confirming new behaviour is correct
	cp -rf tests/test_output/* tests/fixtures/
