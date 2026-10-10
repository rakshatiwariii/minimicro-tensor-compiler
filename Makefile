.PHONY: test bench clean

test:
	pytest tests/

bench:
	python benchmarks/bench_fusion.py

clean:
	rm -rf __pycache__ .pytest_cache src/__pycache__ tests/__pycache__
