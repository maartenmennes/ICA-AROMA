MAKEFLAGS += --no-builtin-rules
.SUFFIXES:
.PHONY: test

test: tests/output.txt
tests/refin/mask.nii.gz:
	cd tests && curl -L https://github.com/rtrhd/test-data/raw/master/icaaroma/0.4.0/refin.tar.bz2 | bunzip2 | tar x
tests/output.txt: tests/refin/mask.nii.gz $(wildcard tests/*.py) $(wildcard ica_aroma/*py)
	python -m pytest -s tests/ | tee $@
