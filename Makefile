# Install directly as standalone python script rather than as a package
BASEDIR := /usr/local
BINDIR  := $(BASEDIR)/bin
DATADIR := $(BASEDIR)/share/aroma

.PHONY: info build install standalone tests clean
info:
	$(info type "make install" to install as a python package or "make standalone" to install icaaroma directly to $(BINDIR))

build:
	python setup.py build

install:
	python setup.py install

standalone:
	install -T -m 0755 icaaroma/aroma.py $(BINDIR)/aroma
	mkdir -p $(DATADIR)
	install -m 0644 icaaroma/data/mask_csf.nii.gz $(DATADIR)
	install -m 0644 icaaroma/data/mask_edge.nii.gz $(DATADIR)
	install -m 0644 icaaroma/data/mask_out.nii.gz $(DATADIR)

tests:
	(cd test; make tests)

clean:
	rm -rf build/ dist/ *.pyc __pycache__ */*.pyc */__pycache__ *.egg-info
	(cd ipynb; make clean)
	(cd test; make clean)
