# Install directly as standalone python script rather than as a package
BASEDIR := /usr/local
BINDIR  := $(BASEDIR)/bin
DATADIR := $(BASEDIR)/share/aroma

.PHONY: info
info:
	$(info type "make install" to install icaaroma to $(BASEDIR))

install:
	install -T -m 0755 icaaroma/aroma.py $(BINDIR)/aroma
	mkdir -p $(DATADIR)
	install -m 0644 icaaroma/data/mask_csf.nii.gz $(DATADIR)
	install -m 0644 icaaroma/data/mask_edge.nii.gz $(DATADIR)
	install -m 0644 icaaroma/data/mask_out.nii.gz $(DATADIR)

.PHONY: test
test:
	(cd test; ./nosetests test_aroma.py)

clean:
	rm -rf build/ dist/ *.pyc __pycache__ */*.pyc */__pycache__ test/out/ *.egg-info
