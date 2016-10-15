tests: refin refout
	./nosetests test_aroma.py

refin:
	curl -L https://github.com/rtrhd/test-data/raw/master/icaaroma/0.4.0/refin.tar.bz2 | tar x

refout:
	curl -L https://github.com/rtrhd/test-data/raw/master/icaaroma/0.4.0/refout.tar.bz2 | tar x

clean:
	rm -rf refin/ refout/ out/
