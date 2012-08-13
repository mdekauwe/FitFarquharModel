# Mainly because I can't be bothered typing python setup blah
build:
	python setup.py build
install:
	cat files.txt | xargs rm -rf
	python setup.py install --record files.txt
	rm files.txt
clean:
	python setup.py clean 
	cat files.txt | xargs rm -rf
	
