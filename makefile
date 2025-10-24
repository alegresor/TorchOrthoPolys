test: 
	pytest --doctest-modules torchorthopolys/ --no-header

testaccept:
	pytest --doctest-modules torchorthopolys/ --no-header --accept

docplots:
	python plot.py 
	cp polys.svg docs/polys.svg

documls:
	@pyreverse -k torchorthopolys/ -o svg 1>/dev/null && mv classes.svg docs/classes.svg

docsetup: docplots documls

docserve: 
	mkdocs serve --livereload

doc: docsetup docserve
	