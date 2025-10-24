test: 
	pytest --doctest-modules torchorthopolys/ --no-header

testaccept:
	pytest --doctest-modules torchorthopolys/ --no-header --accept

doc:
	python plot.py 
	cp polys.svg docs/polys.svg
	mkdocs serve --livereload