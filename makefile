test: 
	pytest --doctest-modules torchorthopolys/ --no-header

testaccept:
	pytest --doctest-modules torchorthopolys/ --no-header --accept

doc:
	mkdocs serve --livereload