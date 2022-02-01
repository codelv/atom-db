docs:
	cd docs
	make html
isort:
	isort atomdb
typecheck:
	mypy atomdb --ignore-missing-imports

reformat:
	black atomdb
	black tests

precommit: isort reformat typecheck
