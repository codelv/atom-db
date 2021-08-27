docs:
	cd docs
	make html
isort:
	isort atomdb
typecheck:
	mypy atomdb --ignore-missing-imports

reformat:
	black atomdb

precommit: isort reformat typecheck
