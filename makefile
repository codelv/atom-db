docs:
	cd docs
	make html
isort:
	isort atomdb
	isort tests
typecheck:
	mypy atomdb --ignore-missing-imports
lintcheck:
	flake8 --ignore=E501 atomdb
	flake8 --ignore=E501 tests
reformat:
	black atomdb
	black tests
test:
	pytest -v tests --cov atomdb --cov-report xml --asyncio-mode auto

precommit: isort reformat typecheck lintcheck
