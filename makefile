docs:
	cd docs
	make html
isort:
	isort atomdb
	isort tests
typecheck:
	mypy atomdb --ignore-missing-imports
lintcheck:
	flake8 --ignore=E501,W503  atomdb tests
reformat:
	black atomdb tests
test:
	pytest -v tests --cov atomdb --cov-report xml --asyncio-mode auto

precommit: isort reformat typecheck lintcheck
