install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload

lint:
	ruff check .
	black --check .

test:
	pytest
