install:
	python -m pip install -r requirements.txt

train:
	python train.py

test:
	pytest -q

run:
	streamlit run app.py
