# NOTE: Recipes below must remain tab-indented for POSIX make compatibility.
.PHONY: setup demo baseline rf bilstm predict dash realtime clean

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

baseline:
	. .venv/bin/activate && python -m src.train --model baseline --windows 0

rf:
	. .venv/bin/activate && python -m src.train --model random_forest --windows 0

bilstm:
	. .venv/bin/activate && pip install -r requirements-ml.txt && python -m src.train --model bilstm --windows 1 --window-size 20 --stride 5

predict:
	. .venv/bin/activate && python -m src.predict --simulate 1 --model random_forest

dash:
	. .venv/bin/activate && python frontend/dashboard.py

realtime:
	. .venv/bin/activate && python -m src.realtime --model bilstm

demo: baseline predict dash

clean:
	rm -rf results/*
