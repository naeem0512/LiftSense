.PHONY: setup demo baseline rf bilstm predict dash realtime clean

MODEL ?= bilstm
SIMULATE ?= 1
CSV ?=
MODEL_PATH ?=
WINDOW_SIZE ?= 20
STRIDE ?= 5

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

baseline:
	. .venv/bin/activate && python -m src.train --model baseline --windows 0

rf:
	. .venv/bin/activate && python -m src.train --model random_forest --windows 0

bilstm:
	. .venv/bin/activate && pip install -r requirements-ml.txt && python -m src.train --model bilstm --windows 1 --window-size $(WINDOW_SIZE) --stride $(STRIDE)

predict:
	. .venv/bin/activate && python -m src.predict $(if $(strip $(CSV)),--csv $(CSV),--simulate $(SIMULATE)) --model $(MODEL) --window-size $(WINDOW_SIZE) --stride $(STRIDE) $(if $(strip $(MODEL_PATH)),--model-path $(MODEL_PATH),)

dash:
	. .venv/bin/activate && python frontend/dashboard.py

realtime:
	. .venv/bin/activate && python -m src.realtime --model $(MODEL) --window-size $(WINDOW_SIZE) --stride $(STRIDE) $(if $(strip $(MODEL_PATH)),--model-path $(MODEL_PATH),)

demo: baseline predict dash

clean:
	rm -rf results/*
