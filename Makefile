
install:
	sudo apt install fluidsynth
	pip install --upgrade pip
	pip install --upgrade -r requirements.txt

train:
	python src/train.py 

