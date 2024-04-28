
clean:
	rm -rf ./training_checkpoints
	rm -f ./output/*.mid

install:
	sudo apt install fluidsynth
	pip install --upgrade pip
	pip install --upgrade -r requirements.txt

play:
	fluidsynth \
		data/robbie-v1.0.0/pop/beatle_help.mid

verify:
	python src/verify.py \
		data/robbie-v1.0.0/pop/beatle_help.mid

train:
	python src/train.py

inference:
	python src/inference.py

store:
	aws s3 sync --delete ./data/robbie-v1.0.0/ s3://midis.robrohan.com
