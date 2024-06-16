
hash = $(shell git log --pretty=format:'%h' -n 1)
-include .env

VERSION:=fc102c83

clean:
	rm -rf ./training_checkpoints
	rm -f ./output/*.mid
	rm -f ./output/*.midi
	rm -f ./output/*.png
	find ./ -name "__pycache__" -exec rm -rf {} \;

install:
	sudo apt install fluidsynth
	pip install --upgrade pip
	pip install --upgrade -r requirements.txt

play:
	fluidsynth \
		data/robbie-v1.0.0/pop/beatle_help.mid

# Step one, remove drum tracks from midi files and make all
# the instruments on to one track
midi_clean:
	mkdir -p ./data/robbie-v1.0.0/clean
	python src/v2/clean.py \
		./data/robbie-v1.0.0 \
		./data/robbie-v1.0.0/clean 
	
	python src/v2/clean.py \
		./data/robbie-v1.0.0/lmd_full \
		./data/robbie-v1.0.0/clean 

	python src/v2/clean.py \
		./data/robbie-v1.0.0/game \
		./data/robbie-v1.0.0/clean 

# Step two, turn the cleaned midi text into UTF-8 text 
# that we can use to train the model
midi_text:
	VERSION=$(VERSION) \
	python src/v2/prep.py

# Step three, train sentencepiece so we have a vocabulary that we 
# can use to train the GPT model
midi_train_token:
	VERSION=$(VERSION) \
	python src/v2/tokenization_train.py

train_sh:
	VERSION=$(VERSION) \
	MINIO_SERVER=$(MINIO_SERVER) \
	MINIO_ACCESS=$(MINIO_ACCESS) \
	MINIO_SECRET=$(MINIO_SECRET) \
	./train.sh

#######################################

# train_v1:
# 	python src/v1/train.py

# inference_v1:
# 	python src/v1/inference.py ./input/melody_75_F#.midi

inference:
	VERSION=$(hash) \
	python src/v2/inference.py ./input/melody_75_F#.midi

#######################################

test_encode_midi:
	python src/v2/tests.py

#######################################

store:
	aws s3 sync --delete ./data/robbie-v1.0.0/ s3://midis.robrohan.com

#######################################

docker_build:
	docker ps; \
	docker build . -t robrohan/songomatic_train

docker_run:
	docker ps; \
	docker run \
		-v $(PWD)/robbie-v1.0.0:/robbie-v1.0.0 \
		-e "VERSION=d15bec4"
		-e "MINIO_SERVER=$(MINIO_SERVER)" \
		-e "MINIO_ACCESS=$(MINIO_ACCESS)" \
		-e "MINIO_SECRET=$(MINIO_SECRET)" \
		robrohan/songomatic_train

docker_push:
	docker ps; \
	docker push robrohan/songomatic_train
