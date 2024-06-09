
hash = $(shell git log --pretty=format:'%h' -n 1)
-include .env

clean:
	rm -rf ./training_checkpoints
	rm -f ./output/*.mid
	rm -f ./output/model.json
	rm -f ./output/*.png

install:
	sudo apt install fluidsynth
	pip install --upgrade pip
	pip install --upgrade -r requirements.txt

play:
	fluidsynth \
		data/robbie-v1.0.0/pop/beatle_help.mid

midi_clean:
	mkdir -p ./data/robbie-v1.0.0/clean
	python src/prep/clean.py \
		./data/robbie-v1.0.0 \
		./data/robbie-v1.0.0/clean 
	
	python src/prep/clean.py \
		./data/robbie-v1.0.0/lmd_full \
		./data/robbie-v1.0.0/clean 

train_sh:
	MINIO_SERVER=$(MINIO_SERVER) \
	MINIO_ACCESS=$(MINIO_ACCESS) \
	MINIO_SECRET=$(MINIO_SECRET) \
	./train.sh

#######################################

train_v1:
	python src/v1/train.py

inference_v1:
	python src/v1/inference.py ./input/melody_75_F#.midi

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
		-e "MINIO_SERVER=$(MINIO_SERVER)" \
		-e "MINIO_ACCESS=$(MINIO_ACCESS)" \
		-e "MINIO_SECRET=$(MINIO_SECRET)" \
		robrohan/songomatic_train

docker_push:
	docker ps; \
	docker push robrohan/songomatic_train