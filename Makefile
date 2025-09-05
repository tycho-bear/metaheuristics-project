IMAGE_NAME = metaheuristics-project

# default arguments
ARGS = de ackley

# build command
build:
	docker build -t $(IMAGE_NAME) .

# run command
run:
	docker run $(IMAGE_NAME) $(ARGS)

# two in one
start: build run

# clean up
clean:
	docker system prune -f
