AUTOTRADER_IMAGE_VERSION := 1.0
IMAGE_NAME := agheva-autotrader-image
CONTAINER_NAME := autotrader-dev
IMAGE_VERSION := ${AUTOTRADER_IMAGE_VERSION}
DOCKER_FILE := autotrader.dockerfile

image: ${DOCKER_FILE}
	docker build -t ${IMAGE_NAME}:${IMAGE_VERSION} -f ${DOCKER_FILE} .

run:
	docker run -d \
          --add-host=host.docker.internal:host-gateway \
          --name ${CONTAINER_NAME} -v ./:/mnt/src -t ${IMAGE_NAME}:${IMAGE_VERSION}

connect:
	docker exec -it ${CONTAINER_NAME} /bin/bash

kill:
	docker stop ${CONTAINER_NAME}
	docker rm ${CONTAINER_NAME}
