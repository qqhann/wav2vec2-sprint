#

Dockerhub available at <https://hub.docker.com/r/patilsuraj/hf-wav2vec>

to build the docker :

```shell
docker build -t hf-wav2vec-sprint -f Dockerfile .
```

to push it to dockerhub
First create a repository on dockerhub

```shell
docker tag hf-wav2vec-sprint your-dockerhub-user/hf-wav2vec-sprint
```

to push it to dockerhub

```shell
docker push your-dockerhub-user/hf-wav2vec-sprint
```
