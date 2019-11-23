# PyTorch implementation of Factorization Machine(FM) BASKET-SENSITIVE FACTORIZATION MACHINE(BFM).

* Paper
  * [Factorization Machines](https://ieeexplore.ieee.org/abstract/document/5694074)
  * [Basket-Sensitive Personalized Item Recommendation](https://www.ijcai.org/proceedings/2017/286)

## How to build a docker container.

CPU ver with docker-compose

```bash
# Building a container
./start.sh
# Enter a docker container
sudo docker exec -it fm zsh
```

GPU ver

```bash
# Building a container
./start_gpu.sh
# Enter a docker container
sudo docker exec -it fm_gpu zsh
```

