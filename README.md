# parallel-xgbregressor
Usando dasf-core, dasf-seismic e dask para paralelizar o treinamento e o model serving de um XGBoost Regressor para o cálculo de atributos sísmicos.

## Instalação

Para instalar o `dasf-seismic-lite`:

1. Clone o repositório do [`dasf-core`](https://github.com/discovery-unicamp/dasf-core) e crie uma imagem docker de cpu. Para isso, você pode executar os seguintes comandos **(em uma pasta fora deste repositório)**, que irá gerar uma imagem docker chamada `dasf:cpu`:

```bash
git clone https://github.com/discovery-unicamp/dasf-core.git
cd dasf-core/build
CONTAINER_CMD=docker ./build_container.sh cpu
```

2. Clone este repositório e crie uma imagem docker que irá instalar o `dasf-seismic-lite`. Neste repositório, foi disponibilizado um *script* para facilitar a criação da imagem docker. Para isso, entre dentro do diretório raiz deste repositório e execute o *script* `build_docker.sh`, que irá gerar uma imagem docker chamada `dasf-seismic:cpu`, conforme abaixo:

```bash
./build_docker.sh
```

## Como usar?
Para testar este código você tem as duas opções descritas abaixo.
## Execução único nó

1. Dentro da pasta `data` deste repositório, você deve colocar seus arquivos sísmicos `npy` ou `zarr` de treino.

2. Execute o comando abaixo para executar o *script* `train-model.py` dentro do container:
    
```bash
./bash_scripts/train-model.sh
```

## Execução multi-nós

1. Instancie um *dask scheduler* (o endereço do *scheduler* será mostrado no terminal, será algo semelhante a `tcp://192.168.1.164:8786`):

```bash
./bash_scripts/dask-scheduler.sh
```

2. Instancie um *dask worker* e conecte-o ao *scheduler* criado no passo anterior (substitua `<scheduler_address>` pelo endereço do *scheduler*, será algo semelhante a `tcp://192.168.1.164:8786`)

```bash
./bash_scripts/dask-worker <scheduler_address>
```

3. Execute a implementação do `train-model.py` passando o endereço do *scheduler* (substitua `<scheduler_address>` pelo endereço do *scheduler*, será algo semelhante a `tcp://192.168.1.164:8786`)

```bash
./bash_scripts/train-model.sh <scheduler_address>
```

**NOTA 1**: Lembre-se de executar os comandos acima na pasta raiz deste repositório.

**NOTA 2**: Os scripts em `./bash_scripts/` rodam todas as combinações de janelas de vizinhos do extrator de features, e treinam para todos os atributos sísmicos. Se você quer rodar apenas uma configuração específica, veja o comando docker no repositório do [dasf-seismic-lite](https://github.com/otavioon/dasf-seismic-lite).

**Nota 3**: Também tem disponível o script `train-model-hyperparam-search.py`. Ele foi uma tentativa de seguir boas práticas de data science, fazendo o split do dataset total e cross fold validation para buscar um bom espaço de hiperparâmetros. Estranhamente o GridSearchCV do dask não lidou bem com o objeto XGBRegressor do dasf-core. Tive um bug que não consegui resolver.

## Acknowledgements
Agradeço ao professor Edson Borin do IC - Unicamp & equipe [dasf-core](https://github.com/discovery-unicamp/dasf-core) pela principal ferramenta deste projeto.
