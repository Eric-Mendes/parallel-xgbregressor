import argparse
import time
from typing import Callable, Literal, Optional, Tuple
import json
import dask.array as da
from dask.distributed import Client, performance_report
import numpy as np
from dasf.datasets import Dataset
from dasf.pipeline import Pipeline
from dasf.pipeline.executors import DaskPipelineExecutor
from dasf.transforms import ArraysToDataFrame
from dasf.utils.decorators import task_handler
from dasf_seismic.attributes.complex_trace import (CosineInstantaneousPhase,
                                                   Envelope,
                                                   InstantaneousFrequency)
import pickle

from utils import (CustomArraysToDataFrame, xCustomArraysToDataFrame,
                   yCustomArraysToDataFrame)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

class MyDataset(Dataset):
    """Classe para carregar dados de um arquivo .zarr"""

    def __init__(
        self, name: str, data_path: str, chunks: dict = {0: "auto", 1: "auto", 2: -1}
    ):
        """Instancia um objeto da classe MyDataset

        Parameters
        ----------
        name : str
            Nome simbolicamente associado ao dataset
        data_path : str
            Caminho para o arquivo .zarr
        chunks: dict
            Tamanho dos chunks para o dask.array
        """
        super().__init__(name=name)
        self.data_path = data_path
        self.chunks = chunks

    def _lazy_load_cpu(self):
        return da.from_zarr(self.data_path, chunks=self.chunks)

    def _load_cpu(self):
        return np.load(self.data_path)

    @task_handler
    def load(self):
        ...


def create_executor(address: str = None) -> DaskPipelineExecutor:
    """Cria um DASK executor

    Parameters
    ----------
    address : str, optional
        Endereço do Scheduler, by default None

    Returns
    -------
    DaskPipelineExecutor
        Um executor Dask
    """
    executor = None
    if address is not None:
        addr = ":".join(address.split(":")[:2])
        port = str(address.split(":")[-1])
        print(f"Criando executor. Endereço: {addr}, porta: {port}")
        executor = DaskPipelineExecutor(
            local=False, use_gpu=False, address=addr, port=port
        )
    else:
        executor = DaskPipelineExecutor(local=True, use_gpu=False)

    executor.client.upload_file("utils.py")
    return executor


def create_pipeline(
    dataset_path: str,
    executor: DaskPipelineExecutor,
    samples_window: int = 1,
    traces_window: int = 1,
    inlines_window: int = 1,
    model_file_name: str = "model.json"
) -> Tuple[Pipeline, Callable]:
    """Cria o pipeline DASF para ser executado

    Parameters
    ----------[d, d1, d2, d3]
    dataset_path : str
        Caminho para o arquivo .npy
    executor : DaskPipelineExecutor
        Executor Dask

    Returns
    -------
    Tuple[Pipeline, Callable]
        Uma tupla, onde o primeiro elemento é o pipeline e o segundo é último operador (model.fit_predict),
        de onde os resultados serão obtidos.
    """
    print("Criando pipeline....")
    # Declarando os operadores necessários
    dataset = MyDataset(name="F3 test dataset", data_path=dataset_path)

    arrays2df = ArraysToDataFrame()
    cos_inst = CosineInstantaneousPhase()
    xgbregressor = xgb.dask.DaskXGBRegressor()
    xgbregressor.load_model(model_file_name)

    # Compondo o pipeline
    pipeline = Pipeline(name="F3 seismic attributes", executor=executor)
    pipeline.add(dataset)
    if samples_window == 0 and traces_window == 0 and inlines_window == 0:
        res = [dataset]
    else:
        res = []
        for shift in range(-samples_window, samples_window + 1, 1):
            if shift != 0:
                a2df = CustomArraysToDataFrame(shift=shift, axis=2)
                pipeline.add(a2df, dataset=dataset)
                res.append(a2df)

        for shift in range(-traces_window, traces_window + 1, 1):
            if shift != 0:
                a2df = CustomArraysToDataFrame(shift=shift, axis=1)
                pipeline.add(a2df, dataset=dataset)
                res.append(a2df)

        for shift in range(-inlines_window, inlines_window + 1, 1):
            if shift != 0:
                a2df = CustomArraysToDataFrame(shift=shift, axis=0)
                pipeline.add(a2df, dataset=dataset)
                res.append(a2df)
    
    x_arr2df = xCustomArraysToDataFrame()
    y_arr2df = yCustomArraysToDataFrame()

    pipeline.add(cos_inst, X=dataset)
    pipeline.add(
        arrays2df,
        dataset=dataset,
        **{f"x{i}": res[i] for i in range(len(res))},
        attr=cos_inst
    )
    pipeline.add(x_arr2df, X=arrays2df)
    pipeline.add(y_arr2df, X=arrays2df)
    pipeline.add(xgbregressor.predict, X=x_arr2df)

    y_true = y_arr2df
    # Retorna o pipeline e o operador xgbregressor, donde os resultados serão obtidos
    return pipeline, xgbregressor.predict, y_true


def run(
    pipeline: Pipeline, last_node: Optional[Callable], output: str, y_true
) -> np.ndarray:
    """Executa o pipeline e retorna o resultado

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline a ser executado
    last_node : Callable
        Último operador do pipeline, de onde os resultados serão obtidos

    Returns
    -------
    np.ndarray
        NumPy array com os resultados
    """
    print("Executando pipeline")
    try:
        start = time.time()
        pipeline.run()
        y_pred = pipeline.get_result_from(last_node).compute()
        y_true_ = pipeline.get_result_from(y_true).compute()
        
        metrics = {"r2": r2_score(y_true_, y_pred), "mae": mean_absolute_error(y_true_, y_pred), "mse": mean_squared_error(y_true_, y_pred)}

        with open(f"metrics.json", 'w') as file:
            json.dump(metrics, file)

        end = time.time()
        np.save(output, y_pred)

        print(f"Feito! Tempo de execução: {end - start:.2f} s")
        time_elapsed = end - start
        return y_pred, time_elapsed
    except:
        return None, -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline")

    parser.add_argument(
        "--ml-model",
        help="Nome do atributo a ser usado para treinar o modelo.",
        type=str
    )
    parser.add_argument(
        "--data", help="Nome do arquivo com o dado sísmico de entrada.", type=str
    )
    parser.add_argument(
        "--inline-window", help="Número de vizinhos na dimensão das inlines.", type=int
    )
    parser.add_argument(
        "--trace-window",
        help="Número de vizinhos na dimensão dos traços de uma inline.",
        type=int,
    )
    parser.add_argument(
        "--samples-window",
        help="Número de vizinhos na dimensão das amostras de um traço.",
        type=int,
    )
    parser.add_argument(
        "--address",
        help="Endereço do dask scheduler para execução do código.",
        type=str,
    )
    parser.add_argument(
        "--output",
        help="Nome do arquivo de saída onde deve ser gravado o atributo sísmico produzido.",
        type=str,
    )
    args = parser.parse_args()
    # Criamos o executor
    executor = create_executor(args.address)
    # Depois o pipeline
    pipeline, last_node, y_true = create_pipeline(
        args.data,
        executor,
        samples_window=args.samples_window,
        traces_window=args.trace_window,
        inlines_window=args.inline_window,
        model_file_name=args.ml_model
    )
    # Executamos e pegamos o resultado
    res, time_elapsed = run(pipeline, last_node, args.output, y_true)

    # Podemos fazer o reshape e printar a primeira inline
    if res is not None:
        res = res.reshape((401, 701, 255))
        import matplotlib.pyplot as plt

        inline = f"{args.inline_window}-{args.trace_window}-{args.samples_window}-inline.png"
        plt.imsave(inline, res[0], cmap="viridis")
        print(f"Figura da inline salva em {inline}")
