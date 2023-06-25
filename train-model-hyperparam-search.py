import argparse
import time
from typing import Callable, Literal, Optional, Tuple

import dasf.ml.xgboost as xgboost
import dask.array as da
import numpy as np
import dask_ml
from dasf.datasets import Dataset
from dasf.pipeline import Pipeline
from dasf.pipeline.executors import DaskPipelineExecutor
from dasf.transforms import ArraysToDataFrame
from dasf.utils.decorators import task_handler
from dasf_seismic.attributes.complex_trace import (CosineInstantaneousPhase,
                                                   Envelope,
                                                   InstantaneousFrequency)

from utils_tmp import (CustomArraysToDataFrame, xCustomArraysToDataFrame,
                   yCustomArraysToDataFrame, TrainTestSplit)


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

    executor.client.upload_file("utils_tmp.py")
    return executor


def create_pipeline(
    dataset_path: str,
    executor: DaskPipelineExecutor,
    pipeline_save_location: str = None,
    attribute: Literal["ENVELOPE", "INST-FREQ", "COS-INST-PHASE"] = "COS-INST-PHASE",
    samples_window: int = 1,
    traces_window: int = 1,
    inlines_window: int = 1,
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
    dataset = MyDataset(name="F3 dataset", data_path=dataset_path)
    envelope = Envelope()
    inst_freq = InstantaneousFrequency()
    cos_inst = CosineInstantaneousPhase()

    arrays2df = ArraysToDataFrame()

    xgbregressor = xgboost.XGBRegressor()

    # grid search esta com bug
    grid_search = dask_ml.model_selection.GridSearchCV(
        estimator=xgbregressor._XGBRegressor__xgb_mcpu,
        param_grid=dict(
            grow_policy=["depthwise", "lossguide"],
            booster=['gbtree', 'dart'],
            reg_alpha=[i/10 for i in range(1, 10+1)],
        ),
        scoring="r2",
        cv=4
    )

    get_y = {
        "ENVELOPE": envelope,
        "INST-FREQ": inst_freq,
        "COS-INST-PHASE": cos_inst,
    }
    # Compondo o pipeline
    pipeline = Pipeline(name="F3 seismic attributes", executor=executor)
    pipeline.add(dataset)

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
    train_test_split = TrainTestSplit(x=samples_window, y=traces_window, z=inlines_window)

    pipeline.add(get_y[attribute], X=dataset)
    pipeline.add(
        arrays2df,
        dataset=dataset,
        **{f"x{i}": res[i] for i in range(len(res))},
        attr=get_y[attribute],
    )
    pipeline.add(train_test_split, X=arrays2df)
    pipeline.add(x_arr2df, X=train_test_split)
    pipeline.add(y_arr2df, X=train_test_split)
    pipeline.add(xgbregressor.fit, X=x_arr2df, y=y_arr2df)

    if pipeline_save_location is not None:
        pipeline._dag_g.render(outfile=pipeline_save_location, cleanup=True)

    # Retorna o pipeline e o operador xgbregressor, donde os resultados serão obtidos
    return pipeline, xgbregressor.fit


def run(
    pipeline: Pipeline, last_node: Optional[Callable], model_file_name: str
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
    start = time.time()
    try:
        pipeline.run()
        res = pipeline.get_result_from(last_node)
        res.save_model(model_file_name)
    except:
        res = None
    end = time.time()

    print(f"Feito! Tempo de execução: {end - start:.2f} s")
    return res, end-start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline")

    parser.add_argument(
        "--attribute",
        help="Nome do atributo a ser usado para treinar o modelo.",
        choices=["ENVELOPE", "INST-FREQ", "COS-INST-PHASE"],
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
        help="Nome do arquivo de saída onde deve ser gravado o modelo treinado.",
        type=str,
    )
    parser.add_argument(
        "--save-pipeline-fig",
        type=str,
        default=None,
        help="Local para salvar a figura do pipeline",
    )

    args = parser.parse_args()

    # Criamos o executor
    executor = create_executor(args.address)
    # Depois o pipeline
    pipeline, last_node = create_pipeline(
        args.data,
        executor,
        pipeline_save_location=args.save_pipeline_fig,
        attribute=args.attribute,
        samples_window=args.samples_window,
        traces_window=args.trace_window,
        inlines_window=args.inline_window,
    )
    # Executamos e pegamos o resultado
    res, time_elapsed = run(pipeline, last_node, args.output)
