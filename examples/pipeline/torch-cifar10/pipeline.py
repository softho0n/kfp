import os
from typing import TypedDict, cast
import kfp
import requests


class EnvironmentVariableError(EnvironmentError):
    """Exception class for Kubeflow authentication"""

    def __init__(self, env_name: str) -> None:
        super().__init__(f"The environment variable {env_name} is not set.")


class OperationOutput(TypedDict, total=False):
    """Type definition for output dictionary element"""

    preprocess: kfp.dsl.ContainerOp
    train: kfp.dsl.ContainerOp
    evaluation: kfp.dsl.ContainerOp
    serving: kfp.dsl.ContainerOp


class KFPClient:
    """Kubeflow authentication util class"""

    def __init__(self):
        host = os.getenv("KF_HOST")
        username = os.getenv("KF_USERNAME")
        password = os.getenv("KF_PASSWD")
        namespace = os.getenv("KF_NAMESPACE")

        if host is None:
            raise EnvironmentVariableError("KF_HOST")
        if username is None:
            raise EnvironmentVariableError("KF_USERNAME")
        if password is None:
            raise EnvironmentVariableError("KF_PASSWD")
        if namespace is None:
            raise EnvironmentVariableError("KF_NAMESPACE")

        self.__host = host
        self.__username = username
        self.__password = password
        self.__namespace = namespace

    def get_kfp_client(self):  # pylint: disable=missing-function-docstring
        session = requests.Session()
        response = session.get(self.__host)

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {"login": self.__username, "password": self.__password}
        session.post(response.url, headers=headers, data=data)
        session_cookie = session.cookies.get_dict()["authservice_session"]

        return kfp.Client(
            host=f"{self.__host}/pipeline",
            cookies=f"authservice_session={session_cookie}",
            namespace=self.__namespace,
        )


def preprocess_op(
    name: str, input_dir: str, output_dir: str, pvolume: kfp.dsl.PipelineVolume
):
    """Preprocess component

    Args:
        name (str): pod name
        input_dir (str): input directory
        output_dir (str): output directory
        pvolume (kfp.dsl.PipelineVolume): persistent volume for pod usage

    Returns:
        _type_: kfp.dsl.ContainerOp
    """
    return kfp.dsl.ContainerOp(
        name=name,
        image="softhoon/torch-preprocess-image:latest",
        command=["python3", "preprocess.py"],
        arguments=[
            "--input_dir",
            input_dir,
            "--output_dir",
            output_dir,
        ],
        file_outputs={"output": "/output.txt"},
        pvolumes={"/mnt/workspace": pvolume},
    )


def train_op(
    name: str,
    input_dir: str,
    output_dir: str,
    epochs: int,
    model_name: str,
    model_version: int,
    pvolume: kfp.dsl.PipelineVolume,
):
    """Training component

    Args:
        name (str): pod name
        input_dir (str): input directory
        output_dir (str): output directory
        epochs (int): number of epochs for training
        model_name (str): model name
        model_version (int): model version
        pvolume (kfp.dsl.PipelineVolume): persistent volume for pod usage

    Returns:
        _type_: kfp.dsl.ContainerOp
    """
    return kfp.dsl.ContainerOp(
        name=name,
        image="softhoon/torch-train-image:latest",
        command=["python3", "train.py"],
        arguments=[
            "--input_dir",
            input_dir,
            "--output_dir",
            output_dir,
            "--epochs",
            epochs,
            "--model_name",
            model_name,
            "--model_version",
            model_version,
        ],
        file_outputs={"output": "/output.txt"},
        pvolumes={"/mnt/workspace": pvolume},
    )


def evaluation_op(
    name: str,
    input_dir: str,
    output_dir: str,
    preprocessed_data_dir: str,
    model_name: str,
    model_version: int,
    pvolume: kfp.dsl.PipelineVolume,
):
    """Evaluation component

    Args:
        name (str): pod name
        input_dir (str): input directory
        output_dir (str): output directory
        preprocessed_data_dir (str): path of preprocessed data
        model_name (str): model name
        model_version (int): model version
        pvolume (kfp.dsl.PipelineVolume): persistent volume for pod usage

    Returns:
        _type_: kfp.dsl.ContainerOp
    """
    return kfp.dsl.ContainerOp(
        name=name,
        image="softhoon/torch-evaluation-image:latest",
        command=["python3", "evaluation.py"],
        arguments=[
            "--input_dir",
            input_dir,
            "--output_dir",
            output_dir,
            "--preprocessed_data_dir",
            preprocessed_data_dir,
            "--model_name",
            model_name,
            "--model_version",
            model_version,
        ],
        file_outputs={"output": "/output.txt"},
        pvolumes={"/mnt/workspace": pvolume},
    )


def serving_op(
    name: str,
    input_dir: str,
    output_dir: str,
    model_name: str,
    model_version: int,
    pvolume: kfp.dsl.PipelineVolume,
):
    """Serving component

    Args:
        name (str): pod name
        input_dir (str): input directory
        output_dir (str): output directory
        model_name (str): model name
        model_version (int): model version
        pvolume (kfp.dsl.PipelineVolume): persistent volume for pod usage

    Returns:
        _type_: kfp.dsl.ContainerOp
    """
    return kfp.dsl.ContainerOp(
        name=name,
        image="softhoon/torch-serving-image:latest",
        command=["python3", "serving.py"],
        arguments=[
            "--input_dir",
            input_dir,
            "--output_dir",
            output_dir,
            "--model_name",
            model_name,
            "--model_version",
            model_version,
        ],
        file_outputs={"output": "/output.txt"},
        pvolumes={"/mnt/workspace": pvolume},
    )


@kfp.dsl.pipeline(
    name="softhoon pytorch pipeline example",
    description="softhoon pytorch pipeline example",
)
def torch_pipeline(  # pylint: disable=missing-function-docstring
    raw_data_dir: str = "/mnt/workspace/raw_data",
    processed_data_dir: str = "/mnt/workspace/processed_data",
    model_dir: str = "/mnt/workspace/saved_model",
    epochs: int = 50,
    model_name: str = "softhoon-pytorch-cifar10",
    model_version: int = 1,
):
    """Pipeline execution function

    Args:
        raw_data_dir(str, optional):
            Path of raw data directory.
            Defaults to "/mnt/workspace/raw_data".
        processed_data_dir (str, optional):
            Path of processed data directory.
            Defaults to "/mnt/workspace/processed_data".
        model_dir (str, optional):
            Model directory.
            Defaults to "/mnt/workspace/saved_model".
        epochs (int, optional):
            Number of epochs.
            Defaults to 50.
        model_name (str, optional):
            Model name.
            Defaults to "softhoon-pytorch-cifar10".
        model_version (int, optional):
            Model version.
            Defaults to 1.

    Returns:
        _type_: output dictionary of each component
    """
    vop = kfp.dsl.VolumeOp(
        name="softhoon",
        resource_name="softhoon-resource",
        size="1Gi",
        modes=kfp.dsl.VOLUME_MODE_RWO,
    )

    op_dict: OperationOutput = {}

    op_dict["preprocess"] = preprocess_op(
        "preprocess", raw_data_dir, processed_data_dir, vop.volume
    )

    op_dict["train"] = train_op(
        "train",
        cast(str, op_dict["preprocess"].output),
        model_dir,
        epochs,
        model_name,
        model_version,
        cast(kfp.dsl.PipelineVolume, op_dict["preprocess"].pvolume),
    )

    op_dict["evaluation"] = evaluation_op(
        "test",
        cast(str, op_dict["train"].output),
        "",
        processed_data_dir,
        model_name,
        model_version,
        cast(kfp.dsl.PipelineVolume, op_dict["train"].pvolume),
    )

    op_dict["serving"] = serving_op(
        "serving",
        cast(str, op_dict["train"].output),
        "",
        model_name,
        model_version,
        cast(kfp.dsl.PipelineVolume, op_dict["train"].pvolume),
    )

    return op_dict


if __name__ == "__main__":
    client = KFPClient().get_kfp_client()
    client.create_run_from_pipeline_func(torch_pipeline, arguments={})
