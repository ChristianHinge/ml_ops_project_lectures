from azureml.core import Workspace, compute
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Experiment
import joblib

ws = Workspace.from_config("src/cloud/config.json")
compute_target = ws.compute_targets['mnist-droplet']
print()
env = Environment.from_pip_requirements('aml_env', 'src/cloud/aml.txt')

config = ScriptRunConfig(
    environment=env,  # set the python environment
    source_directory='.',
    script='src/models/main.py',
    compute_target = compute_target,
    arguments = ["train"]
)

exp = Experiment(ws, 'test5')
run = exp.submit(config)
print(run.get_portal_url())


