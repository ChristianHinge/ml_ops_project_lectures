import azureml.core
from azureml.core import Workspace
import os
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.model import Model

# Load the workspace from the saved config file
ws = Workspace.from_config("src/cloud/config.json")
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

model = ws.models['diabetes_model']
print(model.name, 'version', model.version)

# Set path for scoring script
script_file = os.path.join("src/cloud/score_mnist.py")

from azureml.core import Environment

env = Environment.from_pip_requirements('aml_env', 'src/cloud/aml_req_deploy.txt')

# Configure the scoring environment
inference_config = InferenceConfig(entry_script=script_file,
                                   environment=env)

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
service_name = "mnist-service2"
service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)

#service.wait_for_deployment(True)
#print(service.state)