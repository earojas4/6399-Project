!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="*************")
project = rf.workspace("project-4ohwz").project("plant-diseases-detection-dataset-ground")
version = project.version(3)
dataset = version.download("retinanet")
