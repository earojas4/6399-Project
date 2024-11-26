!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="__________________________")
project = rf.workspace("project-4ohwz").project("plant-diseases-detection-aerial")
version = project.version(4)
dataset = version.download("retinanet")
