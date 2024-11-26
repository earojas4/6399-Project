!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="_____________________")
project = rf.workspace("project-4ohwz").project("combined-nf2nh")
version = project.version(1)
dataset = version.download("retinanet")
