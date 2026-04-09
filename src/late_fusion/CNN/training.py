import yaml
import datetime
from ultralytics import YOLO
import os

def run_train(config_path="src/late_fusion/CNN/config.yaml"):
    """Run the training process for a YOLO model based on a given configuration file.
    The configuration file should contain the following keys: 
    - experiment_name: Name of the MLflow experiment
    - model_variant: The YOLO model variant to use (e.g., 'yolov8n.pt')
    - train_params: Dictionary of training parameters (e.g., epochs, batch size, etc.)
    """
    if not os.path.exists(config_path):
        print(f"Error: File {config_path} not found.")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    run_id = f"{config['model_variant']}_{timestamp}"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = config['experiment_name']
    os.environ["MLFLOW_RUN_NAME"] = run_id
    os.environ["REPORT_TO"] = "mlflow" 
    model = YOLO(config['model_variant'])
    print(f"Starting training : {os.environ['MLFLOW_RUN_NAME']}")
    results = model.train(
        data=os.path.abspath(config['data_path']),
        project="runs/train",
        name=run_id,
        **config['train_params']
 )

    print("Training finished")

if __name__ == "__main__":
    run_train()