```markdown
# Product Type Classification API

This repository contains a deployed API for classifying product types based on images using a trained ONNX model. The API is built on Azure Machine Learning and provides real-time predictions for various jewelry categories.

## Table of Contents
- [Overview](#overview)
- [Model Deployment](#model-deployment)
- [API Functionality](#api-functionality)
- [Usage](#usage)
- [Requirements](#requirements)
- [How to Contribute](#how-to-contribute)
- [License](#license)

## Overview

The API leverages a quantized ONNX model that has been fine-tuned for product type classification. It allows users to submit an image URL and receive predictions for the top 3 product categories, along with their corresponding probabilities.

## Model Deployment

### Deployment Steps

1. **Model Conversion**: The Keras model (`.h5` format) was converted to the ONNX format for improved inference performance and compatibility.

2. **Model Quantization**: The ONNX model was quantized to reduce its size and improve the inference speed.

3. **Model Registration**: The ONNX model was registered in the Azure Machine Learning workspace using the following code:

   ```python
   from azureml.core import Model
   from azureml.core.workspace import Workspace

   # Load the Azure Machine Learning workspace
   ws = Workspace.from_config()

   # Register the ONNX model
   model = Model.register(workspace=ws,
                          model_name='prodtype-classification-best-onnx-model',
                          model_path='models/efficientnetb4_finetune_model_quantized.onnx',
                          description='ONNX model for inference.',
                          tags={'type': 'classification', 'framework': 'onnx'})

   print('Name:', model.name)
   print('Version:', model.version)
   ```

4. **Environment Setup**: The deployment environment was created and registered with the required dependencies:

   ```python
   from azureml.core import Workspace, Environment
   from azureml.core.conda_dependencies import CondaDependencies

   ws = Workspace.from_config()

   environment_name = 'update-prodclass-onnx-environment'
   environment = Environment(name=environment_name)

   environment.python.conda_dependencies = CondaDependencies.create(
       conda_packages=[
           'python=3.8',
           'pip=20.2.4',
           'numpy',
           'onnx',
           'onnxruntime',
           'pillow',
           'requests'
       ],
       pip_packages=[
           'azureml-defaults',
           'inference-schema[numpy-support]'
       ]
   )

   environment.register(workspace=ws)
   ```

5. **Model Deployment**: The model was deployed as a web service on Azure Container Instances (ACI):

   ```python
   from azureml.core import Workspace, Environment, Model
   from azureml.core.model import InferenceConfig
   from azureml.core.webservice import AciWebservice

   ws = Workspace.from_config()
   environment = Environment.get(ws, name='update-prodclass-onnx-environment')
   model = Model(ws, name='prodtype-classification-best-onnx-model')

   inference_config = InferenceConfig(entry_script='score.py', environment=environment)
   aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

   service_name = 'prodtypeclassifier-service'
   service = Model.deploy(workspace=ws,
                          name=service_name,
                          models=[model],
                          inference_config=inference_config,
                          deployment_config=aci_config,
                          overwrite=True)

   service.wait_for_deployment(show_output=True)
   print(f"Service deployed at: {service.scoring_uri}")
   ```

## API Functionality

The deployed API provides the following features:

### 1. Endpoint Description
- **Scoring URL**: The primary endpoint for making predictions is:
  ```
  http://2673fa3e-f456-46d3-801a-5843044b0ebd.eastus2.azurecontainer.io/score
  ```

### 2. Input Format
- The API accepts input in the form of an image URL.
- **Content-Type**: The request should specify `text/plain` as the content type.
- **Request Method**: The API uses the POST method to handle incoming requests.

### 3. Request Body
- The request body should contain a valid image URL pointing to the image that needs to be classified.

**Example Request:**
```bash
curl -X POST "http://2673fa3e-f456-46d3-801a-5843044b0ebd.eastus2.azurecontainer.io/score" \
-H "Content-Type: text/plain" \
-d "https://example.com/image.png"
```

### 4. Response Format
- The API returns a JSON response containing the top 3 predicted product classes along with their respective probabilities.

**Example Response:**
```json
{
  "top_3_classes_predictions": [
    {
      "class_name": "ENGAGEMENT RINGS",
      "probability": 0.9585813879966736
    },
    {
      "class_name": "WEDDING BANDS",
      "probability": 0.013749789446592331
    },
    {
      "class_name": "BRACELETS",
      "probability": 0.00999149028211832
    }
  ]
}
```

### 5. Error Handling
- The API provides error messages for invalid input formats. If the input is neither a valid image URL nor raw image data, the API will respond with an appropriate error message.

**Example Error Response:**
```json
{
  "error": "Invalid input format. Expected image URL or raw image data."
}
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/kempsly343/Adaptive-product-type-classifier.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Adaptive-product-type-classifier
   ```

3. Follow the deployment steps outlined in the **Model Deployment** section to deploy the model and set up the API.

4. Use the provided curl command to make predictions.

## Requirements

- Python 3.8
- Azure Machine Learning SDK
- ONNX
- ONNX Runtime

## How to Contribute

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

### Customization

- **Service URL**: http://beefc70d-f3a9-4231-80d1-606e3fa1ef2f.eastus2.azurecontainer.io/score
- **Clone URL**: Change the `git clone` URL to your actual GitHub repository link.
