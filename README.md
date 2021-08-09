# ship_or_truck_api

API that can classify Image data into truck or ship

#### Machine Learning **[Python Notebook](https://github.com/nandakishormpai2001/ship_or_truck_api/blob/model/ship_or_truck_api.ipynb)**

## Machine Learning Model

Binary Image classifier Built on PyTorch framework using CNN architecture. Currently Project classifies an Image into Ship or Truck

- Framework : PyTorch ( Version: 1.8.0+cpu )
- Architecture : Convolutional Neural Networks
- Validation Accuracy : 91.8%

#### How to train

Upload the **[Python notebook](https://github.com/nandakishormpai2001/ship_or_truck_api/blob/model/ship_or_truck_api.ipynb)** to Google Colab and run each cell for training the model.

#### How It Works

The input image dataset is converted to tensor and is passed through a CNN model, returning an output as ship(0) or truck(1). Input image tensor is passed through two convolutional layers and then flattened and inputted to fully connected layers.

## API

API is built using Flask framework and hosted in Heroku

- Ship Or Truck Classification

  Accepts a POST request with an image in the form of base64 string and returns a string containing the result as ship or truck

#### How to use

API has been built on this classifier. URL = "https://ship-or-truck-api.herokuapp.com/"

User has to send a POST request to the given api with Base64 string of the Image to be input.

```python
import requests
url = "https://ship-or-truck-api.herokuapp.com/"
#imgdata should be the base64 string of image
r = requests.post(url,json = {"image":imgdata})
print(r.text.strip())
```

Output

```python
'{"result":"ship"}'
```

## Thanks for reading the documentation

Report any issues if found [here](https://github.com/nandakishormpai2001/ship_or_truck_api/issues)
