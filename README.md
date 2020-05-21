
# Googly project

---
The objective is to create a service that "googlifies" (or, in layman terms, to convert human eyes in googly, fun eyes) every image taken as an input.

This README has the following sections:
1. Getting Started
2. Setting up
3. Invoking the endpoints
4. Relevant considerations

Each image's prediction seems to take around 30 to 40 ms to run (results obtained in _notebooks/basic.ipynb_)

---
## Getting started
#### Create a conda environment
For the sake of reproducibility, we'll use conda to create an environment.

```
conda create --name googly python=3.8

conda activate googly
```

####Install the project requirements
 
 (Note: one of the libraries, dlib, depends on cmake.
  If your operative system is MacOS, you can install it with ```brew install cmake```)
```
python -m pip install -r requirements.txt
```
#### Add the repository's root folder to PYTHONPATH

```PYTHONPATH``` is necessary, as throughout the service the modules' calls take in account absolute paths,
 rooting from the```src``` folder.

```
export PYTHONPATH="$PYTHONPATH:."
```

####Run test coverage for the whole service
To run the tests and obtain the coverage, you can run
```
pytest --cov-report term-missing --cov=src/ tests/ --cov-fail-under=100
```
---

## Setting up - How to run this service?
There are two ways to run this service: Locally, and by running it as a Docker container, and invoking the POST endpoint.

To start the service locally, it's as simple as running the supposed script:

```
python src/server.py
```

Alternatively, to build and run the Docker container, run the following instructions (it might take a while, as it'll install dlib, which is a weighty distribution):

```
docker build -t googly:latest .
docker run -v ~/path/to/upload/googly_images:/app/googly_images -p 5000:5000 googly
```
As you might have noticed, we are adding a volume to our container. This will be the destination of the googly images!

---
## Invoking the endpoints

Now, the service is running! As you might have noticed in ```views.py```, there are two possible endpoints.
They both follow the use case specified, although with different approaches:

- **/googlify** - takes in the image as multipart/form-data, and saves the resulting googlified picture in a specified directory.
- **/googlify_base64** - takes the image in the form of a base64 string, and returns the image as a base64 string.

#### First endpoint - Invoking /googlify
*Input*:

To invoke the first endpoint, it's as simple as running the following code:
```
curl -F "file=@path/to/your/file.png" http://localhost:5000/googlify
```
This will add the image to the googly_images folder, where you can peruse the googlified images!

*Output:*

Locally, the destination folder can't be modified without accessing the code (CONFIG_DIR variable in ```views_utils```).
But, when we run the service as a container, we bind the googly_folder to an external path of the user's liking,
providing the freedom to decide where the googlified pictures will be added.

#### Second endpoint - Invoking /googlify_base64
*Input:*

This second endpoint comes as a solution entirely deprived of actual files! You can invoke the endpoint through the following command:
```
curl -H "Content-Type: application/json" -d '{"img":"yourbase64string"}' http://localhost:5000/googlify_base64
```
To make the user's life easier, you can find downwards a one-liner that would take the path of a picture, convert it into base64
and invoke the endpoint with it:
```
(echo -n '{"img": "'; base64 ~/path/to/your/image.png; echo '"}') | curl -H "Content-Type: application/json" -d @- http://localhost:5000/googlify_base64
```

*Output:*

The user obtains a base64 string as a result.
 
You can decode the base64 string into an image here: https://codebeautify.org/base64-to-image-converter


## Relevant considerations
1. There's a config file! In ```config.yaml```, feel free to play with the variables inside ```service```.
These variables do the following:
    - **confidence_threshold** - between 0 and 1, how confident we need to be to validate a detected face. Default: 0.5 (50%)
    - **size_multiplier** - considering the actual human eye's size, how many times bigger do we want the googly eye to be? Default: 2 (two times the fun!)
    - **random_max_percent_inc** - every googly eye's size is subject to a random increase! This defines the maximum threshold that the eye can increase in size, relative to the original googly eye. Default: 0.2 (20%)
    - **googly_path** - the path for the googly eye's image! There's two googly options for now.

2. To test the code programmatically, a jupyter notebook can be found and booted up in the ```notebooks``` folder. You can also find images to test the code with here!

3. Models and config taken from the following sources:
    - [eye_eyebrows_22.dat](https://github.com/Luca96/dlib-minified-models/tree/master/face_landmarks)
    - [deploy.prototxt and res10](https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison/models)

## Authors

* **Ricardo Soares** - [rmssoares](https://github.com/rmssoares)

For any inquiries, feel free to open up an issue.