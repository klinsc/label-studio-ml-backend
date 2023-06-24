# create image of python 3.9
FROM python:3.9

# set working directory
WORKDIR /label-studio-ml-backend

# copy all files from current directory to working directory
COPY . .

# install dependencies
RUN --mount=type=cache,target=/root/.cache \
    pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    pip install label-studio-ml==1.0.9
RUN --mount=type=cache,target=/root/.cache \
    pip install -r label_studio_ml/examples/yolov5/requirements.txt
RUN label-studio-ml init my_ml_backend --script label_studio_ml/examples/yolov5/yolov5.py --force

# command to run on container start
# label-studio-ml start .\my_ml_backend
CMD [ "label-studio-ml", "start", "my_ml_backend" ]

# docker build -t klinsc/label-studio-ml .
# docker run -it -p 9090:9090 label-studio-ml
# docker run -it -p 9090:9090 -v /home/label-studio-ml:/app label-studio-ml
# docker run -it -p 9090:9090 -v /home/label-studio-ml:/app label-studio-ml start my_ml_backend --script label_studio_ml/examples/yolov5/yolov5.py --force

