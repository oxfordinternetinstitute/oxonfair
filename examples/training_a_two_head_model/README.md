## Setup for Training a Two-Head Model

1) Download img_align_celeba.zip and list_attr_celeba.txt from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
and move it to data/celeba/.

2) Create a virtual environment with python=3.8, and activate it. 

3) Install your torch and torchvision. We used torch==1.12.1+cu116 and torchvision==0.13.1+cu116.

````commandline
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
````

4) Install other minimal requirements.
   ````
   python -m pip install -r requirements.txt
   ````

5) Run the python script.
    ````
   python two_head_model_demo.py --target_class 9 --protected_class 20
   
