Dlung_Vision
1 create conda environment
We recommend using conda for managing all the packages:
conda create -n dlung_v1 python=3.7
source activate dlung_v1

2 Install dependencies:
pip install git+https://github.com/JoHof/lungmask
pip install pynrrd
pip install pandas
conda install opencv
conda install -c anaconda ipython
conda install -c conda-forge scikit-learn
cd build/box
python setup.py install

3 Preprocess data:
cd data
python filedir.py
cd utils
python preprocess_lidc.py

4 Run inference:
python test_lung.py eval
Then we can get the result in the json: result.json

5 Current Performance:
LUNA16:

fps: 0.125 , sensitivity: 0.8071428571428572
fps: 0.25 , sensitivity: 0.8357142857142857
fps: 0.5 , sensitivity: 0.8928571428571429
fps: 1 , sensitivity: 0.9142857142857143
fps: 2 , sensitivity: 0.95
fps: 4 , sensitivity: 0.9642857142857143
fps: 8 , sensitivity: 0.9642857142857143
