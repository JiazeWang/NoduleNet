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
put the datafolder under data like CT-Lung
cd data
python filedir.py
cd utils
python preprocess_ha.py

4 Run inference:
cd ..
python test_lung.py eval
Then we can get the result in the json: result.json
