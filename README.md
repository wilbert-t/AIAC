# Energy-model
energy modeling and structural analysis of gold clusters 
docker build -t aiac_env .
docker run -it -p 8888:8888 aiac_env

download dependecy conda and pip 
conda install -c conda-forge --file conda_packages.txt

python -m pip install -r pip_packages.txt

conda deactivate
conda env remove -n aiac

# Create environment with isolated pip
conda create -n aiac_clean python=3.11 --no-default-packages -y
conda activate aiac_clean

# Check paths (pip might still be wrong, but that's okay)
which python  # This should be correct

# List all conda packages in current environment
conda list

# check first step result 
python view_pkl.py 

debug this 
  ✗ task3_sensitivity_analysis.pdf (missing)
  ✗ task3_sensitivity_results.pkl (missing)
  
