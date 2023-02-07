# <This script needs to be ran from within s2wav root directory>

# Install core and extra requirements
echo -ne 'Building Dependencies... \r'
pip install -q -r requirements/requirements-core.txt
echo -ne 'Building Dependencies... ########           (33%)\r'
pip install -q -r requirements/requirements-docs.txt
echo -ne 'Building Dependencies... ###########        (66%)\r'
pip install -q -r requirements/requirements-tests.txt
echo -ne 'Building Dependencies... ################   (100%)\r'
echo -ne '\n'

# Install specific converter for building tutorial documentation
conda install pandoc=1.19.2.1 -y

# In install specific version of s2fft 
# (TODO: move this to requirements when deployed to pip)
pip install git+https://github.com/astro-informatics/s2fft.git@feature/precompute

# Build the scattering emulator
pip install -e .