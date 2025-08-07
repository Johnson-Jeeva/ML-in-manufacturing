import jmp
import jmputils
import importlib

# Upgrade pip and build tools
jmputils.jpip('install --upgrade', 'pip setuptools wheel')

# List of libraries to install 
libraries = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'scikit-learn': 'sklearn',
    'tensorflow': 'tensorflow',
    'xgboost': 'xgboost'
    # Add more libraries here as needed
    # 'lightgbm': 'lightgbm',
    # 'seaborn': 'seaborn'
}

# Install all libraries
for pip_name in libraries:
    jmputils.jpip('install --upgrade', pip_name)

# Check versions
print("\nInstalled library versions:")
for pip_name, import_name in libraries.items():
    try:
        module = importlib.import_module(import_name)
        print(f"{pip_name}: {module.__version__}")
    except Exception as e:
        print(f"{pip_name}: Error - {e}")

print("\nAll libraries have been installed and checked.")
