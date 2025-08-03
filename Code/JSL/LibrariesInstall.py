import jmp
import jmputils

# Upgrade pip and setuptools first
jmputils.jpip('install --upgrade', 'pip setuptools wheel')

# Install all essential libraries for ML models used in JMP scripts
required_libraries = [
    'numpy',
    'pandas',
    'matplotlib',
    'scikit-learn',
    'tensorflow',
    'xgboost'
]

# Install each library
for lib in required_libraries:
    jmputils.jpip('install --upgrade', lib)

print("All required libraries for ML models have been installed/updated successfully.")
