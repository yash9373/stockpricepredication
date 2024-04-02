from setuptools import find_packages, setup

# Function to read requirements from a file
def get_requirements(file_path: str) -> list[str]:
    requirements = []
    with open(file_path) as file_obj:
        for line in file_obj:
            requirement = line.strip()
            if not requirement.startswith('-e.'):
                requirements.append(requirement)
    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="yash",
    author_email="kolheyashodip8@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
