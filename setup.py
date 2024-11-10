from setuptools import setup


def read_requirements(path='./requirements.txt'):
    with open(path, encoding='utf-8', errors='ignore') as file:
        install_requires = file.readlines()

    return install_requires


setup(
    name="MMTracker",
    version="0.1.0",
    author="Hamid Mohammadi",
    author_email="sandstormeatwo@gmail.com",
    description="A tool to manually assign tracking id to objects detected by an mmpose detector with some assisted tracking",
    packages=['mm_tracker'],
    scripts=[
        'mmtracker'
    ],
    package_data={'': ['template_config.yml']},
    include_package_data=True,
    install_requires=read_requirements()
)