# mmpose_manual_tracker
A tool to manually assign tracking id to objects detected by an mmpose detector with some assisted tracking


## Installation

To be able to install `mmcv` install cuda-toolkit according to your GPU driver version. For example, for driver version 535 if you are using conda you can install cuda-toolkit 12.1.1 as follows:

```bash
conda install nvidia/label/cuda-12.1.1::cuda-toolkit
```

Then if installing directly from Github:

```bash
python3 -m pip install git+https://github.com/sandstorm12/mmpose_manual_tracker.git
```

Or if you are using a cloned verion, go to the cloned folder and install:

```bash
python3 -m pip install .
```


## Usage

Check your installation by:

```python
mmtracker -h
```

First generate the sample config file:

```python
mmtracker -g
```

This will create a config file in your current directory. Change the parameters for your use.

First apply the detector to obtain the bounding boxes:

```python
mmtracker -d -c config.yml
```

Then start the tracking by:

```python
mmtracker -t -c config.yml
```

## Controls

`e`: go to next frame
`q`: go to previouse frame
`s`: quit after saving the current tracking ids to the specified output
`0-9`: first press selects the bounding box and second press assigns a tracking id


## Contributions

1. Hamid Mohammadi <sandstormeatwo@gmail.com>
