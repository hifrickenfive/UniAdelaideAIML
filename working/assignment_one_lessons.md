# To install pytorch 1.2
Pytorch 1.2 is really old. I had to do a lot of faffing around.
1. Clone the anaconda base environment (360 packages)
1. Deprecate python from 3.8.8 to 3.7. 

However this attempt totally failed. There were way too many package conflicts after deprecating python.

> ```conda install -n <environment_name> python=3.7```
[See this stackoverflow link](https://stackoverflow.com/questions/24664072/how-do-i-clone-a-conda-environment-from-one-python-release-to-another)

So the only solution is to create a clean environment with python 3.7 and conda install pytorch
> ```conda install pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch```
[pytorch archived versions](https://pytorch.org/get-started/previous-versions/)

# Markdown cheatsheet
And by the way I always need a [markdown cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#links).