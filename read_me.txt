Installation:

	1. install Anaconda (also add to path)- first time only
	*NOTE: new anaconda version has issues with YML defs, need to use 3.5.1.0 for this to work
			From:
			https://repo.continuum.io/archive/Anaconda3-5.1.0-Windows-x86_64.exe
	 
	2. install pycharm - first time only

	4. configure new project in pycharm
		a. Add new project
		b. 
		c. Define interpeter:
		File --> settings-> project <namr> --> project interpeter --> dropdown list --> show all --> '+' --> add local --> conda environment
		--> new_env --> cooshe location 
		*inside the anaconda install dir there is a folder called 'env', 
		e.g. 
			E:\tomermeg\MachineLearning\Anaconda3\envs
			
	5. Upgrade tensorflow
	conda install tensorflow
	
	Other options that work:
	pip install --upgrade tensorflow==1.5.0
	pip install --upgrade tensorflow==1.9.0
	
How to run:
	1. open pyCharm
	2. Load the desired project
	3. Right click on the <main>.py file and click on "run " option
	
to use tensorboard explicitly from the termial:
"C:\Users\tomermeged\Anaconda3\envs\tensorflow\Scripts\tensorboard.exe" --logdir="Users\tomermeged\Dropbox\projects\MachineLearning\Lynda\Ex_Files_TensorFlow\Exercise Files\04\logs"
"C:\Users\tomermeged\Anaconda3\envs\tensorflow\Scripts\tensorboard.exe" --logdir="06\logs"

"E:\tomermeg\MachineLearning\Anaconda3\envs\NN-lab1\Scripts\tensorboard.exe" --logdir "C:\Users\tomermeged\Dropbox\projects\MachineLearning\NN-lab\BestPics\best_models\Model_20180510_195403\logs\TB"
    
---------------------------
using git: https://github.com/tomermeged/NN-lab
	1.  open 'Git Bash' app
	2. cd "C:\Users\tomermeged\Dropbox\projects\MachineLearning\NN-lab"
	3. now you can use all the known git cmd: 
	    git status
	    git add <filenames>
	    git commit -m "<commit name>"
	    git push origin master
	4. if git push doesn't work, check the ssh keys
		a.  check for existing keys:
		 ls -al ~/.ssh
		b. display current key:
		cat ~/.ssh/id_rsa.pub
		c. copy paste into the github wesite:
		settings --> SSH and GPG keys --> SSH keys --> new SSH key

    
---------------------------           
to install additional python packages:

use the anaconda prompt

Upgrading Conda:
conda update -n base conda

Upgrading packages:
pip install pip --upgrade
pip install --ignore-installed --upgrade tensorflow
pip install pandas --upgrade
pip install matplotlib --upgrade
pip install tensorflow --upgrade
pip install gensim --upgrade
pip install -U scikit-learn

Pip install Gym (openAI - reinforcement learning "gym.openai.com")

conda install pandas
conda install -c conda-forge google-api-python-client



