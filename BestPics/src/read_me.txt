Installation:

1. install Anaconda - first time only
2. From anaconda prompt: (or CMD)
   *if env is allready set, 
   you can use the anaconda navigator app to 
   open the prompt with the correct env activated
   
   if not go and create the env:
	a.  cd "C:\Users\tomermeged\Dropbox\projects\MachineLearning\Udemy\complete-guide-to-tensorflow-for-deep-learning-with-python\Tensorflow-Bootcamp-master"
	b.  conda env create -f tfdl_env.yml - first time only!!
	c.  Once the env is done - open anaconda prompt from the 
        anaconda navigator app with the correct env activated
    
    
2. install pycharm - first time only

3. configure new project in pycharm
	a. project location
	b. interperter - Conda - new env
	c. install tensorFlow in the same env location
	
4. How to run:
	a. Load the desired project
	b. Right click on the <main>.py file and click on "run " option
	
to use tensorboard explicitly from the termial:
"C:\Users\tomermeged\Anaconda3\envs\tensorflow\Scripts\tensorboard.exe" --logdir="Users\tomermeged\Dropbox\projects\MachineLearning\Lynda\Ex_Files_TensorFlow\Exercise Files\04\logs"
"C:\Users\tomermeged\Anaconda3\envs\tensorflow\Scripts\tensorboard.exe" --logdir="06\logs"
    
---------------------------
using git:
1. open 'Git Bash' app
2. cd "C:\Users\tomermeged\Dropbox\projects\MachineLearning\NN-lab"
3. now you can use all the known git cmd: 
    git status
    git add <filenames>
    git commit -m "<commit name>"
    git push origin master
4. if git push doesn't work, check the ssh keys
    a. check for existing keys:
        ls -al ~/.ssh
    b. display current key:
        cat ~/.ssh/id_rsa.pub
    c. copy paste into the github wesite:
        settings --> SSH and GPG keys --> SSH keys --> new SSH key

    
---------------------------
to install additional python packages - use the anaconda prompt



From Udemy course:
	1. Install Anaconda - first time only!!
	2. From anaconda prompt: (or CMD)
		a. cd "C:\Users\tomermeged\Dropbox\projects\MachineLearning\Udemy\complete-guide-to-tensorflow-for-deep-learning-with-python\Tensorflow-Bootcamp-master"
		b. conda env create -f tfdl_env.yml  - first time only!!
		c. Once the env is done 
			activate tfdeeplearning
			** Make sure you are not inside another env first by typing 'deactivate' **
			 e.g. 
				<tensorflow> c:\users\tomermeged\Dropbox\projects\MachineLearning\Udemy\complete-guide-to-tensorflow-for-deep-learning-with-python\Tensorflow-Bootcamp-master
				Meaning <tensorflow1> is currently activated 
		d. Open jupyter notebook in browser: (from the terminal)
			jupyter notebook 
			

            
to install additional python packages - use the anaconda prompt
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
