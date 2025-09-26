# Impostor hunt in texts

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Summary 🚀

The main task of this competition is to detect fake texts in a given dataset. Each data sample contains two texts - one real and one fake. The data used in this Challenge come from The Messenger journal. Importantly, both texts in each sample - real (optimal for the recipient, as close as possible to the hidden original text) and fake (more or much more distant from the hidden original text) - have been significantly modified using LLMs.

## 🛰️ Architecture

All this project is designed using Kedro. You can find the architecture of the project here: https://bwallyn.github.io/impostor-hunt-in-texts/

### Prepare data

The prepare data pipeline creates the training and test datasets and convert them into a Hugging Face Dataset Dict.

### Feature engineering

The feature engineering pipeline performs augmentation data, creates new features by extracting text info using a NLP model and reconverts it to a dataframe.

### Train model

The train model pipeline conducts the training of a Gradient Boosting model. To do so, it:
- Creates a MLflow experiment or get one if given by the user.
- Initialize the model parameters and create a pydantic object to store them.
- Drop some columns.
- Find the best hyperparameters using bayesian optimization.
- Train a final model using the best hyperparams found previously.


## Set up and contribute

### Set up your environment

To develop on this project, the first part is to set your environment. To do so:
- Create a python virtual environment (if you are using Visual Studio Code, ctrl+shift+p). Don't install the dependencies yet.
- Install `uv` using ```pip install uv```.
- Install the dependencies using ```uv sync --all-extras``` (the --all-extras option installs the dev dependencies).
Now you are all set to contribute to the project.

### Git and branches

We defined some branch policies to protect and contribute in the right way. In a typical Git flow, the following branches are commonly used:
- **`main`**: The main branch contains the production-ready code.
- **`dev`**: The dev branch contains the latest development changes.
- **`feature/*`**: Feature branches are used to develop new features.
- **`release_*`**: Release branches are used to prepare for a new production release.
- **`hotfix_*`**: Hotfix branches are used to quickly fix production issues.

<img src="_assets/git_low_branching.svg" alt="git flow branching strategy" width="600">

### How to contribute

- Create a new branch using the following guidelines:
  - `feature/<branch-name>` to add a feature to the project.
  - `fix/<branch-name>` to fix a bug in the project.
- Code.
- Add unit tests.
- Create a pull request to the `dev` branch.
- Merge and if it validated create a pull request to the `main` branch.


## Kedro

### Overview

This is your new Kedro project, which was generated using `kedro 0.19.12`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

### Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

### How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

### How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

### How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.


### Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

### How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

#### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

#### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

#### IPython
And if you want to run an IPython session:

```
kedro ipython
```

#### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

### Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
