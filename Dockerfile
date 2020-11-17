FROM python:3.7

# Installing git
RUN apt-get update -y
RUN apt-get install -y git
# Installing numpy and jupyterlab
RUN pip install jupyterlab numpy Cython pyarrow

# Installing requirements
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copying examples
COPY ./examples /examples

# Copying this package
COPY . /package
RUN pip install -e /package

# Download models
RUN python -c "from lc_classifier.classifier.models import HierarchicalRandomForest;HierarchicalRandomForest({}).download_model()"

WORKDIR /examples
EXPOSE 8888

CMD ["jupyter", "notebook", "--allow-root", "--ip", "0.0.0.0", "--NotebookApp.token=''"]
