FROM tensorflow/tensorflow:2.2.0rc2-py3-jupyter

# Install Amazon SageMaker training toolkit and smdebug libraries
RUN pip install tensorflow
RUN pip install sagemaker-training
RUN pip install smdebug

COPY agent.py /opt/ml/code/agent.py
COPY get_data.py /opt/ml/code/get_data.py
COPY learn.py /opt/ml/code/learn.py
COPY metrics.py /opt/ml/code/metrics.py
COPY rewards.txt /opt/ml/code/rewards.txt

ENV SAGEMAKER_PROGRAM agent.py
