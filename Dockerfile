FROM pytorch/pytorch
COPY . /deep-ltl
WORKDIR /deep-ltl/src/envs/zones/safety-gymnasium
RUN pip install -e .
WORKDIR /deep-ltl
RUN pip install -r requirements.txt
RUN apt update && apt install -y openjdk-11-jre