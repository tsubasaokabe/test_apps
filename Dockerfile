from ubuntu
WORKDIR /workspace

LABEL maintainer Tsubasa Okabe<tokabe@netprotections.co.jp>

RUN apt-get update
RUN apt-get install python3-pip -y

RUN pip3 install numpy pandas sklearn seaborn
RUN pip3 install chainer Flask

EXPOSE 5000
CMD ["python3", "server.py"]