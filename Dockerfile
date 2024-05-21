FROM python:3.9
RUN apt-get update
RUN pip3 install --no-cache-dir fastapi==0.110.3
RUN pip3 install --no-cache-dir torch==2.3.0
RUN pip3 install --no-cache-dir torchvision==0.18.0
RUN pip3 install --no-cache-dir torchaudio==2.3.0
RUN pip3 install --no-cache-dir jinja2==3.1.4
RUN pip3 install --no-cache-dir python-multipart==0.0.9
RUN pip3 install --no-cache-dir 'uvicorn[standard]'

RUN mkdir /workspace/
COPY . /workspace/

WORKDIR /workspace

ENTRYPOINT ["uvicorn"]
CMD ["--host=0.0.0.0", "--port=80", "main:app"] 

