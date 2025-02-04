FROM python:3.9-slim
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
RUN python3 setup.py sdist bdist_wheel
RUN pip install .
ENTRYPOINT ["python3", "spaghetti/cli_inference.py"]
