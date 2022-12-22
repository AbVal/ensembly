FROM python:3.8-slim

COPY src/requirements.txt /root/src/requirements.txt

RUN chown -R root:root /root

WORKDIR /root/src
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./
RUN chown -R root:root ./
ENV FLASK_APP server.py

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]