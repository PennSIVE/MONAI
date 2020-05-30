FROM projectmonai/monai:latest
RUN apt-get update && apt-get install -y libcairo2-dev libgirepository1.0-dev python3-gi gobject-introspection gir1.2-gtk-3.0 && \
    pip install pycairo pygobject
WORKDIR /src
COPY . .
# make sure HOME is set properly for singularity
ENV HOME=/root