FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 \
        python3 \
    && rm -rf /var/lib/apt/lists/*

# Intel GPU drivers (place .deb files in drivers/ next to this Dockerfile)
# Default location: /home/gta/chuansheng/drivers/
COPY drivers/ /tmp/drivers/
RUN dpkg -i /tmp/drivers/*.deb && rm -rf /tmp/drivers/

COPY Release/ /opt/ov_serve/

WORKDIR /opt/ov_serve
EXPOSE 8080

ENTRYPOINT ["python3", "launch_ov_serve.py"]
CMD ["--port", "8080"]
