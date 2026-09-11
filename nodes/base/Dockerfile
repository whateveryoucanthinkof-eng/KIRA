FROM docker.io/library/alpine:latest

RUN apk add --no-cache \
    bash \
    python3 \
    py3-pip \
    iproute2 \
    iptables \
    tcpdump \
    curl \
    socat \
    bind-tools \
    jq

WORKDIR /app
COPY nodes/ /app/nodes/
COPY workloads/ /app/workloads/
COPY telemetry/ /app/telemetry/

CMD ["/bin/bash"]
