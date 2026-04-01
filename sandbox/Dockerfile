# Minimal sandbox image for shell.exec tool
# Commands run inside this container are isolated from the host
FROM alpine:3.19

# Install commonly needed tools
RUN apk add --no-cache \
    bash \
    curl \
    wget \
    git \
    jq \
    python3 \
    py3-pip \
    nodejs \
    npm

# Non-root user for extra safety
RUN adduser -D -u 1000 sandbox
USER sandbox
WORKDIR /workspace

CMD ["sh"]
