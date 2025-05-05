FROM alpine:3.21.3 as build
ADD . /usr/src/igua
RUN sh /usr/src/igua/misc/download-mmseqs.sh /usr/local/bin
RUN apk add --no-cache rust cargo python3 py3-pip
RUN python3 -m pip wheel gb-io -v --wheel-dir /tmp --no-deps
RUN python3 -m pip wheel /usr/src/igua --wheel-dir /tmp --no-deps

FROM alpine:3.21.3 as run
COPY --from=0 /tmp/*.whl /tmp/
COPY --from=0 /usr/local/bin/mmseqs* /usr/local/bin/
RUN apk add --no-cache python3 py3-pip py3-wheel py3-numpy py3-scipy py3-h5py py3-pandas py3-biopython
RUN python3 -m pip install igua --only-binary :all: --find-links /tmp --break-system-packages

VOLUME "/workdir"
VOLUME "/io"
WORKDIR "/io"
ENTRYPOINT ["igua", "-w", "/workdir"]
CMD ["--help"]
