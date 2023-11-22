FROM alpine:latest as build
ADD . /usr/src/htgcf
RUN apk add --no-cache rust cargo python3 py3-pip
RUN python3 -m pip wheel gb-io -v --wheel-dir /tmp --no-deps
RUN python3 -m pip wheel /usr/src/htgcf --wheel-dir /tmp --no-deps

FROM alpine:latest as run
COPY --from=0 /tmp/*.whl /tmp
RUN apk add --no-cache python3 py3-pip py3-wheel py3-numpy py3-scipy py3-h5py py3-pandas
RUN python3 -m pip install htgcf --only-binary :all: --find-links /tmp

RUN wget 'https://github.com/soedinglab/MMseqs2/releases/download/15-6f452/mmseqs-linux-avx2.tar.gz' -O- | tar xz -C /usr/local/ --strip-components 1 \
	&& mv /usr/local/bin/mmseqs /usr/local/bin/mmseqs_avx2 \
	&& wget 'https://github.com/soedinglab/MMseqs2/releases/download/15-6f452/mmseqs-linux-sse41.tar.gz' -O- | tar xz -C /usr/local/ --strip-components 1 \
	&& mv /usr/local/bin/mmseqs /usr/local/bin/mmseqs_sse41 \
	&& wget 'https://github.com/soedinglab/MMseqs2/releases/download/15-6f452/mmseqs-linux-sse2.tar.gz' -O- | tar xz -C /usr/local/ --strip-components 1 \
	&& mv /usr/local/bin/mmseqs /usr/local/bin/mmseqs_sse2 \
	&& wget 'https://github.com/soedinglab/MMseqs2/raw/master/util/mmseqs_wrapper.sh' -O /bin/mmseqs \
	&& chmod 777 /bin/mmseqs

VOLUME "/io"
WORKDIR "/io"
ENTRYPOINT ["htgcf"]
CMD ["--help"]
