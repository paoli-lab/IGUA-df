FROM alpine:latest as build
ADD . /usr/src/htgcf
RUN apk add --no-cache rust cargo python3 py3-pip py3-wheel py3-numpy py3-scipy py3-h5py py3-pandas py3-build
RUN python3 -m build /usr/src/htgcf --outdir /tmp

FROM alpine:latest as run
COPY --from=0 /tmp/*.whl /tmp
RUN apk add --no-cache python3 py3-pip py3-wheel py3-numpy py3-scipy py3-h5py py3-pandas
RUN python3 -m pip install htgcf --only-binary htgcf --no-index --find-links /tmp
