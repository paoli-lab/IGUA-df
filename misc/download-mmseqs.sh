#!/bin/sh

DIR="$@"
TMP="/tmp"

mkdir -p "$DIR"

if [ $(uname) = "Linux" ]; then
	mkdir -p "$TMP"/mmseqs/mmseqs_avx2 "$TMP"/mmseqs/mmseqs_sse2 "$TMP"/mmseqs/mmseqs_sse41
	wget 'https://github.com/soedinglab/MMseqs2/releases/download/15-6f452/mmseqs-linux-avx2.tar.gz' -O- | tar xz -C "$TMP"/mmseqs/mmseqs_avx2 --strip-components 2
	wget 'https://github.com/soedinglab/MMseqs2/releases/download/15-6f452/mmseqs-linux-sse2.tar.gz' -O- | tar xz -C "$TMP"/mmseqs/mmseqs_sse2 --strip-components 2
	wget 'https://github.com/soedinglab/MMseqs2/releases/download/15-6f452/mmseqs-linux-sse41.tar.gz' -O- | tar xz -C "$TMP"/mmseqs/mmseqs_sse41 --strip-components 2
	wget 'https://github.com/soedinglab/MMseqs2/raw/master/util/mmseqs_wrapper.sh' -O "$TMP"/mmseqs/mmseqs_wrapper.sh
	install -m755 "$TMP"/mmseqs/mmseqs_sse2/mmseqs "$DIR"/mmseqs_sse2
	install -m755 "$TMP"/mmseqs/mmseqs_sse41/mmseqs "$DIR"/mmseqs_sse41
	install -m755 "$TMP"/mmseqs/mmseqs_avx2/mmseqs "$DIR"/mmseqs_avx2
	install -m755 "$TMP"/mmseqs/mmseqs_wrapper.sh "$DIR"/mmseqs
elif [ $(uname) = "Darwin" ]; then
	wget 'https://github.com/soedinglab/MMseqs2/releases/download/15-6f452/mmseqs-osx-universal.tar.gz' -O- | tar xz -C "$DIR" --strip-components 2
else
	echo "Unsupported platform: $(uname)"
	exit 1
fi
