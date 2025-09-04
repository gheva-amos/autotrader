FROM agheva-base-image:1.0

RUN yum install -y cmake gdb python3-devel protobuf-devel python3-pip

WORKDIR /mnt/src
CMD ["sleep", "99d"]
