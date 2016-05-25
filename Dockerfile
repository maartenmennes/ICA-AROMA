# Installs ICA-AROMA to a centos image with FSL pre-installed
FROM mcin/centos-fsl-5:latest

# Install necessary python packages
RUN yum update -y; yum clean all
RUN yum install -y numpy scipy

# Add everything to the container
ADD . /ICA-AROMA

